"""
Full definition of a GPT Language Model with Mixture-of-Recursions (MoR).
Supports:
- Dynamic routing (Expert-Choice / Token-Choice)
- Two KV caching strategies for inference: recursion-wise and recursive sharing
- Training-time sequence compression (Gather-Scatter) for FLOPs and memory savings
  with explicit rho (sorted gather indices) semantics
- Inference prefill/decode writes K/V into the loop-wise KV cache at original
  positions via explicit position (rho) bookkeeping
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
#  Dynamic Routing Mechanisms
# =============================================================================

class BaseRouter(nn.Module):
    def __init__(self, hidden_size: int, router_type: str = "linear"):
        super().__init__()
        self.hidden_size = hidden_size
        self.router_type = router_type
        if router_type == "linear":
            self.router = nn.Linear(hidden_size, 1, bias=False)
        elif router_type == "mlp":
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.router(hidden_states)


class ExpertChoiceRouter(BaseRouter):
    def __init__(self, hidden_size: int, router_type: str = "linear",
                 beta_percentile: float = 0.8, auxiliary_loss_weight: float = 0.01):
        super().__init__(hidden_size, router_type)
        self.beta_percentile = beta_percentile
        self.auxiliary_loss_weight = auxiliary_loss_weight

    def compute_threshold(self, scores: torch.Tensor) -> torch.Tensor:
        flat_scores = scores.view(-1)
        k = int(self.beta_percentile * flat_scores.numel())
        threshold, _ = torch.kthvalue(flat_scores, k)
        return threshold

    def forward(self, hidden_states: torch.Tensor, recursion_step: int,
                previous_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_scores = self.router(hidden_states).squeeze(-1)
        routing_scores = torch.sigmoid(raw_scores)
        threshold = self.compute_threshold(routing_scores)
        selection_mask = (routing_scores >= threshold).float()
        if previous_mask is not None:
            selection_mask = selection_mask * previous_mask
        auxiliary_loss = self._compute_auxiliary_loss(routing_scores, selection_mask)
        return routing_scores, selection_mask, auxiliary_loss

    def _compute_auxiliary_loss(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = scores.shape
        if seq_len > 1:
            mask_diff = torch.diff(mask, dim=-1)
            consistency_loss = torch.abs(mask_diff).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=mask.device)
        selection_rate = mask.mean()
        sparsity_loss = F.relu(0.1 - selection_rate)
        over_selection_loss = F.relu(selection_rate - 0.9)
        auxiliary_loss = (consistency_loss + sparsity_loss + over_selection_loss) * self.auxiliary_loss_weight
        return auxiliary_loss


class TokenChoiceRouter(BaseRouter):
    def __init__(self, hidden_size: int, max_recursion_depth: int, router_type: str = "linear"):
        super().__init__(hidden_size, router_type)
        self.max_recursion_depth = max_recursion_depth
        if router_type == "linear":
            self.router = nn.Linear(hidden_size, max_recursion_depth)
        elif router_type == "mlp":
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, max_recursion_depth)
            )

    def forward(self, hidden_states: torch.Tensor, current_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_logits = self.router(hidden_states)
        depth_probs = F.softmax(depth_logits, dim=-1)
        continue_probs = depth_probs[:, :, current_depth+1:].sum(dim=-1)
        continue_mask = (continue_probs > 0.5).float()
        return depth_probs, continue_mask


# =============================================================================
#  KV Cache (for inference only)
# =============================================================================

class MoRKVCache:
    """
    KV cache for inference supporting two modes:
    - 'recursion_wise': each recursion step maintains its own cache.
    - 'recursive_share': only step 0 is stored, reused for all steps.
    All writes/reads are indexed by ORIGINAL sequence positions (rho), so that
    future tokens can attend to the correct causal entries.
    """
    def __init__(self, mode: str, max_batch_size: int, max_seq_len: int,
                 num_heads: int, head_dim: int, num_layers: int,
                 num_recursion_steps: int, device: torch.device):
        self.mode = mode
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_recursion_steps = num_recursion_steps
        self.device = device

        self.caches = {}
        self.lengths = {}
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        for step in range(num_recursion_steps):
            for layer in range(num_layers):
                self.caches[(step, layer)] = {
                    'key': torch.zeros(cache_shape, dtype=torch.float16, device=device),
                    'value': torch.zeros(cache_shape, dtype=torch.float16, device=device)
                }
                self.lengths[(step, layer)] = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def get_cache(self, step: int, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == 'recursive_share':
            step = 0
        return (self.caches[(step, layer)]['key'],
                self.caches[(step, layer)]['value'],
                self.lengths[(step, layer)])

    def update_cache(self, step: int, layer: int,
                     key_states: torch.Tensor, value_states: torch.Tensor,
                     routing_mask: Optional[torch.Tensor] = None,
                     positions: Optional[torch.Tensor] = None):
        """
        positions: [B, S] original sequence positions (rho) of the compact
        inputs -- scatter K/V back to their original cache slots.
        """
        if self.mode == 'recursive_share':
            step = 0

        batch_size, num_heads, seq_len, head_dim = key_states.shape
        if positions is None:
            # Fallback: sequential append (no gather happened)
            current_len = self.lengths[(step, layer)][0].item()
            end = min(current_len + seq_len, self.max_seq_len)
            actual_seq_len = end - current_len
            if routing_mask is not None:
                mask_expanded = routing_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=key_states.dtype)
                key_states = key_states * mask_expanded
                value_states = value_states * mask_expanded
            self.caches[(step, layer)]['key'][:batch_size, :, current_len:end] = key_states[:, :, :actual_seq_len]
            self.caches[(step, layer)]['value'][:batch_size, :, current_len:end] = value_states[:, :, :actual_seq_len]
            self.lengths[(step, layer)][:batch_size] = end
        else:
            # Explicit rho: write each compact row back to its ORIGINAL position
            for b in range(batch_size):
                for t in range(seq_len):
                    pos = positions[b, t].item()
                    if pos < self.max_seq_len:
                        if routing_mask is None or routing_mask[b, t] > 0:
                            self.caches[(step, layer)]['key'][b, :, pos:pos+1] = key_states[b, :, t:t+1]
                            self.caches[(step, layer)]['value'][b, :, pos:pos+1] = value_states[b, :, t:t+1]
            max_pos = positions.max().item() + 1
            self.lengths[(step, layer)][:batch_size] = max(max_pos, self.lengths[(step, layer)][0].item())

    def clear(self):
        for step in range(self.num_recursion_steps):
            for layer in range(self.num_layers):
                self.caches[(step, layer)]['key'].zero_()
                self.caches[(step, layer)]['value'].zero_()
                self.lengths[(step, layer)].zero_()


# =============================================================================
#  LoopFormer Components
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x,
                cache: Optional[MoRKVCache] = None,
                step_idx: Optional[int] = None,
                layer_idx: Optional[int] = None,
                routing_mask: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None):
        """
        x: [B, T, C]. During training gather-scatter, x is a gathered compact
        tensor whose rows correspond to original positions `positions` (rho).
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        use_cache = (cache is not None and step_idx is not None and layer_idx is not None)

        if use_cache:
            # Scatter newly computed K/V into cache slots at ORIGINAL
            # positions (rho). During single-token decode, positions collapse
            # to the identity and this reduces to tail concatenation.
            cache.update_cache(step_idx, layer_idx, k, v,
                               routing_mask=routing_mask, positions=positions)
            cached_k, cached_v, length = cache.get_cache(step_idx, layer_idx)
            full_len = length[0].item()
            full_k = cached_k[:, :, :full_len]
            full_v = cached_v[:, :, :full_len]
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, full_k, full_v,
                    attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False
                )
            else:
                att = (q @ full_k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ full_v
        else:
            # Training / prefill without cache: causal attention over the
            # current (compact) tensor. For gathered subsets of the sequence,
            # this applies causal masking in the compact coordinate system.
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.intermediate_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.intermediate_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LoopFormerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.mlp = MLP(config)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cache: Optional[MoRKVCache] = None,
                step_idx: Optional[int] = None,
                layer_idx: Optional[int] = None,
                routing_mask: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate_msa, gate_mlp, scale_msa, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)

        attn_out = self.attn(
            self.norm_1(x) * (1 + scale_msa.unsqueeze(1)),
            cache=cache, step_idx=step_idx, layer_idx=layer_idx,
            routing_mask=routing_mask, positions=positions
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_out = self.mlp(
            self.norm_2(x) * (1 + scale_mlp.unsqueeze(1))
        )
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class SharedBlock(nn.Module):
    def __init__(self, depth, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            LoopFormerBlock(config) for _ in range(depth)
        ])

    def forward(self, x, c,
                cache: Optional[MoRKVCache] = None,
                step_idx: Optional[int] = None,
                routing_mask: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None):
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, c, cache=cache, step_idx=step_idx, layer_idx=layer_idx,
                      routing_mask=routing_mask, positions=positions)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# =============================================================================
#  GPT / MoR Model with Gather-Scatter training
# =============================================================================

@dataclass
class GPTConfig:
    model_type: str = 'mor_loopformer'
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 3          # layers per recursion block
    n_head: int = 32
    n_embd: int = 2048
    dropout: float = 0.0
    bias: bool = False
    intermediate_dim: int = 5120

    # MoR specific
    router_type: Optional[str] = 'token_choice'          # 'expert_choice' or 'token_choice'
    router_beta: float = 0.8
    router_aux_weight: float = 0.01
    max_recursion_depth: int = 8
    cache_mode: Optional[str] = None           # 'recursion_wise' or 'recursive_share' (inference only)
    use_cache: bool = False                    # enable cache during generation


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = SharedBlock(config.n_layer, config),
            norm_f = nn.RMSNorm(config.n_embd),
        ))

        self.time_embedder = TimestepEmbedder(config.n_embd)
        self.dt_embedder = TimestepEmbedder(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Router
        self.router = None
        if config.router_type == 'expert_choice':
            self.router = ExpertChoiceRouter(
                config.n_embd,
                beta_percentile=config.router_beta,
                auxiliary_loss_weight=config.router_aux_weight
            )
        elif config.router_type == 'token_choice':
            self.router = TokenChoiceRouter(
                config.n_embd,
                max_recursion_depth=config.max_recursion_depth
            )
        elif config.router_type is not None:
            raise ValueError(f"Unsupported router_type: {config.router_type}")

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, steps=None,
                cache: Optional[MoRKVCache] = None,
                positions: Optional[torch.Tensor] = None):
        """
        Training (targets not None):
            Gather-Scatter per loop:
              1) GATHER  : compact tensor of active tokens, rho^(i) = sorted
                           active positions (kept for scatter-back)
              2) ROUTE   : routing decision on the compact tensor
              3) COMPUTE : shared block on kept tokens only
              4) SCATTER : index_put updated states back to H at rho
              5) UPDATE  : active_mask / prev_mask by rho
        Inference (targets is None):
            If cache is None, normal forward (eval).
            If cache is provided, only processes new tokens (T=1, rho = pos)
            and writes K/V into the loop-wise cache at original positions.
        """
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")

        if steps is None:
            steps = [1/8] * self.config.max_recursion_depth

        # Position embeddings
        if positions is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)
        else:
            pos = positions

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)   # [B, T, C] full-length roster H

        # ============================================================
        #   Training / prefill (no cache): Gather-Scatter compression
        # ============================================================
        if targets is not None or cache is None:
            # active_mask: [B, T] bool, True = token still in recursion
            active_mask = torch.ones(b, t, dtype=torch.bool, device=device)
            # prev_mask_full: [B, T] float, hierarchical mask for ExpertChoice
            prev_mask_full = torch.ones(b, t, dtype=torch.float, device=device)

            aux_losses = []
            ti = torch.zeros(b, dtype=x.dtype, device=x.device)

            for step_idx, dt in enumerate(steps):
                dt_base = torch.ones_like(ti) * dt
                te = self.time_embedder(ti)
                dte = self.dt_embedder(dt_base)
                c = te + dte   # [B, C]

                # advance timestep once per loop (before any early `continue`,
                # so all batches stay time-synchronized)
                ti = ti + dt

                for batch_idx in range(b):
                    # ------------------------------------------------------
                    # 1) GATHER: rho^(i) = sorted active positions
                    # ------------------------------------------------------
                    active_positions = torch.where(active_mask[batch_idx])[0]   # sorted => rho^(i)
                    num_active = active_positions.size(0)
                    if num_active == 0:
                        continue

                    x_b = x[batch_idx]                                          # [T, C]
                    x_active = x_b.index_select(0, active_positions)            # [A, C]  <-- GATHER
                    prev_mask = prev_mask_full[batch_idx].index_select(0, active_positions)

                    # ------------------------------------------------------
                    # 2) ROUTE on the gathered compact tensor
                    # ------------------------------------------------------
                    x_active_b = x_active.unsqueeze(0)                          # [1, A, C]
                    if isinstance(self.router, ExpertChoiceRouter):
                        _, mask, aux_loss = self.router(x_active_b, step_idx, prev_mask)
                        aux_losses.append(aux_loss)
                    elif isinstance(self.router, TokenChoiceRouter):
                        _, mask = self.router(x_active_b, step_idx)
                    else:
                        mask = torch.ones_like(x_active_b[:, :, 0])
                    routing_mask = mask.squeeze(0)                              # [A]
                    keep_mask = routing_mask > 0.5                              # bool [A]

                    # ------------------------------------------------------
                    # 3) COMPUTE shared block on kept tokens only
                    # ------------------------------------------------------
                    if keep_mask.any():
                        keep_local_idx = keep_mask.nonzero(as_tuple=True)[0]    # [K] (sorted)
                        # keep_positions is the rho^(i) of the kept tokens:
                        # sorted subset of active_positions, order preserved
                        keep_positions = active_positions.index_select(0, keep_local_idx)
                        x_keep = x_active.index_select(0, keep_local_idx)       # [K, C] <-- inner GATHER

                        c_b = c[batch_idx:batch_idx + 1, :]                     # [1, C]
                        x_updated = self.transformer.h(
                            x_keep.unsqueeze(0), c_b,
                            cache=None, step_idx=step_idx,
                            routing_mask=None, positions=None
                        ).squeeze(0)                                            # [K, C]

                        # --------------------------------------------------
                        # 4) SCATTER back to H at original positions (rho)
                        #    index_put is autograd-safe: gradients flow back
                        #    along this indexed assignment to x_updated
                        # --------------------------------------------------
                        batch_col = torch.full((keep_positions.numel(),), batch_idx,
                                               dtype=torch.long, device=device)
                        x = x.index_put((batch_col, keep_positions), x_updated)

                        # --------------------------------------------------
                        # 5) UPDATE active set / prev mask at rho
                        # --------------------------------------------------
                        row = active_mask[batch_idx].scatter(0, active_positions, False)
                        row = row.scatter(0, keep_positions, True)
                        active_mask[batch_idx] = row

                        prev_row = prev_mask_full[batch_idx].index_put(
                            (active_positions,), routing_mask
                        )
                        prev_mask_full[batch_idx] = prev_row
                    else:
                        # nobody keeps: all currently-active tokens exit
                        row = active_mask[batch_idx].scatter(0, active_positions, False)
                        active_mask[batch_idx] = row
                        prev_row = prev_mask_full[batch_idx].index_put(
                            (active_positions,), routing_mask
                        )
                        prev_mask_full[batch_idx] = prev_row

            # After recursion, apply final norm.
            # H now holds each token's latest state at its original position,
            # so loss alignment below is positional by construction.
            x = self.transformer.norm_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                if aux_losses:
                    loss = loss + sum(aux_losses)
                return logits, loss, x
            else:
                # Prefill without targets (e.g., during generate prefill)
                logits = self.lm_head(x[:, -1:, :])
                loss = None
                return logits, loss, x

        # ============================================================
        #   Inference with cache (targets is None and cache is not None)
        #   Single-token decode: gather/scatter degenerate to identity /
        #   in-place write; rho collapses to the current position.
        # ============================================================
        else:
            aux_losses = []
            previous_mask = None
            ti = torch.zeros(b, dtype=x.dtype, device=x.device)

            for step_idx, dt in enumerate(steps):
                dt_base = torch.ones_like(ti) * dt
                te = self.time_embedder(ti)
                dte = self.dt_embedder(dt_base)
                c = te + dte

                ti = ti + dt

                if self.router is not None:
                    if isinstance(self.router, ExpertChoiceRouter):
                        _, mask, aux_loss = self.router(x, step_idx, previous_mask)
                        aux_losses.append(aux_loss)
                    elif isinstance(self.router, TokenChoiceRouter):
                        _, mask = self.router(x, step_idx)
                    else:
                        mask = torch.ones_like(x[:, :, 0])
                    routing_mask = mask
                else:
                    routing_mask = torch.ones(b, t, device=x.device)

                # rho for cache writes: the original positions of the new
                # tokens (identity mapping during single-token decode)
                rho = pos if positions is not None else \
                    torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)

                x_new = self.transformer.h(x, c, cache=cache, step_idx=step_idx,
                                           routing_mask=routing_mask, positions=rho)

                # Update only active tokens
                x = torch.where(routing_mask.unsqueeze(-1).bool(), x_new, x)
                previous_mask = routing_mask

            x = self.transformer.norm_f(x)
            logits = self.lm_head(x[:, -1:, :])
            loss = None
            return logits, loss, x

    # -------------------------------------------------------------------------
    #   Generation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        device = idx.device
        b, t = idx.shape
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]

        cache = None
        if self.config.use_cache and self.config.cache_mode is not None:
            n_head = self.config.n_head
            head_dim = self.config.n_embd // n_head
            num_layers = self.config.n_layer
            num_steps = self.config.max_recursion_depth
            max_seq_len = self.config.block_size

            cache = MoRKVCache(
                mode=self.config.cache_mode,
                max_batch_size=b,
                max_seq_len=max_seq_len,
                num_heads=n_head,
                head_dim=head_dim,
                num_layers=num_layers,
                num_recursion_steps=num_steps,
                device=device
            )

            # Prefill cache with prompt, one token at a time.
            # positions here is the explicit rho: each token's K/V for every
            # loop it survives is written to its ORIGINAL cache slot.
            for pos_i in range(t):
                current_idx = idx[:, pos_i:pos_i+1]
                positions = torch.tensor([[pos_i]], dtype=torch.long, device=device)
                with torch.no_grad():
                    _ = self(current_idx, targets=None, steps=None,
                             cache=cache, positions=positions)

        generated = idx
        for _ in range(max_new_tokens):
            if generated.size(1) > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            if cache is not None:
                last_token = generated[:, -1:]
                current_pos = generated.size(1) - 1
                positions = torch.tensor([[current_pos]], dtype=torch.long, device=device)
                logits, _, _ = self(last_token, targets=None, steps=None,
                                    cache=cache, positions=positions)
            else:
                logits, _, _ = self(generated, targets=None, steps=None)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)

        return generated

    # -------------------------------------------------------------------------
    #  Utility methods
    # -------------------------------------------------------------------------

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        KEEP = ["model_type", "block_size", "vocab_size", "n_layer", "n_head", "n_embd",
                "dropout", "bias", "intermediate_dim", "router_type", "router_beta",
                "router_aux_weight", "max_recursion_depth", "cache_mode", "use_cache"]

        def _as_dict(cfg_like):
            if cfg_like is None:
                return {}
            if isinstance(cfg_like, GPTConfig):
                return cfg_like.__dict__.copy()
            if isinstance(cfg_like, dict):
                return cfg_like.copy()
            raise TypeError("config must be None, a GPTConfig, or a dict")

        def _strip_prefixes(sd, prefixes=("gpt.", "model.", "module.")):
            out = {}
            for k, v in sd.items():
                kk = k
                changed = True
                while changed:
                    changed = False
                    for p in prefixes:
                        if kk.startswith(p):
                            kk = kk[len(p):]
                            changed = True
                out[kk] = v
            return out

        def _unwrap_state_dict(obj):
            if isinstance(obj, dict):
                for key in ("model", "state_dict", "model_state_dict"):
                    if key in obj and isinstance(obj[key], dict):
                        return obj[key]
            return obj

        def _fix_tied(sd):
            if "lm_head.weight" not in sd and "transformer.wte.weight" in sd:
                sd["lm_head.weight"] = sd["transformer.wte.weight"]
            if "transformer.wte.weight" not in sd and "lm_head.weight" in sd:
                sd["transformer.wte.weight"] = sd["lm_head.weight"]
            return sd

        if str(model_path).endswith((".pt", ".pth")):
            ckpt = torch.load(model_path, map_location="cpu")
            sd = _unwrap_state_dict(ckpt)
            sd = _strip_prefixes(sd)
            sd = _fix_tied(sd)
            base = GPTConfig().__dict__.copy()
            base.update(_as_dict(config))
            gcfg = GPTConfig(**base)
            model = cls(gcfg)
            model.load_state_dict(sd, strict=True)
            model.eval()
            return model

        from transformers import AutoConfig, AutoModelForCausalLM
        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_cfg,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        def _get(name, *alts, default=None):
            for n in (name, *alts):
                if hasattr(hf_cfg, n):
                    v = getattr(hf_cfg, n)
                    if v is not None:
                        return v
            return default

        core_config = {
            "model_type": _get("model_type", default=None),
            "block_size": _get("block_size", "n_positions", "max_position_embeddings"),
            "vocab_size": _get("vocab_size"),
            "n_layer": _get("n_layer", "num_hidden_layers"),
            "n_head": _get("n_head", "num_attention_heads"),
            "n_embd": _get("n_embd", "hidden_size"),
            "dropout": _get("dropout", "resid_pdrop", default=0.0),
            "bias": _get("bias", default=False),
            "intermediate_dim": _get("intermediate_dim", "n_inner", default=None),
        }
        if core_config["intermediate_dim"] is None:
            core_config["intermediate_dim"] = 4 * int(core_config["n_embd"])
        core_config = {k: v for k, v in core_config.items() if k in KEEP and v is not None}
        core_config.update(_as_dict(config))
        gcfg = GPTConfig(**core_config)
        model = cls(gcfg)
        sd = hf_model.state_dict()
        sd = _strip_prefixes(sd)
        sd = _fix_tied(sd)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
