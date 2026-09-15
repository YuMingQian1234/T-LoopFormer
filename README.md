# T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning with Dynamic Routing

---

This repository contains the official implementation of **T-LoopFormer**.

The codebase is a fork of **NanoGPT**, and we intentionally keep it as close as possible to the original implementation for clarity and reproducibility. Beyond the looped / elastic-depth components, the main architectural difference is using **RMSNorm** instead of **LayerNorm**.

---

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
## Training
```bash
torchrun --standalone --nproc_per_node=6 train.py
```
## Cite
@misc{yu2026tloopformertokenlevelelasticdepthlooped,
      title={T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing}, 
      author={Mingqian Yu and Wenpeng Zhang and Peilin Zhao},
      year={2026},
      eprint={2609.15160},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.15160}, 
}
