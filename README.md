# T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning with Dynamic Routing

---

This repository contains the official implementation of **T-LoopFormer**.

The codebase is a fork of **NanoGPT**, and we intentionally keep it as close as possible to the original implementation for clarity and reproducibility. Beyond the looped / elastic-depth components, the main architectural difference is using **RMSNorm** instead of **LayerNorm**.

---

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```


