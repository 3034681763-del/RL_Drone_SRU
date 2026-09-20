# RL Drone SRU

Differentiable-physics training code for vision-based drone flight with a
spatially enhanced recurrent Worker (SRU).

The SRU Worker keeps a fixed-size spatial memory over depth features and is a
drop-in alternative to the causal Transformer Worker. The main training entry
point supports alternating Worker and loss-generation-network (LGN) updates,
differentiable meta rollouts, online random maps, and optional global guidance.

## Environment

The CUDA extension has been tested with:

- Python 3.11
- PyTorch 2.2.2
- CUDA 11.8

Build the extension from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda
PIP_NO_BUILD_ISOLATION=1 pip install ./src --no-build-isolation --no-use-pep517
python -c "import torch; import quadsim_cuda; print(torch.__version__)"
```

See [`src/README.md`](src/README.md) for additional build troubleshooting.

## Training

The SRU backbone is the default:

```bash
python mmgj_transformer.py \
  --worker_backbone sru \
  --sru_hidden_channels 96 \
  --exp_name sru_run
```

Use the Transformer baseline with:

```bash
python mmgj_transformer.py --worker_backbone transformer --exp_name transformer_run
```

Additional training and potential-guidance options are documented in
[`使用说明_训练与势场引导.md`](使用说明_训练与势场引导.md).

## Tests

The model-level tests do not require a running simulator:

```bash
python -m unittest discover -s tests -v
```

## Repository Contents

- `WorkNet_sru.py`: spatial recurrent Worker
- `WorkNet_transformer.py`: causal Transformer baseline
- `LossGenNet_transformer.py`: dynamic loss-generation network
- `mmgj_transformer.py`: alternating Worker/LGN training entry point
- `utils/`: rollout, planning, logging, tensor, and checkpoint helpers
- `src/`: differentiable CUDA simulator extension
- `tests/`: unit tests for models, losses, gradients, and checkpoint mapping

Generated checkpoints, logs, binary extensions, virtual environments, and
local visualization artifacts are intentionally excluded from version control.
