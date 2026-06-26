# OlmoEarth 2

A clean rewrite of `olmoearth_pretrain` — a library for developing Earth-system
foundation models. Built on [`ai2-olmo-core`](https://github.com/allenai/OLMo-core)
(Trainer / FSDP / distributed checkpointing / config + dotted-CLI overrides).

It supports the full lifecycle in one codebase:

1. **Pretraining dataset construction** (`olmoearth2/dataset_construction/`)
2. **Pretraining** — multi-modal FlexiViT encoder + shallow Predictor, frozen-target
   latent-prediction (`ContrastiveLatentMIM`) objective
3. **In-loop evaluation** — KNN / linear-probe / fine-tune downstream tasks
   (`olmoearth2/eval/`)
4. **Fine-tuning** and offline evaluation (incl. optional baseline-model comparison)

Plus a thin, minimal-dependency inference / embedding API (`olmoearth2/inference`).

See [`PLAN.md`](PLAN.md) for the full design and phased plan.

## The blessed model (v1)

The crystallized config (`configs/olmoearth_base.py`) is the v1.1 `rope_simple`
recipe: a 768-d / 12-encoder / 4-decoder FlexiViT over 9 modalities (S2 L2A, S1,
Landsat, WorldCover, SRTM, OSM-raster, WRI canopy, CDL, WorldCereal), with 2D
axial RoPE spatial encodings, simple 3-number temporal/lat-lon encodings,
`random_time_with_decode` masking, and a frozen target encoder
(`ema_decay=(1.0, 1.0)` is a load-bearing no-op).

## Install

```bash
pip install -e .                       # core / inference (SDPA path, no flash-attn)
pip install -e '.[training]'           # + ai2-olmo-core, wandb, data deps
pip install -e '.[training,beaker]'    # + Beaker launch
```

Attention runs on a PyTorch SDPA path by default; `flash-attn` is an optional
speedup (`.[flash-attn]`). If `use_flash_attn=True` is requested without it
installed, the model logs a warning and falls back to SDPA.

## Train

Local single-GPU smoke run (nano, 100 steps; reads `OLMOEARTH2_H5_DIR`):

```bash
python configs/olmoearth_nano.py train_single smoke local \
  --trainer.max_duration.value=100 --trainer.max_duration.unit=steps \
  --data_loader.global_batch_size=32 --data_loader.num_workers=4 \
  --train_module.rank_microbatch_size=32
```

Launch the base model on Beaker (8×H100, urgent):

```bash
python configs/olmoearth_base.py launch <run_name> ai2/jupiter \
  --launch.num_gpus=8 --launch.priority=urgent
```

`dry_run` prints the fully-built config without running. Storage roots are
config/env-driven (`OLMOEARTH2_H5_DIR`); no hardcoded paths live outside
`olmoearth2/launch/`.

## Test

```bash
pytest tests/        # CPU-only smoke tests (imports, model build, registries)
```
