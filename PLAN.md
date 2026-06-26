# OlmoEarth 2 — Rewrite Plan

A clean, minimal, production-ready rewrite of `olmoearth_pretrain`. It supports
four capabilities — pretraining dataset construction, pretraining, in-loop
evaluation, and fine-tuning — plus a thin inference/embedding API and an optional
baseline-comparison harness. Everything else is dropped.

---

## 1. Decisions

| Decision | Choice |
|---|---|
| Training backbone | **Keep `ai2-olmo-core`** — reuse its `Trainer`, FSDP, distributed checkpointing, optimizer/scheduler, `Config`/dotted-CLI overrides, and `Callback` framework. |
| Model/objective scope | Pin one blessed **config** (below), but keep the loss / masking / encoding **registries** and their alternative entries for ablations. Delete only non-dispatched code (alternative objectives, deprecated aliases, the `helios` shim). |
| Dataset construction | Target `oe_pretrain_corpus_v2` / the new rslearn `storage=` API. This is a *dataset-creation* API; the corpus-v2 train-time reader is **net-new code**, and switching to it is a data re-baseline (cross-format resume / number reproduction are not expected to hold). Build behind a `DatasetReader` protocol. |
| Checkpoints | Keep olmo-core's PyTorch DCP format. A converter handles key/path remap and strips the frozen target encoder on inference export. |
| Baseline models | Port all 11 external models as opt-in `eval/baselines` adapters. |
| Open-source | Repo is open-sourceable. Beaker is a normal optional dependency. No hardcoded `/weka/...` paths — storage roots are config/env-driven. |

### The blessed model (olmoearth2 v1)

From `scripts/official/v1_1/rope_simple_encodings.py` (`v1_1/base.py` + 2D RoPE +
simple separate-path encodings):

- **Backbone:** multi-modal FlexiViT encoder + shallow Predictor. `base` = 768-d,
  12 encoder layers, 4 decoder layers, 12 heads (also nano/tiny/large via a size table).
- **Patch embedding:** GSD-aware flexible patch size (1–8); per-pixel MLP hidden
  `[64]` before patchification; linear projection; single-bandset tokenization for
  S2 (12 bands) and Landsat (11 bands); random band dropout (max 0.2) on S2 + Landsat only.
- **Encodings:** axial 2D RoPE spatial (base 10000, coord scale 0.25); "separate"
  encoding mode with a learnable per-modality/per-bandset channel embed (dim 128);
  simple 3-number temporal `[frac_year, sin, cos]`; simple 3-number lat/lon
  unit-sphere `[x, y, z]`; lat/lon dropout 0.5.
- **Objective:** frozen-target latent prediction (`ContrastiveLatentMIM` in the old
  code). An online encoder + a frozen target encoder — a seeded deepcopy of the
  encoder at init that never updates (`ema_decay=(1.0,1.0)` makes the EMA update a
  no-op; keep `ema_decay` as a config param but pin it; keep `reinit_targets` off).
  The Predictor regresses masked-token predictions toward the frozen target across
  two masked views. Loss = `modality_patch_discrimination_masked_negatives_vec`
  (τ=0.1; `same_target_threshold=0.999` false-negative masking on decode-only
  modalities) + global InfoNCE between the two pooled views (weight 0.05). KoLeo is
  available as an anti-collapse option.
- **Masking:** `random_time_with_decode` (encode 0.5 / decode 0.5 / random 0.5);
  decode-only modalities: WorldCover, SRTM, OSM-raster, WRI-canopy, CDL, WorldCereal.
  `num_masked_views=2` is coupled to the InfoNCE term.
- **Modalities (9):** Sentinel-2 L2A, Sentinel-1, Landsat, WorldCover, SRTM,
  OpenStreetMap-raster, WRI canopy height, CDL, WorldCereal. ERA5 (era5_10) keeps its
  spec + converter but is off by default.
- **Optim:** AdamW (lr 1e-4, wd 0.02, β₂ 0.95), cosine + 8k warmup, grad-clip 1.0,
  FSDP bf16 params / fp32 reduce, global batch 512, token budget 2250, 300 epochs.

---

## 2. What to keep, port, and drop

### `nn/` → `model/`
- **Port:** `flexi_vit.py` Encoder/Predictor, `attention.py` (Attention/Mlp/Block,
  flash + qk-norm + RoPE), `encodings.py` (RoPE 2D + simple temporal/latlon),
  `flexi_patch_embed.py`, `tokenization.py`, `pooling.py`, `latent_mim.py`
  (contrastive variant), `utils.py` (`unpack_encoder_output`).
- **Keep:** the encoding-mode dispatch (separate/composite/absolute, multifreq
  temporal/latlon) as selectable options; `register_tokens` and the
  `embedding_projector` (used by fine-tuning).
- **Add:** an SDPA attention fallback in `attention.py` (the old code hard-`raise`s
  without flash-attn). Parity-test SDPA against the flash path.
- **Drop:** `st_model.py`, `pooled_modality_predictor.py`, `flexihelios.py`,
  `mae.py`, `galileo.py`, the Reconstructor (MAE) branch.

### `train/` → `train/`
- **Port:** `train_module/contrastive_latentmim.py` + the `OlmoEarthTrainModule`
  base; callbacks `wandb.py`, `speed_monitor.py`, `evaluator_callback.py`.
- **Keep registries:** `loss.py` and `masking.py` keep registry dispatch + their
  alternative entries (pin `modality_patch_discrimination_masked_negatives_vec` +
  `InfoNCE` as the default; `KoLeo`, MAE/L1/L2/cosine, other discrimination losses,
  and the `random`/`time`/`space`/modality-cross strategies stay reachable). Each
  kept entry gets a CI smoke test.
- **Drop:** deprecated loss aliases; `train_module/{latent_mim,galileo,mae}.py`.

### `evals/` → `eval/`
- **Port (core):** in-loop `DownstreamEvaluatorCallback`, `embeddings.py`,
  `eval_wrapper.py` (OlmoEarth wrapper), `linear_probe.py`, `knn.py`, `metrics.py`,
  `finetune/{train,evaluate,model}.py`, dataset loaders + `normalize.py` + configs,
  `studio_ingest/registry.py`.
- **Port (optional extra):** all 11 `evals/models/*` baseline wrappers (Clay,
  Prithvi, Satlas, CROMA, AnySat, Terramind, Presto, DinoV3, Tessera, Galileo,
  Panopticon) → `eval/baselines/` as thin adapters to a shared `EvalModel` interface.
- **Drop:** `embedding_diagnostics.py`, `embedding_transforms.py`,
  `studio_ingest/{ingest,cli,band_stats}.py`.

### `data/` + `dataset/` → `data/`
- **Port:** `dataset.py` (train-time reader), `dataloader.py` (olmo-core
  `DataLoaderBase` subclass with masking/token-budget/collate), `constants.py` →
  `modalities.py` (9 + ERA5 specs), `normalize.py` + `norm_configs/*.json` (verbatim,
  for number comparability), `transform.py`, `collate.py`.
- **Reader behind a protocol:** H5 implementation now; corpus-v2 implementation in
  Phase 6. Thread crop/mask RNG through a seeded, checkpointable per-worker generator.
- **Drop:** `visualize.py` (→ debug script), `concat.py` (if unused under corpus-v2).

### `dataset_creation/` → `dataset_construction/`
- **Rebuild on the rslearn `storage=` API:** the 9 modality converters (S2 L2A, S1,
  Landsat, WorldCover, SRTM, OSM-raster, WRI canopy, CDL, WorldCereal) + ERA5
  (era5_10) + OSM-sampling window creation.
- **Drop:** unused modalities (NAIP×2, coarse `era5`, gse, worldpop, eurocrops,
  s2-l1c), the GCP-Batch L1C infra, one-off utility scripts.

### `internal/` → `launch/`
- **Port:** `experiment.py` builder + `SubCmd` enum + `main()`; model-size table;
  Beaker launch glue in `launch/beaker.py` behind `[beaker]` extras. Storage roots
  from config/env (e.g. `OLMOEARTH2_DATA_ROOT`).
- **Reduce:** fold `all_evals`, `full_eval_sweep*`, `checkpoint_sweep_evals` into
  one `eval/sweep.py`.
- **Drop:** the `Helios*` aliases.

### `inference_benchmarking/` → optional `tools/benchmark.py`.

### Drop entirely
`scripts/archived/`, committed checkpoints under `scripts/official/local_output/`,
`train_latlon.log`, `helios/` + `_compat.py`, `.venv`, `wandb/`, `*.egg-info`,
caches, `.swp` files.

**Net:** ~47k LOC → ~12–15k LOC core (+ ~7.5k optional baselines + construction).

---

## 3. Repository structure

```
olmoearth2/
├── pyproject.toml
├── README.md
├── PLAN.md
├── LICENSE
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── olmoearth2/
│   ├── __init__.py
│   ├── datatypes.py            # Sample, MaskedSample, TokensAndMasks, MaskValue
│   ├── config.py               # olmo_core.Config + project base config
│   ├── data/
│   │   ├── reader.py           # DatasetReader protocol
│   │   ├── modalities.py       # ModalitySpec, BandSet, 9 + ERA5 specs
│   │   ├── normalize.py
│   │   ├── norm_configs/*.json
│   │   ├── dataset.py          # H5 reader (now) / corpus-v2 reader (Phase 6)
│   │   ├── dataloader.py       # olmo-core DataLoaderBase subclass
│   │   ├── transform.py
│   │   └── collate.py
│   ├── model/
│   │   ├── patch_embed.py
│   │   ├── attention.py        # flash + SDPA fallback
│   │   ├── encodings.py
│   │   ├── tokenization.py
│   │   ├── pooling.py
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── olmoearth.py        # frozen-target latent prediction
│   │   └── sizes.py            # nano/tiny/base/large
│   ├── train/
│   │   ├── masking.py
│   │   ├── loss.py
│   │   ├── train_module.py
│   │   └── callbacks/{wandb,speed_monitor,eval}.py
│   ├── eval/
│   │   ├── embeddings.py
│   │   ├── probe.py
│   │   ├── knn.py
│   │   ├── finetune.py
│   │   ├── metrics.py
│   │   ├── sweep.py
│   │   ├── datasets/
│   │   └── baselines/          # optional, per-model
│   ├── dataset_construction/
│   │   ├── windows.py          # OSM-sampled window creation
│   │   └── modalities/*.py
│   ├── inference/
│   │   └── model_loader.py     # load from HF/local; minimal-dep inference
│   └── launch/
│       ├── experiment.py       # main(), SubCmd, builder pattern
│       └── beaker.py           # isolated; no Weka paths
├── configs/                    # olmoearth_{nano,tiny,base,large}.py
├── scripts/                    # thin CLI entrypoints
├── tools/                      # benchmark, checkpoint converter
└── tests/
    ├── test_model_parity.py    # online-encoder parity; SDPA vs flash
    ├── test_loss_parity.py     # pin the _vec loss
    ├── test_objective.py       # target encoder bit-static across steps
    ├── test_data.py            # shapes/norm/fingerprint + determinism
    ├── test_masking.py
    ├── test_eval_invariants.py # online encoder, eval-mode restore, PCA train-only
    ├── test_registry_smoke.py  # touch every kept registry entry
    └── artifacts/              # 9-modality fixture
```

**Dependency extras:**
- core (default): `torch` (CPU index default; GPU opt-in), `einops`, `numpy`,
  `huggingface_hub`, `universal-pathlib` — inference on the SDPA path, no flash-attn.
- `[training]`: `ai2-olmo-core==2.3.0` (pinned; upgrade-canary CI job), `wandb`, `matplotlib`, `pandas`.
- `[flash-attn]`: `flash-attn` (opt-in GPU speedup).
- `[data]`: `rslearn`, `gcsfs`, `rasterio`, `hdf5plugin`.
- `[eval]`: `geobench`, `scikit-learn`, `torchmetrics`, `rioxarray` (no lightning/terratorch).
- `[eval-baselines]`: per-model; only `clay`/`terramind`/`satlas` pull heavy deps,
  the other 8 are self-contained. Prefer PyPI over git sources.
- `[beaker]`: `beaker-py==1.34.1`, `google-cloud-compute`.
- `[dev]`: `ruff`, `mypy`, `pytest`, `pre-commit`, secret-detection.

---

## 4. Phased implementation

Each phase ends with a green test gate. Parity tests validate inference (weights →
embeddings, batch → loss); training dynamics and number reproduction have their own
gates (Phases 3 and 8). Estimated ~15–19 engineer-weeks; Phase 6 is the main risk.

**Phase 0 — Scaffold.** Repo skeleton, `pyproject.toml` with extras, ruff/mypy,
pre-commit, CI, LICENSE, `datatypes.py`, `config.py`. *Gate: package imports, CI green.*

**Phase 0.5 — Interface freeze + parity harness.** Freeze three contracts:
(a) the `DatasetReader` protocol (return types, dtypes, `[H,W,T,C]` shapes,
timestamps, `missing_timesteps_masks`, normalization ownership); (b) checkpoint
invariants (frozen target encoder in training state, reproducibly seeded; stripped
on export); (c) the deterministic dataloader contract (per-worker RNG seeding;
checkpointable crop/mask generator). Stand up the harness: a frozen old-repo venv,
published nano + base checkpoints pulled locally, and a checked-in 9-modality
fixture (the H5 dir carries 11 modalities incl. gse/worldpop). *Gate: contracts
merged; old repo importable; fixture + checkpoints present.*

**Phase 1 — Model.** Port encoder/predictor/attention/encodings/patch_embed/
tokenization/pooling; pin the blessed config, keep the encoding-mode registry; add
the SDPA fallback. *Gate: load a v1_1 checkpoint, online-encoder embeddings match
the old code within tolerance; param-count matches by size; SDPA-vs-flash parity.*

**Phase 2 — Data (train-time).** `reader.py` (H5 impl), `modalities.py`,
`normalize.py`, `dataset.py`, `dataloader.py`, transforms/collate. *Gate: a batch
loads; shapes/masks/normalization match the old loader on the fixture;
data-population + shuffle-order parity (fingerprint reproduced); per-worker
determinism holds.*

**Phase 3 — Train.** `masking.py`, `loss.py` (pin + numerically parity-test the
`_vec` loss), `train_module.py`, wandb/speed callbacks. *Gate: nano run trains
locally; distributional loss/grad-norm parity vs old nano over a window; target
encoder asserted bit-static across steps.*

**Phase 4 — In-loop eval.** Evaluator callback + embeddings/probe/knn/finetune +
2–3 datasets (eurosat KNN, mados linear-probe). Invariants: online encoder,
`model.eval()` save/restore, PCA fit on train split only, verbatim norm configs,
rank-0 guard + correct FSDP sharded forward. *Gate: callback runs during a nano run
and matches the old harness on one dataset; invariants tested.*

**Phase 5 — Launch.** `experiment.py` builder + `SubCmd`; `beaker.py`;
`configs/olmoearth_{nano..large}.py`. *Gate: `dry_run` builds the full config;
`launch` submits a Beaker job; CI grep gate fails on `/weka` or `ai2/` literals
outside `launch/`.*

**Phase 6 — Dataset construction.** corpus-v2 train-time reader (`DatasetReader`
impl) + the rslearn `storage=` construction pipeline for the 9 modalities + OSM
window sampling. Run a spike alongside Phase 2 to validate the reader against the
real `storage=` shape. *Gate: construct a small dataset end-to-end, train a nano run
on it, and confirm a short fresh run's eval-suite metrics don't shift vs the H5
baseline.*

**Phase 7 — Fine-tuning + sweeps + baselines.** `eval/finetune.py`, `eval/sweep.py`,
all 11 `eval/baselines/*` adapters behind per-model extras. *Gate: finetune a
checkpoint on one dataset; one baseline evaluates through the shared interface.*

**Phase 8 — Inference + docs.** `inference/model_loader.py` (HF/local, minimal deps),
the DCP→inference converter in `tools/`, concise docs. *Gates: (1) load a published
model with core deps only (SDPA path) and extract embeddings; (2) load a published
checkpoint via the converter and reproduce its in-loop eval-suite numbers within
tolerance.*
