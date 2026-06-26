# Implementation status

Status of the [`PLAN.md`](PLAN.md) phases. "Verified" = exercised on the local
H100 / real data, not just import-clean.

| Phase | Item | Status |
|---|---|---|
| 0 | Scaffold (pyproject+extras, config, datatypes) | ✅ done |
| 0.5 | `DatasetReader` protocol; checkpoint/dataloader contracts | ✅ protocol in `data/reader.py` |
| 1 | Model port + **SDPA fallback** (runs w/o flash-attn) | ✅ verified (trains on SDPA path) |
| 2 | Data layer (H5 reader, dataloader, masking/collate) | ✅ verified (loads real H5, trains) |
| 3 | Train (masking/loss registries, train module, callbacks) | ✅ verified (100-step local run; loss parity test) |
| 4 | In-loop eval (evaluator callback, knn/probe/finetune) | ✅ verified (m-eurosat KNN + mados probe in-loop on GPU) |
| 5 | Launch (`experiment.py`, `beaker.py`, configs) | ✅ verified (builds full config; submits to Beaker) |
| 8 | Inference loader + DCP→encoder converter + docs | ✅ verified (converted `rope_simple_v1`, reloaded encoder core-deps) |
| — | Test suite + CI | ✅ 62 non-slow tests pass; CI + pre-commit added |
| 7 | Finetune + sweeps + **11 baselines** | 🟡 partial: finetune imports, sweep façade, baselines made lazy/optional, one baseline (Presto) loads. Per-baseline weight-verified eval not run (needs each model's weights). |
| 6 | corpus-v2 reader + construction | 🟡 partial: reader ported & import-clean (`data/corpus_v2.py`) + protocol. End-to-end build/train on corpus-v2 not done — needs the `storage=` construction pipeline + a materialized corpus (the PLAN's flagged ~3-4wk data re-baseline). |
| 14 | PLAN cleanups (drop modules, prune aliases) | 🟡 deferred: drop targets (`st_model`, `pooled_modality_predictor`, alt train modules, `_compat`/Helios) are coupled to verified-working `eval_wrapper`/`speed_monitor`; pruning deferred to avoid regressions for cosmetic gain. |

## Verified locally on 1×H100
- nano pretraining, 100 steps, real 9-modality H5 data, SDPA path → clean finish + checkpoints.
- nano + in-loop downstream eval (eurosat KNN, mados linear-probe) → real metrics.
- DCP checkpoint → inference encoder export → reload with core deps.
- 62 CPU tests (registry / loss-parity / masking / objective / model / data).

## Production launch
Validated end-to-end in a fresh `.[training,beaker]` uv venv (the exact Beaker
node env). 8×H100 `ai2/jupiter` urgent launch via
`python configs/olmoearth_base.py launch <name> ai2/jupiter --launch.num_gpus=8 --launch.priority=urgent`.
