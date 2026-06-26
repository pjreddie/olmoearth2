# OlmoEarth 2

A minimal, production-ready rewrite of `olmoearth_pretrain` — a library for
developing Earth-system foundation models.

It supports the full lifecycle in one clean codebase:

1. **Pretraining dataset construction** (on rslearn / corpus-v2)
2. **Pretraining** (multi-modal ViT, ContrastiveLatentMIM objective, on `ai2-olmo-core`)
3. **In-loop evaluation** (KNN / linear-probe / fine-tune downstream tasks during training)
4. **Fine-tuning** and offline evaluation (incl. optional baseline-model comparison)

Plus a thin, minimal-dependency inference / embedding-extraction API.

> **Status:** planning. See [`PLAN.md`](PLAN.md) for the full functionality,
> structure, and phased implementation plan. No code yet.
