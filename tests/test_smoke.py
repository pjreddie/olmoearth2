"""Smoke tests: package imports, blessed model builds, registries resolve.

These are CPU-only and require no data — they exercise the ported model / loss /
masking on the SDPA attention path (no flash-attn needed).
"""

import sys
from pathlib import Path

import pytest

# Make configs/ importable (v1_lib + entrypoints live there).
CONFIGS = Path(__file__).resolve().parent.parent / "configs"
sys.path.insert(0, str(CONFIGS))


def test_package_imports():
    """Core subpackages import without flash-attn / beaker."""
    import olmoearth2  # noqa: F401
    import olmoearth2.data.dataset  # noqa: F401
    import olmoearth2.model.flexi_vit  # noqa: F401
    import olmoearth2.train.loss  # noqa: F401
    import olmoearth2.train.masking  # noqa: F401


def test_blessed_nano_model_builds():
    """The blessed nano config builds a real model with parameters."""
    import torch

    from olmoearth2.launch.experiment import CommonComponents

    import v1_lib

    common = CommonComponents(
        run_name="t",
        save_folder="/tmp/oe2-test",
        training_modalities=list(v1_lib.TRAINING_MODALITIES),
        tokenization_config=v1_lib._tokenization_config(),
    )
    model = v1_lib.make_build_model_config("nano")(common).build()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    # SDPA path must be the default (no flash-attn required to run).
    assert model.encoder.use_flash_attn is False


def test_loss_and_masking_registries_resolve():
    """The pinned blessed loss + masking entries are reachable via registry."""
    from olmoearth2.train.loss import LossConfig
    from olmoearth2.train.masking import MaskingConfig

    loss = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives_vec",
            "tau": 0.1,
            "same_target_threshold": 0.999,
            "mask_negatives_for_modalities": [],
        }
    )
    assert loss.build() is not None

    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": [],
        }
    )
    assert masking is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
