"""Touch every kept registry entry (loss + masking).

Builds each registered loss type and each masking strategy type by introspecting
the registries. Includes explicit checks on the blessed entries:
``modality_patch_discrimination_masked_negatives_vec``, ``InfoNCE`` (loss) and
``random_time_with_decode`` (masking). CPU-only, no data.
"""

import inspect

import pytest

from olmoearth2.train.loss import LOSS_REGISTRY, Loss, LossConfig
from olmoearth2.train.masking import (
    MASKING_STRATEGY_REGISTRY,
    MaskingConfig,
    MaskingStrategy,
)

# Required constructor kwargs for the few masking strategies that need them.
# Everything else builds from defaults.
_MASKING_REQUIRED_KWARGS: dict[str, dict] = {
    "random_fixed_modality": {"decoded_modalities": ["worldcover"]},
    "selectable_modality": {
        "decodable_modalities": ["worldcover"],
        "fully_mask_modalities": [],
        "max_to_mask": 1,
    },
    "selectable_random_range_modality": {
        "decodable_modalities": ["worldcover"],
        "fully_mask_modalities": [],
        "max_to_mask": 1,
    },
}


def _loss_keys() -> list[str]:
    return sorted(LOSS_REGISTRY.keys())


def _masking_keys() -> list[str]:
    return sorted(MASKING_STRATEGY_REGISTRY.keys())


def test_blessed_entries_present():
    """The blessed loss + masking registry keys must exist."""
    assert "modality_patch_discrimination_masked_negatives_vec" in _loss_keys()
    assert "InfoNCE" in _loss_keys()
    assert "random_time_with_decode" in _masking_keys()


@pytest.mark.parametrize("key", _loss_keys())
def test_every_loss_instantiates(key: str):
    """Every registered loss builds from defaults and is a Loss."""
    cls = LOSS_REGISTRY.get_class(key)
    loss = cls()
    assert isinstance(loss, Loss)


@pytest.mark.parametrize("key", _masking_keys())
def test_every_masking_strategy_instantiates(key: str):
    """Every registered masking strategy builds and is a MaskingStrategy."""
    cls = MASKING_STRATEGY_REGISTRY.get_class(key)
    kwargs = _MASKING_REQUIRED_KWARGS.get(key, {})
    strat = cls(**kwargs)
    assert isinstance(strat, MaskingStrategy)


def test_blessed_loss_via_config():
    """The blessed _vec loss builds through LossConfig and computes a callable."""
    loss = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives_vec",
            "tau": 0.1,
            "same_target_threshold": 0.999,
            "mask_negatives_for_modalities": [],
        }
    ).build()
    assert isinstance(loss, Loss)
    assert hasattr(loss, "compute")


def test_blessed_infonce_via_config():
    """The InfoNCE contrastive loss builds through LossConfig."""
    loss = LossConfig(loss_config={"type": "InfoNCE", "weight": 0.05}).build()
    assert isinstance(loss, Loss)


def test_blessed_masking_via_config():
    """The random_time_with_decode strategy builds through MaskingConfig."""
    strat = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ["worldcover"],
        }
    ).build()
    assert isinstance(strat, MaskingStrategy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
