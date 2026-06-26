"""The frozen target encoder must be bit-static across an online-encoder step.

The blessed objective (`LatentMIM`) holds a `target_encoder` that is a deepcopy of
the online encoder at init, with `requires_grad=False`. The train module pins
`ema_decay=(1.0, 1.0)`, which makes the EMA update a no-op. This test:

  1. asserts the blessed config pins `ema_decay == (1.0, 1.0)`,
  2. asserts the train module's `update_target_encoder` early-returns for (1, 1)
     (by source inspection — running the full train step needs a Trainer),
  3. builds the nano model and performs a real optimizer step on the *online*
     encoder, then asserts every target-encoder parameter is byte-identical.

CPU-only, tiny tensors, no data.
"""

import copy
import inspect
import sys
from pathlib import Path

import pytest
import torch

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
sys.path.insert(0, str(CONFIGS))


def _build_nano_model():
    from olmoearth2.launch.experiment import CommonComponents

    import v1_lib

    common = CommonComponents(
        run_name="t",
        save_folder="/tmp/oe2-objective-test",
        training_modalities=list(v1_lib.TRAINING_MODALITIES),
        tokenization_config=v1_lib._tokenization_config(),
    )
    return v1_lib.make_build_model_config("nano")(common).build()


def _make_masked_sample(model):
    """A tiny MaskedOlmoEarthSample the online encoder can consume."""
    from olmoearth2.data.constants import Modality
    from olmoearth2.datatypes import MaskedOlmoEarthSample, MaskValue

    torch.manual_seed(0)
    b, h, w, t = 1, 4, 4, 2

    def _mod(name):
        nb = Modality.get(name).num_bands
        nbs = Modality.get(name).num_band_sets
        data = torch.randn(b, h, w, t, nb)
        mask = torch.full(
            (b, h, w, t, nbs),
            fill_value=MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
        )
        return data, mask

    s2, s2_mask = _mod("sentinel2_l2a")
    s1, s1_mask = _mod("sentinel1")
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2021, (b, t, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    latlon = torch.randn(b, Modality.LATLON.num_bands)
    latlon_mask = torch.full(
        (b, Modality.LATLON.num_band_sets),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.float32,
    )
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=s2_mask,
        sentinel1=s1,
        sentinel1_mask=s1_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        timestamps=timestamps,
    )


def test_blessed_ema_decay_is_pinned_to_one():
    """The blessed train-module config pins ema_decay = (1.0, 1.0)."""
    from olmoearth2.launch.experiment import CommonComponents

    import v1_lib

    common = CommonComponents(
        run_name="t",
        save_folder="/tmp/oe2-objective-test",
        training_modalities=list(v1_lib.TRAINING_MODALITIES),
        tokenization_config=v1_lib._tokenization_config(),
    )
    tm_config = v1_lib.build_train_module_config(common)
    assert tuple(tm_config.ema_decay) == (1.0, 1.0)


def test_update_target_encoder_is_noop_for_unit_ema():
    """update_target_encoder early-returns when start_ema == end_ema == 1.0."""
    from olmoearth2.train.train_module import train_module as tm_mod

    src = inspect.getsource(tm_mod.OlmoEarthTrainModule.update_target_encoder)
    # The guard clause that makes the EMA a no-op.
    assert "self.start_ema == 1.0 and self.end_ema == 1.0" in src
    assert "return" in src.split("self.start_ema == 1.0 and self.end_ema == 1.0", 1)[1]


def test_target_encoder_requires_grad_false():
    """Target encoder params are frozen at build time."""
    model = _build_nano_model()
    assert any(True for _ in model.target_encoder.parameters())
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())


def test_target_encoder_bit_static_after_online_step():
    """After an optimizer step on the online encoder, target params are unchanged."""
    model = _build_nano_model()
    model.train()

    target_before = [p.detach().clone() for p in model.target_encoder.parameters()]

    x = _make_masked_sample(model)
    optim = torch.optim.AdamW(
        [p for p in model.encoder.parameters() if p.requires_grad], lr=1e-3
    )

    # Run the online encoder only (target update is a no-op under ema=(1,1)).
    optim.zero_grad()
    out = model.encoder(x, patch_size=2)
    tokens_and_masks = out["tokens_and_masks"]
    flat_tokens, _ = tokens_and_masks.flatten_all_tokens_and_masks()
    loss = flat_tokens.float().pow(2).mean()
    loss.backward()
    optim.step()

    # Online encoder moved.
    online_moved = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
        if p.requires_grad
    )
    assert online_moved, "online encoder received no gradient"

    # Target encoder is byte-identical.
    target_after = list(model.target_encoder.parameters())
    assert len(target_after) == len(target_before)
    for before, after in zip(target_before, target_after):
        assert torch.equal(before, after), "target encoder parameter changed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
