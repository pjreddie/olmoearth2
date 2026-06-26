"""Apply the blessed `random_time_with_decode` masking to a synthetic batch.

Asserts mask-value invariants: masks only ever take the four MaskValue values,
shapes/band-set counts are right, missing tokens stay missing, decode-only
modalities are never placed in the encoder mask, and encode-decode modalities
do produce encoder tokens. CPU-only, no data files.
"""

import torch

from olmoearth2.data.constants import MISSING_VALUE, Modality
from olmoearth2.datatypes import MaskValue, OlmoEarthSample
from olmoearth2.train.masking import (
    MaskedOlmoEarthSample,
    RandomTimeWithDecodeMaskingStrategy,
)

ONLY_DECODE = ["worldcover", "srtm"]


def _make_batch(b=6, h=8, w=8, t=6):
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2021, (b, t, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # (B, T, 3)

    return OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, Modality.SENTINEL2_L2A.num_bands)),
        sentinel1=torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands)),
        worldcover=torch.ones((b, h, w, 1, Modality.WORLDCOVER.num_bands)),
        srtm=torch.ones((b, h, w, 1, Modality.SRTM.num_bands)),
        latlon=torch.ones((b, Modality.LATLON.num_bands)),
        timestamps=timestamps,
    )


def _apply():
    import random as _random

    import numpy as _np

    torch.manual_seed(0)
    _np.random.seed(0)
    _random.seed(0)
    batch = _make_batch()
    strat = RandomTimeWithDecodeMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        random_ratio=0.5,
        only_decode_modalities=ONLY_DECODE,
    )
    return strat.apply_mask(batch, patch_size=2)


VALID_MASK_VALUES = {
    MaskValue.ONLINE_ENCODER.value,
    MaskValue.TARGET_ENCODER_ONLY.value,
    MaskValue.DECODER.value,
    MaskValue.MISSING.value,
}


def test_mask_values_are_in_enum():
    """Every mask entry is one of the four MaskValue values."""
    masked = _apply()
    assert isinstance(masked, MaskedOlmoEarthSample)
    for name in masked._fields:
        if not name.endswith("_mask"):
            continue
        mask = getattr(masked, name)
        if mask is None:
            continue
        uniq = set(torch.unique(mask).tolist())
        assert uniq <= VALID_MASK_VALUES, f"{name} has out-of-range values {uniq}"


def test_mask_band_set_shapes():
    """Each mask matches its data spatial/temporal shape, last dim == num_band_sets."""
    masked = _apply()
    for mask_name in masked._fields:
        if not mask_name.endswith("_mask"):
            continue
        mask = getattr(masked, mask_name)
        if mask is None:
            continue
        modality_name = masked.get_unmasked_modality_name(mask_name)
        data = getattr(masked, modality_name)
        modality = Modality.get(modality_name)
        assert mask.shape[:-1] == data.shape[:-1], f"{mask_name} shape mismatch"
        assert mask.shape[-1] == modality.num_band_sets, (
            f"{mask_name} last dim != num_band_sets"
        )


def test_decode_only_modalities_never_in_encoder():
    """Decode-only modalities must never be placed in the encoder mask."""
    masked = _apply()
    for modality_name in ONLY_DECODE:
        mask = getattr(masked, f"{modality_name}_mask")
        assert mask is not None
        n_encoder = (mask == MaskValue.ONLINE_ENCODER.value).sum().item()
        assert n_encoder == 0, f"{modality_name} has {n_encoder} encoder tokens"


def test_decode_only_modalities_are_decoded():
    """Decode-only (non-missing) modalities are all decode tokens."""
    masked = _apply()
    for modality_name in ONLY_DECODE:
        mask = getattr(masked, f"{modality_name}_mask")
        # No missing values were injected, so all should be DECODER.
        assert (mask == MaskValue.DECODER.value).all()


def test_encode_decode_modality_has_encoder_and_decoder_tokens():
    """Across the batch, encode-decode modalities yield both encode and decode tokens."""
    masked = _apply()
    for modality_name in ["sentinel2_l2a", "sentinel1"]:
        mask = getattr(masked, f"{modality_name}_mask")
        assert mask is not None
        n_encoder = (mask == MaskValue.ONLINE_ENCODER.value).sum().item()
        n_decoder = (mask == MaskValue.DECODER.value).sum().item()
        assert n_encoder > 0, f"{modality_name} produced no encoder tokens"
        assert n_decoder > 0, f"{modality_name} produced no decoder tokens"


def test_ratios_roughly_honored_for_encode_decode_modalities():
    """Encode + decode + target fractions should sum to ~1 and be non-degenerate."""
    masked = _apply()
    mask = masked.sentinel2_l2a_mask
    total = mask.numel()
    enc = (mask == MaskValue.ONLINE_ENCODER.value).sum().item() / total
    dec = (mask == MaskValue.DECODER.value).sum().item() / total
    tgt = (mask == MaskValue.TARGET_ENCODER_ONLY.value).sum().item() / total
    miss = (mask == MaskValue.MISSING.value).sum().item() / total
    assert abs((enc + dec + tgt + miss) - 1.0) < 1e-6
    # No missing data injected.
    assert miss == 0.0
    # With encode/decode/random all 0.5 we expect a meaningful split (not all one).
    assert enc > 0.0 and dec > 0.0


def test_missing_tokens_stay_missing():
    """Injected MISSING data must produce MISSING mask entries, never encoder."""
    torch.manual_seed(1)
    b, h, w, t = 4, 8, 8, 6
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2021, (b, t, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    sentinel1 = torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands))
    sentinel1[b // 2 :] = MISSING_VALUE  # half the batch missing
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, Modality.SENTINEL2_L2A.num_bands)),
        sentinel1=sentinel1,
        timestamps=timestamps,
    )
    masked = RandomTimeWithDecodeMaskingStrategy(
        encode_ratio=0.5, decode_ratio=0.5, random_ratio=0.5, only_decode_modalities=[]
    ).apply_mask(batch, patch_size=2)
    s1_mask = masked.sentinel1_mask
    # Missing samples must be entirely MISSING.
    assert (s1_mask[b // 2 :] == MaskValue.MISSING.value).all()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
