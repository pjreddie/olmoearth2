"""Numerical parity of the blessed `_vec` loss: olmoearth2 vs olmoearth_pretrain.

``modality_patch_discrimination_masked_negatives_vec`` is a byte-identical port,
so on identical seeded random tensors the two implementations must agree to 1e-5.
If the old package cannot be imported, we fall back to asserting olmoearth2's loss
is finite and deterministic across two calls with the same seed.
"""

import pytest
import torch

import olmoearth2.train.loss as oe2_loss
from olmoearth2.datatypes import MaskValue

LOSS_KEY = "modality_patch_discrimination_masked_negatives_vec"


def _build_tokens_and_masks(tokens_and_masks_cls, *, seed: int):
    """Build a small TokensAndMasks with two modalities and mixed mask values.

    Layout matches the flattened-per-modality contract used by the loss:
    modality tensor (B, P_H, P_W, T, BandSets, D) and mask (B, P_H, P_W, T, BandSets).
    """
    torch.manual_seed(seed)
    b, ph, pw, t, bs, d = 2, 2, 2, 2, 1, 8

    def _one():
        tok = torch.randn(b, ph, pw, t, bs, d)
        # mask of shape (B, P_H, P_W, T, BandSets) with a mix of encoder/decoder/target
        flat = torch.randint(0, 3, (b, ph, pw, t, bs))  # 0,1,2 = enc/target/decoder
        # Guarantee at least some decoder tokens per sample (loss only scores decoder).
        flat[:, 0, 0, 0, 0] = MaskValue.DECODER.value
        flat[:, 1, 1, 1, 0] = MaskValue.DECODER.value
        return tok, flat.to(torch.float32)

    s2_tok, s2_mask = _one()
    s1_tok, s1_mask = _one()
    return tokens_and_masks_cls(
        sentinel2_l2a=s2_tok,
        sentinel2_l2a_mask=s2_mask,
        sentinel1=s1_tok,
        sentinel1_mask=s1_mask,
    )


def _compute_oe2(seed: int) -> torch.Tensor:
    loss = oe2_loss.LOSS_REGISTRY.get_class(LOSS_KEY)(
        tau=0.1, same_target_threshold=0.999, mask_negatives_for_modalities=[]
    )
    preds = _build_tokens_and_masks(oe2_loss.TokensAndMasks, seed=seed)
    targets = _build_tokens_and_masks(oe2_loss.TokensAndMasks, seed=seed + 1)
    return loss.compute(preds, targets)


def test_oe2_loss_finite_and_deterministic():
    """olmoearth2 loss is finite and bit-identical across two same-seed calls."""
    a = _compute_oe2(seed=1234)
    b = _compute_oe2(seed=1234)
    assert torch.isfinite(a).all(), "loss is not finite"
    assert torch.equal(a, b), "loss is not deterministic across identical-seed calls"


def test_loss_parity_against_olmoearth_pretrain():
    """Numerically compare the _vec loss against the olmoearth_pretrain port."""
    oep_loss = pytest.importorskip("olmoearth_pretrain.train.loss")
    if LOSS_KEY not in oep_loss.LOSS_REGISTRY.keys():
        pytest.skip("old repo lacks the _vec loss key")

    # The old repo's TokensAndMasks lives in nn.flexi_vit (re-exported).
    oep_fv = pytest.importorskip("olmoearth_pretrain.nn.flexi_vit")
    oep_tam_cls = oep_fv.TokensAndMasks

    seed = 4321

    oe2 = oe2_loss.LOSS_REGISTRY.get_class(LOSS_KEY)(
        tau=0.1, same_target_threshold=0.999, mask_negatives_for_modalities=[]
    )
    oep = oep_loss.LOSS_REGISTRY.get_class(LOSS_KEY)(
        tau=0.1, same_target_threshold=0.999, mask_negatives_for_modalities=[]
    )

    oe2_preds = _build_tokens_and_masks(oe2_loss.TokensAndMasks, seed=seed)
    oe2_targets = _build_tokens_and_masks(oe2_loss.TokensAndMasks, seed=seed + 7)
    oep_preds = _build_tokens_and_masks(oep_tam_cls, seed=seed)
    oep_targets = _build_tokens_and_masks(oep_tam_cls, seed=seed + 7)

    out2 = oe2.compute(oe2_preds, oe2_targets)
    out1 = oep.compute(oep_preds, oep_targets)

    assert torch.isfinite(out2).all() and torch.isfinite(out1).all()
    assert torch.allclose(out2, out1, atol=1e-5, rtol=1e-5), (
        f"loss parity mismatch: oe2={out2.item()} vs oep={out1.item()}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
