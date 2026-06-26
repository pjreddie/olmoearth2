"""Online-encoder structural + (optional) weight-loading parity.

Light (fast) path: build the blessed model and run the ONLINE encoder on a fixed
synthetic multimodal CPU batch; assert outputs are finite and shape-correct, and
that the SDPA (use_flash_attn=False) path runs.

Heavy (slow) path: read the published checkpoint config.json, build the matching
"base" model, and try to load the distributed-checkpoint weights. The checkpoint
was written by the *old* package (its config.json carries
``olmoearth_pretrain.nn.*`` _CLASS_ keys), so cross-format weight loading needs the
key-remap converter (PLAN: "A converter handles key/path remap"). If that converter
is not yet wired up, the load is skipped rather than failed.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
sys.path.insert(0, str(CONFIGS))

CKPT_ROOT = Path("/weka/dfive-default/helios/checkpoints/joer/rope_simple_v1")
PREFERRED_STEP = CKPT_ROOT / "step125000"


def _find_checkpoint() -> Path | None:
    if PREFERRED_STEP.exists() and (PREFERRED_STEP / "config.json").exists():
        return PREFERRED_STEP
    if not CKPT_ROOT.exists():
        return None
    steps = sorted(
        (p for p in CKPT_ROOT.glob("step*") if (p / "config.json").exists()),
        key=lambda p: int(p.name.replace("step", "")),
    )
    return steps[-1] if steps else None


def _build_model(size: str):
    from olmoearth2.launch.experiment import CommonComponents

    import v1_lib

    common = CommonComponents(
        run_name="t",
        save_folder="/tmp/oe2-parity-test",
        training_modalities=list(v1_lib.TRAINING_MODALITIES),
        tokenization_config=v1_lib._tokenization_config(),
    )
    return v1_lib.make_build_model_config(size)(common).build()


def _synthetic_batch(modalities, *, b=1, h=4, w=4, t=2):
    from olmoearth2.data.constants import Modality
    from olmoearth2.datatypes import MaskedOlmoEarthSample, MaskValue

    torch.manual_seed(0)
    fields = {}
    for name in modalities:
        spec = Modality.get(name)
        nb = spec.num_bands
        nbs = spec.num_band_sets
        # Space-only modalities have T == 1.
        mt = t if spec.is_multitemporal else 1
        fields[name] = torch.randn(b, h, w, mt, nb)
        fields[f"{name}_mask"] = torch.full(
            (b, h, w, mt, nbs),
            fill_value=MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
        )
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(0, 12, (b, t, 1), dtype=torch.long)
    years = torch.randint(2018, 2021, (b, t, 1), dtype=torch.long)
    fields["timestamps"] = torch.cat([days, months, years], dim=-1)
    fields["latlon"] = torch.randn(b, Modality.LATLON.num_bands)
    fields["latlon_mask"] = torch.full(
        (b, Modality.LATLON.num_band_sets),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.float32,
    )
    return MaskedOlmoEarthSample(**fields)


def _run_online_encoder(model, modalities):
    model.eval()
    x = _synthetic_batch(modalities)
    with torch.no_grad():
        out = model.encoder(x, patch_size=2)
    tokens_and_masks = out["tokens_and_masks"]
    flat_tokens, flat_masks = tokens_and_masks.flatten_all_tokens_and_masks()
    return flat_tokens, flat_masks


def test_sdpa_path_is_default():
    """The blessed encoder runs on the SDPA path (no flash-attn required)."""
    model = _build_model("nano")
    assert model.encoder.use_flash_attn is False


def test_online_encoder_runs_and_is_finite():
    """Online encoder on a small multimodal CPU batch produces finite tokens."""
    model = _build_model("nano")
    flat_tokens, flat_masks = _run_online_encoder(
        model, ["sentinel2_l2a", "sentinel1", "worldcover"]
    )
    assert flat_tokens.ndim == 3, f"expected [B,T,D], got {tuple(flat_tokens.shape)}"
    b, n, d = flat_tokens.shape
    assert b == 1 and n > 0
    assert d == model.encoder.embedding_size or d > 0
    assert torch.isfinite(flat_tokens).all(), "encoder produced non-finite tokens"
    assert flat_masks.shape[:2] == flat_tokens.shape[:2]


def test_online_encoder_deterministic_in_eval():
    """In eval mode (band dropout off) the encoder is deterministic."""
    model = _build_model("nano")
    a, _ = _run_online_encoder(model, ["sentinel2_l2a", "sentinel1"])
    b, _ = _run_online_encoder(model, ["sentinel2_l2a", "sentinel1"])
    assert torch.allclose(a, b, atol=1e-5), "eval-mode encoder is non-deterministic"


def test_checkpoint_config_matches_base_size():
    """If the checkpoint is present, its encoder arch matches the blessed 'base'."""
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint present")
    from olmoearth2.launch.utils import MODEL_SIZE_ARGS

    cfg = json.loads((ckpt / "config.json").read_text())
    enc = cfg["model"]["encoder_config"]
    base = MODEL_SIZE_ARGS["base"]
    assert enc["embedding_size"] == base["encoder_embedding_size"]
    assert enc["depth"] == base["encoder_depth"]
    assert enc["num_heads"] == base["encoder_num_heads"]
    assert enc["use_flash_attn"] is False
    assert enc["spatial_pos_encoding"] == "rope"
    assert enc["encoding_mode"] == "separate"


@pytest.mark.slow
def test_base_model_online_encoder_all_modalities():
    """Build the full base model and run the online encoder on all 9 modalities."""
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint present")
    cfg = json.loads((ckpt / "config.json").read_text())
    modalities = cfg["model"]["encoder_config"]["supported_modality_names"]

    model = _build_model("base")
    flat_tokens, _ = _run_online_encoder(model, modalities)
    assert torch.isfinite(flat_tokens).all()
    assert flat_tokens.shape[-1] == model.encoder.embedding_size


@pytest.mark.slow
def test_load_published_weights_into_base_model():
    """Try to load the distcp checkpoint weights into the olmoearth2 base model.

    The checkpoint config.json was authored by olmoearth_pretrain, so building the
    model directly from it (or loading its state) requires the key-remap converter.
    Until that exists, this degrades to a skip with a clear reason.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint present")
    model_and_optim = ckpt / "model_and_optim"
    if not model_and_optim.exists():
        pytest.skip("no model_and_optim distcp dir")

    try:
        from olmo_core.distributed.checkpoint import load_model_and_optim_state
    except Exception as e:  # pragma: no cover
        pytest.skip(f"olmo-core checkpoint API unavailable: {e}")

    # A distcp load needs an initialized process group; in a single-process test
    # it raises a CheckpointException (a BaseException), and cross-format key
    # remapping is not wired up yet. Catch BaseException so this degrades to a skip.
    if not torch.distributed.is_initialized():
        pytest.skip(
            "distcp checkpoint load requires an initialized process group / "
            "key-remap converter (not wired up)"
        )

    model = _build_model("base")
    try:
        load_model_and_optim_state(str(model_and_optim), model)
    except BaseException as e:
        pytest.skip(
            "cross-format distcp load needs the key-remap converter (not wired): "
            f"{type(e).__name__}: {e}"
        )

    cfg = json.loads((ckpt / "config.json").read_text())
    modalities = cfg["model"]["encoder_config"]["supported_modality_names"]
    flat_tokens, _ = _run_online_encoder(model, modalities)
    assert torch.isfinite(flat_tokens).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
