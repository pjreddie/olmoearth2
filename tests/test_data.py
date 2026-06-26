"""H5 dataset reader: load one sample and check shapes / dtype / normalization.

Skipped gracefully when the Weka H5 directory is absent (set OLMOEARTH2_H5_DIR to
override). CPU-only; reads a single sample.
"""

import os
from pathlib import Path

import pytest
import torch

from olmoearth2.data.constants import Modality

DEFAULT_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
H5_DIR = os.environ.get("OLMOEARTH2_H5_DIR", DEFAULT_H5_DIR)

TRAINING_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

pytestmark = pytest.mark.skipif(
    not Path(H5_DIR).exists(), reason=f"H5 dir not present: {H5_DIR}"
)


@pytest.fixture(scope="module")
def one_sample():
    from olmoearth2.data.dataset import GetItemArgs, OlmoEarthDatasetConfig

    ds = OlmoEarthDatasetConfig(
        h5py_dir=H5_DIR, training_modalities=TRAINING_MODALITIES
    ).build()
    ds.prepare()
    assert len(ds) > 0
    args = GetItemArgs(idx=0, patch_size=4, sampled_hw_p=8, token_budget=2250)
    _, sample = ds[args]
    return sample


def test_sample_has_expected_modalities(one_sample):
    """Every requested training modality is present in the loaded sample."""
    present = set(one_sample.modalities)
    for m in TRAINING_MODALITIES:
        assert m in present, f"missing modality {m}"


def test_modality_shapes_are_hwtc(one_sample):
    """Spatial modalities are [H, W, T, C] with C == num_bands."""
    for m in one_sample.modalities:
        t = torch.as_tensor(getattr(one_sample, m))
        assert t.ndim == 4, f"{m} should be [H,W,T,C], got {tuple(t.shape)}"
        h, w, time, c = t.shape
        assert h > 0 and w > 0 and time >= 1
        assert c == Modality.get(m).num_bands, (
            f"{m} channel dim {c} != num_bands {Modality.get(m).num_bands}"
        )


def test_timestamps_shape(one_sample):
    """Timestamps are [T, 3] (day, month, year)."""
    ts = torch.as_tensor(one_sample.timestamps)
    assert ts.ndim == 2 and ts.shape[-1] == 3, f"timestamps shape {tuple(ts.shape)}"


def test_dtype_is_float32(one_sample):
    """All modality tensors are float32."""
    for m in one_sample.modalities:
        t = torch.as_tensor(getattr(one_sample, m))
        assert t.dtype == torch.float32, f"{m} dtype {t.dtype}"


def test_normalization_applied(one_sample):
    """Normalized values are finite and in a sane range (not raw reflectances)."""
    for m in one_sample.modalities:
        t = torch.as_tensor(getattr(one_sample, m))
        assert torch.isfinite(t).all(), f"{m} has non-finite values"
        # Normalized features should be bounded; raw DN values would be in the
        # thousands. Allow generous headroom for clipping at a few std.
        assert t.abs().max() <= 50.0, f"{m} max {t.abs().max()} looks un-normalized"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
