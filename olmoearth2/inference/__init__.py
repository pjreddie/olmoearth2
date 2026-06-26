"""Minimal-dependency inference / embedding API for OlmoEarth2."""

from olmoearth2.inference.model_loader import (
    ModelID,
    load_encoder_from_path,
    load_model_from_id,
    load_model_from_path,
)

__all__ = [
    "ModelID",
    "load_model_from_id",
    "load_model_from_path",
    "load_encoder_from_path",
]
