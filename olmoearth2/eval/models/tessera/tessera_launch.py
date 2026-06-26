"""Tessera model launch script for evaluation."""

import logging

from olmoearth2.eval.models.tessera.tessera import TesseraConfig
from olmoearth2.launch.experiment import (
    CommonComponents,
)
from olmoearth2.model.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for Tessera evaluation."""
    model_config = TesseraConfig()
    return model_config
