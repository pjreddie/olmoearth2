"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth2.eval.models import ClayConfig
from olmoearth2.launch.experiment import (
    CommonComponents,
)
from olmoearth2.model.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = ClayConfig()
    return model_config
