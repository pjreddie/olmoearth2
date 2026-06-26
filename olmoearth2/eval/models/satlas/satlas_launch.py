"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth2.eval.models.satlas.satlas import SatlasConfig
from olmoearth2.launch.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> SatlasConfig:
    """Build the model config for an experiment."""
    model_config = SatlasConfig()
    return model_config
