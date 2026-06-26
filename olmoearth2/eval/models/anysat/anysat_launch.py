"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth2.eval.models import AnySatConfig
from olmoearth2.launch.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> AnySatConfig:
    """Build the model config for an experiment."""
    model_config = AnySatConfig()
    return model_config
