"""Deprecated module. Please import from olmoearth2.model.flexivit instead.

Maintained for backwards compatibility with old checkpoints.
"""

import sys
import warnings

import olmoearth2.model.flexi_vit as flexivit

from .flexi_vit import *  # noqa: F403

warnings.warn(
    "olmoearth2.model.flexi_vit is deprecated. "
    "Please import from olmoearth2.model.flexivit instead.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = flexivit
