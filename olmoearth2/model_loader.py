"""Back-compat shim — moved to :mod:`olmoearth2.inference.model_loader`."""

from olmoearth2.inference.model_loader import *  # noqa: F401,F403
from olmoearth2.inference.model_loader import (  # noqa: F401
    ModelID,
    load_model_from_id,
    load_model_from_path,
)
