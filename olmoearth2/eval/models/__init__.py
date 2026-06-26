"""Models for evals."""

from enum import StrEnum
from typing import Any

from olmoearth2.eval.models.anysat.anysat import AnySat, AnySatConfig
from olmoearth2.eval.models.clay.clay import Clay, ClayConfig
from olmoearth2.eval.models.croma.croma import CROMA_SIZES, Croma, CromaConfig
from olmoearth2.eval.models.dinov3.constants import DinoV3Models
from olmoearth2.eval.models.dinov3.dinov3 import DINOv3, DINOv3Config
from olmoearth2.eval.models.galileo import GalileoConfig, GalileoWrapper
from olmoearth2.eval.models.galileo.single_file_galileo import (
    MODEL_SIZE_TO_WEKA_PATH as GALILEO_MODEL_SIZE_TO_WEKA_PATH,
)
from olmoearth2.eval.models.panopticon.panopticon import (
    Panopticon,
    PanopticonConfig,
)
from olmoearth2.eval.models.presto.presto import PrestoConfig, PrestoWrapper
from olmoearth2.eval.models.prithviv2.prithviv2 import (
    PrithviV2,
    PrithviV2Config,
    PrithviV2Models,
)
from olmoearth2.eval.models.satlas.satlas import Satlas, SatlasConfig
from olmoearth2.eval.models.terramind.terramind import (
    TERRAMIND_SIZES,
    Terramind,
    TerramindConfig,
)
from olmoearth2.eval.models.tessera.tessera import Tessera, TesseraConfig


class BaselineModelName(StrEnum):
    """Enum for baseline model names."""

    DINO_V3 = "dino_v3"
    PANOPTICON = "panopticon"
    GALILEO = "galileo"
    SATLAS = "satlas"
    CROMA = "croma"
    PRESTO = "presto"
    ANYSAT = "anysat"
    TESSERA = "tessera"
    PRITHVI_V2 = "prithvi_v2"
    TERRAMIND = "terramind"
    CLAY = "clay"


MODELS_WITH_MULTIPLE_SIZES: dict[BaselineModelName, Any] = {
    BaselineModelName.CROMA: CROMA_SIZES,
    BaselineModelName.DINO_V3: list(DinoV3Models),
    BaselineModelName.GALILEO: GALILEO_MODEL_SIZE_TO_WEKA_PATH.keys(),
    BaselineModelName.PRITHVI_V2: list(PrithviV2Models),
    BaselineModelName.TERRAMIND: TERRAMIND_SIZES,
}


def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a model."""
    if model_name == BaselineModelName.DINO_V3:
        return "olmoearth2/evals/models/dinov3/dino_v3_launch.py"
    elif model_name == BaselineModelName.GALILEO:
        return "olmoearth2/evals/models/galileo/galileo_launch.py"
    elif model_name == BaselineModelName.PANOPTICON:
        return "olmoearth2/evals/models/panopticon/panopticon_launch.py"
    elif model_name == BaselineModelName.TERRAMIND:
        return "olmoearth2/evals/models/terramind/terramind_launch.py"
    elif model_name == BaselineModelName.SATLAS:
        return "olmoearth2/evals/models/satlas/satlas_launch.py"
    elif model_name == BaselineModelName.CROMA:
        return "olmoearth2/evals/models/croma/croma_launch.py"
    elif model_name == BaselineModelName.CLAY:
        return "olmoearth2/evals/models/clay/clay_launch.py"
    elif model_name == BaselineModelName.PRESTO:
        return "olmoearth2/evals/models/presto/presto_launch.py"
    elif model_name == BaselineModelName.ANYSAT:
        return "olmoearth2/evals/models/anysat/anysat_launch.py"
    elif model_name == BaselineModelName.TESSERA:
        return "olmoearth2/evals/models/tessera/tessera_launch.py"
    elif model_name == BaselineModelName.PRITHVI_V2:
        return "olmoearth2/evals/models/prithviv2/prithviv2_launch.py"
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# TODO: assert that they all store a patch_size variable and supported modalities
__all__ = [
    "Panopticon",
    "PanopticonConfig",
    "GalileoWrapper",
    "GalileoConfig",
    "DINOv3",
    "DINOv3Config",
    "Terramind",
    "TerramindConfig",
    "Satlas",
    "SatlasConfig",
    "Croma",
    "CromaConfig",
    "Clay",
    "ClayConfig",
    "PrestoWrapper",
    "PrestoConfig",
    "AnySat",
    "AnySatConfig",
    "Tessera",
    "TesseraConfig",
    "PrithviV2",
    "PrithviV2Config",
]
