"""Baseline eval models, behind per-model optional extras.

Each baseline pulls heavy, model-specific deps (terratorch, timm, claymodel,
satlaspretrain, ...). Imports are therefore **lazy** (PEP 562): importing this
package costs nothing, and a given baseline's deps are only required when that
baseline's symbol is actually accessed. This implements the PLAN's
"opt-in per-model extras" for the 11 baseline adapters.
"""

from enum import StrEnum
from typing import Any

# symbol name -> (submodule, attribute) for lazy resolution
_LAZY: dict[str, tuple[str, str]] = {
    "AnySat": ("anysat.anysat", "AnySat"),
    "AnySatConfig": ("anysat.anysat", "AnySatConfig"),
    "Clay": ("clay.clay", "Clay"),
    "ClayConfig": ("clay.clay", "ClayConfig"),
    "Croma": ("croma.croma", "Croma"),
    "CromaConfig": ("croma.croma", "CromaConfig"),
    "CROMA_SIZES": ("croma.croma", "CROMA_SIZES"),
    "DinoV3Models": ("dinov3.constants", "DinoV3Models"),
    "DINOv3": ("dinov3.dinov3", "DINOv3"),
    "DINOv3Config": ("dinov3.dinov3", "DINOv3Config"),
    "GalileoConfig": ("galileo", "GalileoConfig"),
    "GalileoWrapper": ("galileo", "GalileoWrapper"),
    "Panopticon": ("panopticon.panopticon", "Panopticon"),
    "PanopticonConfig": ("panopticon.panopticon", "PanopticonConfig"),
    "PrestoConfig": ("presto.presto", "PrestoConfig"),
    "PrestoWrapper": ("presto.presto", "PrestoWrapper"),
    "PrithviV2": ("prithviv2.prithviv2", "PrithviV2"),
    "PrithviV2Config": ("prithviv2.prithviv2", "PrithviV2Config"),
    "PrithviV2Models": ("prithviv2.prithviv2", "PrithviV2Models"),
    "Satlas": ("satlas.satlas", "Satlas"),
    "SatlasConfig": ("satlas.satlas", "SatlasConfig"),
    "Terramind": ("terramind.terramind", "Terramind"),
    "TerramindConfig": ("terramind.terramind", "TerramindConfig"),
    "TERRAMIND_SIZES": ("terramind.terramind", "TERRAMIND_SIZES"),
    "Tessera": ("tessera.tessera", "Tessera"),
    "TesseraConfig": ("tessera.tessera", "TesseraConfig"),
}


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


def __getattr__(name: str) -> Any:
    """Lazily import a baseline symbol (keeps each model's deps optional)."""
    import importlib

    if name in _LAZY:
        submod, attr = _LAZY[name]
        module = importlib.import_module(f"olmoearth2.eval.models.{submod}")
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def models_with_multiple_sizes() -> dict[BaselineModelName, Any]:
    """Return the size options per multi-size baseline (resolved lazily)."""
    from olmoearth2.eval.models.galileo.single_file_galileo import (
        MODEL_SIZE_TO_WEKA_PATH,
    )

    return {
        BaselineModelName.CROMA: __getattr__("CROMA_SIZES"),
        BaselineModelName.DINO_V3: list(__getattr__("DinoV3Models")),
        BaselineModelName.GALILEO: list(MODEL_SIZE_TO_WEKA_PATH.keys()),
        BaselineModelName.PRITHVI_V2: list(__getattr__("PrithviV2Models")),
        BaselineModelName.TERRAMIND: __getattr__("TERRAMIND_SIZES"),
    }


_LAUNCH_SCRIPTS: dict[BaselineModelName, str] = {
    BaselineModelName.DINO_V3: "dinov3/dino_v3_launch.py",
    BaselineModelName.GALILEO: "galileo/galileo_launch.py",
    BaselineModelName.PANOPTICON: "panopticon/panopticon_launch.py",
    BaselineModelName.TERRAMIND: "terramind/terramind_launch.py",
    BaselineModelName.SATLAS: "satlas/satlas_launch.py",
    BaselineModelName.CROMA: "croma/croma_launch.py",
    BaselineModelName.CLAY: "clay/clay_launch.py",
    BaselineModelName.PRESTO: "presto/presto_launch.py",
    BaselineModelName.ANYSAT: "anysat/anysat_launch.py",
    BaselineModelName.TESSERA: "tessera/tessera_launch.py",
    BaselineModelName.PRITHVI_V2: "prithviv2/prithviv2_launch.py",
}


def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a baseline model."""
    return f"olmoearth2/eval/models/{_LAUNCH_SCRIPTS[BaselineModelName(model_name)]}"


__all__ = ["BaselineModelName", "models_with_multiple_sizes", "get_launch_script_path", *_LAZY]
