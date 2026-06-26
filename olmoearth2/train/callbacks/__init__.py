"""Callbacks for the trainer specific to OlmoEarth.

``DownstreamEvaluatorCallbackConfig`` is imported lazily because the evaluator
pulls in the eval stack (rioxarray, geobench, ...). Training that doesn't use
the in-loop evaluator (and the lighter ``[training]`` install) therefore does
not need those deps just to import the speed/wandb callbacks.
"""

from typing import TYPE_CHECKING

from .speed_monitor import HeliosSpeedMonitorCallback, OlmoEarthSpeedMonitorCallback
from .wandb import HeliosWandBCallback, OlmoEarthWandBCallback

if TYPE_CHECKING:
    from .evaluator_callback import DownstreamEvaluatorCallbackConfig

__all__ = [
    "DownstreamEvaluatorCallbackConfig",
    "OlmoEarthSpeedMonitorCallback",
    "OlmoEarthWandBCallback",
    "HeliosSpeedMonitorCallback",
    "HeliosWandBCallback",
]


def __getattr__(name: str):
    """Lazily import the evaluator callback (keeps eval deps optional)."""
    if name == "DownstreamEvaluatorCallbackConfig":
        from .evaluator_callback import DownstreamEvaluatorCallbackConfig

        return DownstreamEvaluatorCallbackConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
