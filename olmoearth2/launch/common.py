"""Common utilities for launching experiments.

Beaker imports are deliberately deferred into the functions that need them so
that local training and inference do not pay the (very slow) cost of importing
``olmo_core.launch.beaker`` / ``google.cloud.compute``.
"""

import logging

from upath import UPath

from olmoearth2.data.constants import Modality
from olmoearth2.launch.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
)

logger = logging.getLogger(__name__)
BUDGET = "ai2/atec-olmoearth"
WORKSPACE = "ai2/earth-systems"
PROJECT_NAME = "olmoearth2"

WEKA_CLUSTER_NAMES = [
    "jupiter",
    "saturn",
    "neptune",
    "ceres",
    "triton",
    "titan",
    "rhea",
]

LOCAL_CLUSTER_NAME = "local"
ANONYMOUS_USER = "anonymous"


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=50,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        std_multiplier=2.0,
    )


def get_root_dir(cluster: str) -> str:
    """Get the root directory where the save_folder will be stored."""
    if any(weka_cluster_name in cluster for weka_cluster_name in WEKA_CLUSTER_NAMES):
        root_dir = f"/weka/dfive-default/{PROJECT_NAME}"
    elif "augusta" in cluster:
        root_dir = f"/unused/{PROJECT_NAME}"
    elif LOCAL_CLUSTER_NAME in cluster:
        root_dir = "./local_output"
    else:
        raise ValueError(f"Cluster {cluster} is not supported")
    return root_dir


def extract_nccl_debug_from_overrides(overrides: list[str]) -> bool:
    """Extract the nccl_debug flag from the overrides."""
    for override in overrides:
        if override.startswith("--common.nccl_debug="):
            return override.split("=")[1].lower() in ("true", "1", "yes")
    return False


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    TRAINING_MODALITIES = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
    ]
    if cmd == SubCmd.launch:
        cmd_to_launch = SubCmd.train
    elif cmd == SubCmd.launch_evaluate:
        cmd_to_launch = SubCmd.evaluate
    elif cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep
    elif cmd == SubCmd.launch_benchmark:
        cmd_to_launch = SubCmd.benchmark
    else:
        cmd_to_launch = cmd

    nccl_debug = extract_nccl_debug_from_overrides(overrides)
    if cluster == LOCAL_CLUSTER_NAME:
        # Defer the (heavy) beaker import; only needed for env-var side effects.
        from olmoearth2.launch.beaker import set_nccl_debug_env_vars

        set_nccl_debug_env_vars(nccl_debug=nccl_debug, local=True)
        launch_config = None
        beaker_user = ANONYMOUS_USER
    else:
        from olmo_core.internal.common import get_beaker_username
        from olmo_core.launch.beaker import is_running_in_beaker

        from olmoearth2.launch.beaker import build_launch_config

        launch_config = build_launch_config(
            name=f"{run_name}-{cmd_to_launch}",
            cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
            clusters=cluster,
            nccl_debug=nccl_debug,
        )
        if cmd == SubCmd.launch:
            launch_config.retries = 2
        beaker_user = get_beaker_username() or ANONYMOUS_USER
        if is_running_in_beaker() and beaker_user is None:
            raise ValueError(
                "Failed to get Beaker username. Make sure you are authenticated "
                "with Beaker if you are not running on a local cluster."
            )

    root_dir = get_root_dir(cluster)
    return CommonComponents(
        run_name=run_name,
        save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
        launch=launch_config,
        training_modalities=TRAINING_MODALITIES,
    )
