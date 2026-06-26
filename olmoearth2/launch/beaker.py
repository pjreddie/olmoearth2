"""Beaker launch glue, isolated so core/training imports never pull in heavy.

Importing ``olmo_core.launch.beaker`` transitively imports
``google.cloud.compute_v1`` (via ``select_beaker_hosts``), which is extremely
slow to import in some environments. Keeping all of that here means local
training (``train_single``) and inference never pay that cost — these symbols
are only imported when actually launching on Beaker.
"""

import logging
import os
from dataclasses import dataclass

from olmo_core.internal.common import get_beaker_username
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerPriority,
    BeakerWekaBucket,
    ExperimentSpec,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

from olmoearth2.launch.common import BUDGET, WORKSPACE

logger = logging.getLogger(__name__)

DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET = BeakerWekaBucket(
    "dfive-default", "/weka/dfive-default"
)


@dataclass
class OlmoEarthBeakerLaunchConfig(BeakerLaunchConfig):
    """Extend BeakerLaunchConfig with a hostnames option to target hosts."""

    hostnames: list[str] | None = None

    def build_experiment_spec(
        self, torchrun: bool = True, entrypoint: str | None = None
    ) -> ExperimentSpec:
        """Build the experiment spec, optionally pinning specific hostnames."""
        spec = super().build_experiment_spec(torchrun, entrypoint)
        if self.hostnames:
            constraints = spec.tasks[0].constraints
            constraints.cluster = None
            constraints.hostname = self.hostnames
        return spec


def set_nccl_debug_env_vars(
    nccl_debug: bool, local: bool = False
) -> list[BeakerEnvVar] | None:
    """Set NCCL debug env vars (returns Beaker env vars, or sets them locally)."""
    nccl_settings = {
        "NCCL_DEBUG": "DETAIL" if nccl_debug else "WARN",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000000000" if nccl_debug else "0",
        "TORCH_NCCL_BLOCKING_WAIT": "1" if nccl_debug else "0",
    }
    if not local:
        return [BeakerEnvVar(name=k, value=v) for k, v in nccl_settings.items()]
    for k, v in nccl_settings.items():
        os.environ[k] = v
    return None


def build_launch_config(
    *,
    name: str,
    cmd: list[str],
    clusters: list[str] | str,
    task_name: str = "train",
    workspace: str = WORKSPACE,
    budget: str = BUDGET,
    nccl_debug: bool = False,
) -> OlmoEarthBeakerLaunchConfig:
    """Build a Beaker launch config for an OlmoEarth experiment."""
    if isinstance(clusters, str):
        clusters = [clusters]
    weka_buckets: list[BeakerWekaBucket] = [DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET]
    for c in clusters:
        if "augusta" in c:
            if len(clusters) > 1:
                raise ValueError(
                    "Jobs targeting Augusta should not target other clusters since "
                    "Weka will not be mounted"
                )
            weka_buckets = []

    beaker_user = get_beaker_username()
    env_vars = [
        BeakerEnvVar(
            name="GOOGLE_APPLICATION_CREDENTIALS", value="/etc/gcp_credentials.json"
        ),
    ]
    nccl_debug_env_vars = set_nccl_debug_env_vars(nccl_debug=nccl_debug)
    if nccl_debug_env_vars is not None:
        env_vars.extend(nccl_debug_env_vars)

    return OlmoEarthBeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters,
        weka_buckets=weka_buckets,
        beaker_image=f"petew/{OLMoCoreBeakerImage.stable_cu128}",
        num_nodes=1,
        num_gpus=1,
        shared_memory="256GiB",
        shared_filesystem=True,
        allow_dirty=False,
        priority=BeakerPriority.high,
        env_vars=env_vars,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(
                name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"
            ),  # nosec
            BeakerEnvSecret(
                name="GITHUB_TOKEN", secret=f"{beaker_user}_GITHUB_TOKEN"
            ),  # nosec
            BeakerEnvSecret(
                name="GCP_CREDENTIALS", secret="HELIOS_GCP_CREDENTIALS"
            ),  # nosec
        ],
        setup_steps=[
            'echo "$GCP_CREDENTIALS" > $GOOGLE_APPLICATION_CREDENTIALS',
            "conda install gh --channel conda-forge",
            "gh auth status",
            "gh repo clone $REPO_URL .",
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            "pip install uv",
            'export PATH="/root/.local/bin:$PATH" ',
            # SDPA attention path (default) — no flash-attn build required.
            "uv sync --extra training --extra beaker",
            "venv_path=$(uv run python -c 'import sys; print(sys.executable)')",
            'source "$(dirname "$venv_path")/activate"',
            "uv pip show torch",
            "uv run python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())'",
        ],
    )
