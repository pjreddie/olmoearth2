"""Shared builders for the blessed OlmoEarth2 v1 model.

This is the crystallized config from ``v1_1/rope_simple_encodings.py`` (v1.1
base + 2D RoPE + simple separate-path encodings). Builders are parameterized by
model size and a few flags so the per-size entrypoints stay thin.

Storage root is config/env-driven: set ``OLMOEARTH2_H5_DIR`` to point at the
training H5 directory (defaults to the blessed corpus on Weka).
"""

from __future__ import annotations

import logging
import os

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth2.data.constants import Modality
from olmoearth2.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth2.data.dataset import OlmoEarthDatasetConfig
from olmoearth2.launch.experiment import CommonComponents, OlmoEarthVisualizeConfig, SubCmd
from olmoearth2.launch.common import build_common_components as build_common_components_default
from olmoearth2.launch.utils import MODEL_SIZE_ARGS
from olmoearth2.model.flexihelios import EncoderConfig, PredictorConfig
from olmoearth2.model.latent_mim import LatentMIMConfig
from olmoearth2.model.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth2.train.callbacks import (
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth2.train.loss import LossConfig
from olmoearth2.train.masking import MaskingConfig
from olmoearth2.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# --- topology ---
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
RANDOM_BAND_DROPOUT_MAX_RATE = 0.2
PATCH_EMBED_HIDDEN_SIZES: list[int] = [64]

# --- 2D RoPE spatial + simple separate-path encodings (rope_simple_encodings) ---
SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 0.25
ENCODING_MODE = "separate"
CHANNEL_ENCODING_DIM = 128
TEMPORAL_ENCODING_DIM = 3
LATLON_ENCODING_DIM = 3
TEMPORAL_ENCODING_TYPE = "simple"
LATLON_ENCODING_TYPE = "simple"
LATLON_DROPOUT_RATE = 0.5

DEFAULT_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)


def h5_dir() -> str:
    """Resolve the training H5 directory from env (falls back to blessed corpus)."""
    return os.environ.get("OLMOEARTH2_H5_DIR", DEFAULT_H5_DIR)


TRAINING_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

BAND_DROPOUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
]

S2_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        [
            "B02", "B03", "B04", "B08", "B05", "B06",
            "B07", "B8A", "B11", "B12", "B01", "B09",
        ],
    ]
)

LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


def _masking_config(tokenization_config: TokenizationConfig | None = None) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


def _apply_encodings(cfg) -> None:
    cfg.spatial_pos_encoding = SPATIAL_POS_ENCODING
    cfg.rope_base = ROPE_BASE
    cfg.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    cfg.encoding_mode = ENCODING_MODE
    cfg.channel_encoding_dim = CHANNEL_ENCODING_DIM
    cfg.temporal_encoding_dim = TEMPORAL_ENCODING_DIM
    cfg.latlon_encoding_dim = LATLON_ENCODING_DIM
    cfg.temporal_encoding_type = TEMPORAL_ENCODING_TYPE
    cfg.latlon_encoding_type = LATLON_ENCODING_TYPE
    cfg.latlon_dropout_rate = LATLON_DROPOUT_RATE


def make_build_common_components(wandb_project: str):
    """Build the common-components builder (sets the full 9-modality list)."""

    def build_common_components(
        script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
    ) -> CommonComponents:
        config = build_common_components_default(script, cmd, run_name, cluster, overrides)
        config.training_modalities = list(TRAINING_MODALITIES)
        config.tokenization_config = _tokenization_config()
        return config

    return build_common_components


def make_build_model_config(model_size_key: str):
    """Build the model-config builder for the given size."""

    def build_model_config(common: CommonComponents) -> LatentMIMConfig:
        model_size = MODEL_SIZE_ARGS[model_size_key]
        encoder_config = EncoderConfig(
            embedding_size=model_size["encoder_embedding_size"],
            num_heads=model_size["encoder_num_heads"],
            depth=model_size["encoder_depth"],
            mlp_ratio=model_size["mlp_ratio"],
            supported_modality_names=common.training_modalities,
            max_patch_size=MAX_PATCH_SIZE,
            drop_path=0.1,
            max_sequence_length=12,
            tokenization_config=common.tokenization_config,
            band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
            random_band_dropout=True,
            band_dropout_modalities=BAND_DROPOUT_MODALITIES,
            patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
        )
        decoder_config = PredictorConfig(
            encoder_embedding_size=model_size["encoder_embedding_size"],
            decoder_embedding_size=model_size["decoder_embedding_size"],
            depth=model_size["decoder_depth"],
            mlp_ratio=model_size["mlp_ratio"],
            num_heads=model_size["decoder_num_heads"],
            supported_modality_names=common.training_modalities,
            max_sequence_length=12,
            tokenization_config=common.tokenization_config,
        )
        _apply_encodings(encoder_config)
        _apply_encodings(decoder_config)
        return LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

    return build_model_config


def build_train_module_config(common: CommonComponents) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the contrastive latent-MIM train module config (frozen target)."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={"type": "InfoNCE", "weight": 0.05}
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        masking_config=_masking_config(common.tokenization_config),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config (H5 reader)."""
    return OlmoEarthDatasetConfig(
        h5py_dir=h5_dir(),
        training_modalities=common.training_modalities,
    )


def build_eval_tasks(eval_interval_steps: int = 4000) -> dict:
    """Build the in-loop downstream-eval task set (Phase 4).

    Imported lazily by the trainer builder so the ``[training]``-only install
    does not need the eval stack (geobench / sklearn / rioxarray).
    """
    from olmoearth2.eval.datasets.normalize import NormMethod
    from olmoearth2.eval.metrics import EvalMetric
    from olmoearth2.model.flexi_vit import PoolingType
    from olmoearth2.train.callbacks.evaluator_callback import (
        DownstreamTaskConfig,
        EvalMode,
    )

    return {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.KNN,
            primary_metric=EvalMetric.ACCURACY,
            eval_interval=Duration.steps(eval_interval_steps),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            probe_lr=0.01,
            eval_interval=Duration.steps(eval_interval_steps),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.LINEAR_PROBE,
            primary_metric=EvalMetric.MICRO_F1,
        ),
    }


def make_build_trainer_config(
    *,
    wandb_project: str,
    wandb_enabled: bool,
    downstream_eval: bool = False,
    eval_interval_steps: int = 4000,
):
    """Build the trainer-config builder.

    Set ``downstream_eval=True`` to attach the in-loop ``DownstreamEvaluator``
    (Phase 4); requires the ``[eval]`` extra.
    """

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        MAX_DURATION = Duration.epochs(300)
        checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
        wandb_callback = OlmoEarthWandBCallback(
            name=common.run_name,
            project=wandb_project,
            entity="eai-ai2",
            enabled=wandb_enabled,
        )
        trainer = (
            TrainerConfig(
                work_dir=common.save_folder,
                load_strategy=LoadStrategy.if_available,
                save_folder=common.save_folder,
                cancel_check_interval=25,
                metrics_collect_interval=10,
                max_duration=MAX_DURATION,
                checkpointer=checkpointer_config,
            )
            .with_callback("wandb", wandb_callback)
            .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
            .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
            .with_callback("beaker", BeakerCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=5000,
                    ephemeral_save_interval=250,
                ),
            )
        )
        if downstream_eval:
            # Lazy import keeps the eval stack optional.
            from olmoearth2.train.callbacks import DownstreamEvaluatorCallbackConfig

            trainer = trainer.with_callback(
                "downstream_evaluator",
                DownstreamEvaluatorCallbackConfig(
                    tasks=build_eval_tasks(eval_interval_steps)
                ),
            )
        return trainer

    return build_trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )
