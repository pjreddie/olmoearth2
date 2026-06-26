"""OlmoEarth2 v1 — base size (768-d, 12 enc / 4 dec). The blessed pretrain config."""

from v1_lib import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
    make_build_common_components,
    make_build_model_config,
    make_build_trainer_config,
)

from olmoearth2.launch.experiment import main

WANDB_PROJECT = "olmoearth2_v1_base"


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=make_build_common_components(WANDB_PROJECT),
        model_config_builder=make_build_model_config("base_shallow_decoder"),
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=make_build_trainer_config(
            wandb_project=WANDB_PROJECT, wandb_enabled=True
        ),
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
