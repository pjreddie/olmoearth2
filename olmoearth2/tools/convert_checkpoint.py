"""Convert an olmo-core DCP training checkpoint to an inference export.

A training checkpoint is a distributed checkpoint (``model_and_optim/*.distcp``)
of the full ``LatentMIM`` model (online encoder + frozen target encoder +
predictor). For inference we only need the **online encoder**, so this tool:

  1. rebuilds the full model from the checkpoint's ``config.json``,
  2. loads the DCP weights (requires ``[training]`` / olmo-core),
  3. extracts ``model.encoder`` (dropping the frozen target encoder + predictor),
  4. writes ``<out>/weights.pth`` (encoder state dict) and ``<out>/config.json``
     (the encoder config, with ``_CLASS_`` so it round-trips via
     ``olmoearth2.inference.load_encoder_from_path`` using core deps only).

Usage:
    python -m olmoearth2.tools.convert_checkpoint \
        /weka/.../rope_simple_v1/step125000 /tmp/oe2-export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state

from olmoearth2.config import Config
from olmoearth2.inference.model_loader import patch_legacy_encoder_config

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"


def convert_checkpoint(checkpoint_dir: str | Path, out_dir: str | Path) -> Path:
    """Convert a DCP training checkpoint to an inference encoder export."""
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (checkpoint_dir / CONFIG_FILENAME).open() as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)

    model_config = Config.from_dict(config_dict["model"])
    model = model_config.build()

    dcp_dir = checkpoint_dir / "model_and_optim"
    load_model_and_optim_state(str(dcp_dir), model)

    # Strip the frozen target encoder + predictor: export the online encoder only.
    encoder = model.encoder
    encoder_state = {k: v.cpu() for k, v in encoder.state_dict().items()}
    torch.save(encoder_state, out_dir / WEIGHTS_FILENAME)

    # Persist the encoder config (with _CLASS_) so the inference loader can
    # rebuild the encoder with core deps only.
    encoder_config = model_config.encoder_config
    out_config = {"model": encoder_config.as_dict(include_class_name=True, json_safe=True)}
    with (out_dir / CONFIG_FILENAME).open("w") as f:
        json.dump(out_config, f, indent=2)

    n_params = sum(v.numel() for v in encoder_state.values())
    print(
        f"Exported online encoder ({n_params:,} params, {len(encoder_state)} tensors) "
        f"to {out_dir}"
    )
    return out_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="DCP checkpoint dir (has config.json + model_and_optim/)")
    parser.add_argument("out_dir", help="output directory for the inference export")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_dir, args.out_dir)


if __name__ == "__main__":
    main()
