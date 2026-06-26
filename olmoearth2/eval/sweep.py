"""Consolidated eval-sweep entry point (PLAN Phase 7).

The PLAN folds ``all_evals`` / ``full_eval_sweep`` / ``full_eval_sweep_finetune``
/ ``checkpoint_sweep_evals`` into one surface. The heavy implementations live in
``olmoearth2.launch.*`` (they depend on the ``[eval]`` extra); this module is the
single documented place to reach them, imported lazily so the core install does
not pull eval/baseline deps.

Sweep kinds:
  * ``linear_probe`` / ``knn`` hyperparameter sweep over eval tasks
    (``full_eval_sweep``)
  * ``finetune`` sweep (``full_eval_sweep_finetune``)
  * ``checkpoint`` sweep — eval many steps of one run (``checkpoint_sweep_evals``)
"""

from __future__ import annotations

from typing import Any


def run_probe_sweep(*args: Any, **kwargs: Any) -> Any:
    """Run the linear-probe / knn hyperparameter sweep."""
    from olmoearth2.launch.full_eval_sweep import main

    return main(*args, **kwargs)


def run_finetune_sweep(*args: Any, **kwargs: Any) -> Any:
    """Run the fine-tune sweep."""
    from olmoearth2.launch.full_eval_sweep_finetune import main

    return main(*args, **kwargs)


def run_checkpoint_sweep(*args: Any, **kwargs: Any) -> Any:
    """Run downstream eval across many checkpoints of a single run."""
    from olmoearth2.launch.checkpoint_sweep_evals import evaluate_checkpoints

    return evaluate_checkpoints(*args, **kwargs)
