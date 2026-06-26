"""Studio Dataset Ingestion Module (Internal Use Only).

This module provides tooling to ingest datasets from the Studio platform
into the OlmoEarth evaluation system. It handles:

1. Copying data from GCS to Weka
2. Computing normalization statistics (via band_stats.py)
3. Registering datasets in the eval registry

⚠️  INTERNAL USE ONLY - This is for AI2 internal workflows.

For external users: Set OLMOEARTH_EVAL_DATASETS env var to point to your
local directory containing downloaded rslearn datasets.

Usage:
    python -m olmoearth2.eval.studio_ingest.cli ingest ...

    # Compute band stats separately:
    python -m olmoearth2.eval.studio_ingest.band_stats ...

See README.md in this directory for full documentation.
"""

from olmoearth2.eval.studio_ingest.registry import (
    Registry,
    get_dataset_entry,
    list_dataset_names,
    load_registry,
)
from olmoearth2.eval.studio_ingest.schema import EvalDatasetEntry

__all__ = [
    "EvalDatasetEntry",
    "Registry",
    "get_dataset_entry",
    "list_dataset_names",
    "load_registry",
]
