"""The ``DatasetReader`` protocol — the train-time data contract (PLAN Phase 0.5).

A reader is an indexable source of :class:`~olmoearth2.datatypes.OlmoEarthSample`
objects. The H5 implementation (:class:`olmoearth2.data.dataset.OlmoEarthDataset`)
is the current default; the corpus-v2 implementation
(:class:`olmoearth2.data.corpus_v2.CorpusV2Dataset`) targets the new rslearn
``storage=`` API. Both satisfy this protocol so the dataloader is agnostic to the
backing store.

Contract:
  * ``len(reader)`` returns the number of samples.
  * ``reader[args]`` returns ``(index, OlmoEarthSample)`` where the sample holds
    one ``[H, W, T, C]`` float32 tensor per modality (already normalized), plus
    ``timestamps`` and per-modality ``missing_timesteps_masks``.
  * Normalization is owned by the reader (samples come out normalized).
  * Crop / patch RNG is threaded in via ``GetItemArgs`` so the dataloader, not
    the reader, controls determinism.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from olmoearth2.datatypes import OlmoEarthSample


@runtime_checkable
class DatasetReader(Protocol):
    """Structural protocol for a train-time sample source."""

    def __len__(self) -> int:
        """Return the number of samples available."""
        ...

    def __getitem__(self, args) -> tuple[int, OlmoEarthSample]:
        """Return ``(index, sample)`` for the given get-item args."""
        ...
