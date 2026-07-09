"""Samplers for cloud Zarr lazy datasets."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class CloudZarrChunkBatchSampler(Sampler[list[int]]):
    """Build batches that stay near source/basin/time chunks.

    The sampler expects a ``CloudZarrLazyDataset``-like object exposing
    ``_lookup_sources``, ``_lookup_basins`` and ``_lookup_starts``.  Keeping a
    batch close to one source/time chunk avoids pathological cloud Zarr random
    reads while still shuffling chunk groups each epoch.
    """

    is_batch_sampler = True

    def __init__(
        self,
        dataset,
        batch_size: int,
        time_chunk_size: int | None = None,
        basin_chunk_size: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        sampling: str | None = None,
        max_batches_per_epoch: int | None = None,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        seq_len = int(dataset.rho + dataset.horizon)
        self.time_chunk_size = int(time_chunk_size or max(1, seq_len, self.batch_size))
        self.basin_chunk_size = int(max(1, basin_chunk_size))
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sampling = sampling or getattr(dataset, "lazy_cfg", {}).get(
            "sampling", "weighted_by_sample"
        )
        self.max_batches_per_epoch = max_batches_per_epoch
        self.seed = seed
        self._epoch = 0
        self._batches_by_source = self._build_batches_by_source()
        if len(dataset) > 0 and self.drop_last and len(self) == 0:
            raise ValueError(
                "CloudZarrChunkBatchSampler produced zero batches. "
                f"batch_size={self.batch_size}, "
                f"time_chunk_size={self.time_chunk_size}, "
                f"basin_chunk_size={self.basin_chunk_size}. "
                "Increase time_chunk_size/basin_chunk_size or disable drop_last."
            )

    def _build_batches_by_source(self) -> dict[int, list[list[int]]]:
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        sources = np.asarray(self.dataset._lookup_sources)
        basins = np.asarray(self.dataset._lookup_basins)
        starts = np.asarray(self.dataset._lookup_starts)
        for idx in range(len(self.dataset)):
            key = (
                int(sources[idx]),
                int(basins[idx]) // self.basin_chunk_size,
                int(starts[idx]) // self.time_chunk_size,
            )
            groups[key].append(idx)

        rng = random.Random(self.seed)
        batches_by_source: dict[int, list[list[int]]] = defaultdict(list)
        for key, indices in groups.items():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches_by_source[key[0]].append(batch)
        return dict(batches_by_source)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(None if self.seed is None else self.seed + self._epoch)
        self._epoch += 1
        batches_by_source = {
            source: [batch.copy() for batch in batches]
            for source, batches in self._batches_by_source.items()
        }
        for batches in batches_by_source.values():
            if self.shuffle:
                rng.shuffle(batches)

        if self.sampling == "balanced_by_dataset":
            batches = self._balanced_batches(batches_by_source, rng)
        else:
            batches = list(itertools.chain.from_iterable(batches_by_source.values()))
            if self.shuffle:
                rng.shuffle(batches)

        if self.max_batches_per_epoch is not None:
            batches = batches[: self.max_batches_per_epoch]
        return iter(batches)

    def _balanced_batches(
        self,
        batches_by_source: dict[int, list[list[int]]],
        rng: random.Random,
    ) -> list[list[int]]:
        non_empty = {k: v for k, v in batches_by_source.items() if v}
        if not non_empty:
            return []
        target = max(len(batches) for batches in non_empty.values())
        balanced = []
        for source, batches in non_empty.items():
            if len(batches) < target:
                repeats = math.ceil(target / len(batches))
                expanded = (batches * repeats)[:target]
            else:
                expanded = batches[:target]
            balanced.extend(expanded)
        if self.shuffle:
            rng.shuffle(balanced)
        return balanced

    def __len__(self) -> int:
        if self.sampling == "balanced_by_dataset":
            non_empty = [batches for batches in self._batches_by_source.values() if batches]
            if not non_empty:
                return 0
            length = max(len(batches) for batches in non_empty) * len(non_empty)
        else:
            length = sum(len(batches) for batches in self._batches_by_source.values())
        if self.max_batches_per_epoch is not None:
            return min(length, self.max_batches_per_epoch)
        return length
