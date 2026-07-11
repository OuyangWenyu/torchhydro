"""Chunk-aware batch sampling for cloud Zarr datasets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class _ChunkGroup:
    source: int
    basin_chunk: int
    time_chunk: int
    start: int
    stop: int
    batch_count: int


class CloudZarrChunkBatchSampler(Sampler[list[int]]):
    """Yield batches aligned with physical basin/time chunks.

    A single compact sorted index is retained instead of one Python list per
    sample. Chunk groups are shuffled between epochs, while all batches from a
    chunk stay adjacent so persistent DataLoader workers can reuse cached data.
    """

    is_batch_sampler = True
    _SAMPLING_MODES = {
        "weighted_by_sample",
        "balanced_by_dataset",
        "weighted_by_dataset",
    }

    def __init__(
        self,
        dataset,
        batch_size: int,
        time_chunk_size: int | None = None,
        basin_chunk_size: int | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        sampling: str | None = None,
        max_batches_per_epoch: int | None = None,
        seed: int | None = None,
        worker_affinity_count: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_batches_per_epoch is not None and max_batches_per_epoch <= 0:
            raise ValueError("max_batches_per_epoch must be positive when set.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.sampling = sampling or getattr(dataset, "lazy_cfg", {}).get(
            "sampling", "weighted_by_sample"
        )
        if self.sampling not in self._SAMPLING_MODES:
            supported = ", ".join(sorted(self._SAMPLING_MODES))
            raise ValueError(
                f"Unsupported sampling mode {self.sampling!r}: {supported}."
            )
        self.max_batches_per_epoch = max_batches_per_epoch
        self.seed = seed
        self.worker_affinity_count = max(1, int(worker_affinity_count))
        self._epoch = 0
        self._requested_time_chunk_size = time_chunk_size
        self._requested_basin_chunk_size = basin_chunk_size
        self.effective_chunk_shapes = self._resolve_chunk_shapes()
        first_shape = self.effective_chunk_shapes.get(0, (1, self.batch_size))
        self.basin_chunk_size = (
            int(basin_chunk_size) if basin_chunk_size is not None else first_shape[0]
        )
        self.time_chunk_size = first_shape[1]
        self._index_order, self._groups_by_source = self._build_compact_groups()
        self._natural_batches_by_source = {
            source: sum(group.batch_count for group in groups)
            for source, groups in self._groups_by_source.items()
        }
        self.epoch_sample_counts: dict[int, int] = {}
        if len(dataset) > 0 and self.drop_last and len(self) == 0:
            raise ValueError(
                "CloudZarrChunkBatchSampler produced zero batches. "
                f"batch_size={self.batch_size}, "
                f"effective_chunk_shapes={self.effective_chunk_shapes}. "
                "Increase chunk sizes or disable drop_last."
            )

    @property
    def _sequence_length(self) -> int:
        return int(self.dataset.rho + self.dataset.horizon)

    @property
    def index_storage_nbytes(self) -> int:
        return self._index_order.nbytes

    def _resolve_chunk_shapes(self) -> dict[int, tuple[int, int]]:
        inferred = getattr(self.dataset, "source_chunk_shapes", {})
        shapes = {}
        for source_idx in range(len(self.dataset.sources)):
            inferred_basin, inferred_time = inferred.get(source_idx, (1, 1))
            basin_size = (
                int(self._requested_basin_chunk_size)
                if self._requested_basin_chunk_size is not None
                else int(inferred_basin)
            )
            time_size = (
                int(self._requested_time_chunk_size)
                if self._requested_time_chunk_size is not None
                else int(inferred_time)
            )
            state = self.dataset.source_states[source_idx]
            assert state.basin_positions is not None
            selected_chunk_ids = state.basin_positions // max(1, basin_size)
            _, selected_counts = np.unique(selected_chunk_ids, return_counts=True)
            max_selected_basins = (
                int(selected_counts.max()) if selected_counts.size else 1
            )
            minimum_time_for_batch = math.ceil(self.batch_size / max_selected_basins)
            shapes[source_idx] = (
                max(1, basin_size),
                max(
                    1,
                    time_size,
                    minimum_time_for_batch,
                    self._sequence_length,
                ),
            )
        return shapes

    def chunk_key(self, index: int) -> tuple[int, int, int]:
        source = int(self.dataset._lookup_sources[index])
        local_basin = int(self.dataset._lookup_basins[index])
        start = int(self.dataset._lookup_starts[index])
        state = self.dataset.source_states[source]
        assert state.basin_positions is not None
        global_basin = int(state.basin_positions[local_basin])
        global_time = state.time_offset + start - int(self.dataset.warmup_length)
        basin_size, time_size = self.effective_chunk_shapes[source]
        return source, global_basin // basin_size, global_time // time_size

    def _build_compact_groups(
        self,
    ) -> tuple[np.ndarray, dict[int, list[_ChunkGroup]]]:
        sample_count = len(self.dataset)
        if sample_count == 0:
            return np.empty(0, dtype=np.uint8), {}

        sources = np.asarray(self.dataset._lookup_sources)
        basins = np.asarray(self.dataset._lookup_basins)
        starts = np.asarray(self.dataset._lookup_starts)
        basin_chunk_ids = np.empty(sample_count, dtype=np.uint32)
        time_chunk_ids = np.empty(sample_count, dtype=np.uint32)

        for source_idx, state in enumerate(self.dataset.source_states):
            left = int(np.searchsorted(sources, source_idx, side="left"))
            right = int(np.searchsorted(sources, source_idx, side="right"))
            if left == right:
                continue
            assert state.basin_positions is not None
            basin_size, time_size = self.effective_chunk_shapes[source_idx]
            local_basins = basins[left:right].astype(np.int64, copy=False)
            global_basins = state.basin_positions[local_basins]
            global_times = (
                state.time_offset
                + starts[left:right].astype(np.int64, copy=False)
                - int(self.dataset.warmup_length)
            )
            basin_chunk_ids[left:right] = global_basins // basin_size
            time_chunk_ids[left:right] = global_times // time_size

        order = np.lexsort((time_chunk_ids, basin_chunk_ids, sources))
        sorted_sources = sources[order]
        sorted_basins = basin_chunk_ids[order]
        sorted_times = time_chunk_ids[order]
        changes = (
            np.flatnonzero(
                (sorted_sources[1:] != sorted_sources[:-1])
                | (sorted_basins[1:] != sorted_basins[:-1])
                | (sorted_times[1:] != sorted_times[:-1])
            )
            + 1
        )
        boundaries = np.concatenate(([0], changes, [sample_count]))
        compact_order = self._compact_index(order)
        groups_by_source: dict[int, list[_ChunkGroup]] = {}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            size = int(stop - start)
            batch_count = (
                size // self.batch_size
                if self.drop_last
                else math.ceil(size / self.batch_size)
            )
            if batch_count == 0:
                continue
            source = int(sorted_sources[start])
            groups_by_source.setdefault(source, []).append(
                _ChunkGroup(
                    source=source,
                    basin_chunk=int(sorted_basins[start]),
                    time_chunk=int(sorted_times[start]),
                    start=int(start),
                    stop=int(stop),
                    batch_count=batch_count,
                )
            )
        return compact_order, groups_by_source

    @staticmethod
    def _compact_index(values: np.ndarray) -> np.ndarray:
        maximum = int(values.max()) if values.size else 0
        if maximum <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        elif maximum <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif maximum <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            dtype = np.uint64
        return values.astype(dtype, copy=False)

    def __iter__(self) -> Iterator[list[int]]:
        self.epoch_sample_counts = {}
        epoch_seed = (
            random.randrange(2**32) if self.seed is None else self.seed + self._epoch
        )
        self._epoch += 1
        rng = random.Random(epoch_seed)
        np_rng = np.random.default_rng(epoch_seed)

        if self.sampling == "weighted_by_sample":
            groups = [
                group
                for source_groups in self._groups_by_source.values()
                for group in source_groups
            ]
            if self.shuffle:
                rng.shuffle(groups)
            if not self.drop_last:
                if self.worker_affinity_count > 1:
                    yield from self._affinity_group_batches(groups, np_rng)
                else:
                    yield from self._coalesced_group_batches(groups, np_rng)
                return
            yielded = 0
            for group in groups:
                for batch in self._group_batches(group, np_rng):
                    if (
                        self.max_batches_per_epoch is not None
                        and yielded >= self.max_batches_per_epoch
                    ):
                        return
                    yielded += 1
                    self._record_batch_sources(batch)
                    yield batch
            return

        targets = self._target_batches_by_source()
        runs: list[tuple[_ChunkGroup, int]] = []
        for source, target in targets.items():
            remaining = target
            groups = self._groups_by_source.get(source, [])
            while remaining > 0 and groups:
                cycle = groups.copy()
                if self.shuffle:
                    rng.shuffle(cycle)
                for group in cycle:
                    take = min(group.batch_count, remaining)
                    runs.append((group, take))
                    remaining -= take
                    if remaining == 0:
                        break
        if self.shuffle:
            rng.shuffle(runs)
        for group, take in runs:
            for batch_pos, batch in enumerate(self._group_batches(group, np_rng)):
                if batch_pos >= take:
                    break
                self._record_batch_sources(batch)
                yield batch

    def _record_batch_sources(self, batch: list[int]) -> None:
        sources, counts = np.unique(
            np.asarray(self.dataset._lookup_sources)[batch], return_counts=True
        )
        for source, count in zip(sources, counts):
            source = int(source)
            self.epoch_sample_counts[source] = self.epoch_sample_counts.get(
                source, 0
            ) + int(count)

    def _group_batches(
        self,
        group: _ChunkGroup,
        rng: np.random.Generator,
    ) -> Iterator[list[int]]:
        indices = self._index_order[group.start : group.stop].astype(
            np.int64,
            copy=True,
        )
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch.tolist()

    def _coalesced_group_batches(
        self,
        groups: list[_ChunkGroup],
        rng: np.random.Generator,
    ) -> Iterator[list[int]]:
        """Merge per-chunk tails so full epochs do not create many tiny batches."""

        yielded = 0
        for batch in self._iter_coalesced_group_batches(groups, rng):
            if self._batch_limit_reached(yielded):
                return
            self._record_batch_sources(batch)
            yielded += 1
            yield batch

    def _iter_coalesced_group_batches(
        self,
        groups: list[_ChunkGroup],
        rng: np.random.Generator,
    ) -> Iterator[list[int]]:
        pending = np.empty(0, dtype=np.int64)
        for group in groups:
            indices = self._index_order[group.start : group.stop].astype(
                np.int64, copy=True
            )
            if self.shuffle:
                rng.shuffle(indices)
            if pending.size:
                needed = self.batch_size - pending.size
                take = min(needed, indices.size)
                pending = np.concatenate((pending, indices[:take]))
                indices = indices[take:]
                if pending.size == self.batch_size:
                    yield pending.tolist()
                    pending = np.empty(0, dtype=np.int64)
            full_stop = indices.size - indices.size % self.batch_size
            for start in range(0, full_stop, self.batch_size):
                yield indices[start : start + self.batch_size].tolist()
            if full_stop < indices.size:
                pending = indices[full_stop:]
        if pending.size:
            yield pending.tolist()

    def _affinity_group_batches(
        self,
        groups: list[_ChunkGroup],
        rng: np.random.Generator,
    ) -> Iterator[list[int]]:
        """Interleave balanced chunk lanes to encourage per-worker cache reuse."""

        lane_count = min(self.worker_affinity_count, max(1, len(groups)))
        lanes: list[list[_ChunkGroup]] = [[] for _ in range(lane_count)]
        lane_samples = [0] * lane_count
        for group in groups:
            lane = min(range(lane_count), key=lane_samples.__getitem__)
            lanes[lane].append(group)
            lane_samples[lane] += group.stop - group.start

        iterators = [
            iter(self._iter_coalesced_group_batches(lane, rng)) for lane in lanes
        ]
        active = [True] * lane_count
        yielded = 0
        while any(active):
            for lane, iterator in enumerate(iterators):
                if not active[lane]:
                    continue
                try:
                    batch = next(iterator)
                except StopIteration:
                    active[lane] = False
                    continue
                if self._batch_limit_reached(yielded):
                    return
                self._record_batch_sources(batch)
                yielded += 1
                yield batch

    def _batch_limit_reached(self, yielded: int) -> bool:
        return (
            self.max_batches_per_epoch is not None
            and yielded >= self.max_batches_per_epoch
        )

    def _target_batches_by_source(self) -> dict[int, int]:
        sources = [
            source
            for source, count in self._natural_batches_by_source.items()
            if count > 0
        ]
        if not sources:
            return {}
        if self.sampling == "balanced_by_dataset":
            if self.max_batches_per_epoch is None:
                target = max(self._natural_batches_by_source.values())
                return {source: target for source in sources}
            weights = {source: 1.0 for source in sources}
            return self._allocate_batches(self.max_batches_per_epoch, weights)

        total = self.max_batches_per_epoch
        if total is None:
            total = sum(self._natural_batches_by_source.values())
        weights = {
            source: max(0.0, float(self.dataset.sources[source].weight))
            for source in sources
        }
        if sum(weights.values()) <= 0:
            raise ValueError("At least one lazy Zarr dataset weight must be positive.")
        return self._allocate_batches(total, weights)

    @staticmethod
    def _allocate_batches(total: int, weights: dict[int, float]) -> dict[int, int]:
        weight_sum = sum(weights.values())
        raw = {
            source: total * weight / weight_sum for source, weight in weights.items()
        }
        allocated = {source: int(math.floor(value)) for source, value in raw.items()}
        remainder = total - sum(allocated.values())
        ranked = sorted(
            weights,
            key=lambda source: (raw[source] - allocated[source], weights[source]),
            reverse=True,
        )
        for source in ranked[:remainder]:
            allocated[source] += 1
        return allocated

    def __len__(self) -> int:
        natural = sum(self._natural_batches_by_source.values())
        if self.sampling == "weighted_by_sample":
            if not self.drop_last:
                natural = math.ceil(len(self.dataset) / self.batch_size)
            if self.max_batches_per_epoch is None:
                return natural
            return min(natural, self.max_batches_per_epoch)
        return sum(self._target_batches_by_source().values())
