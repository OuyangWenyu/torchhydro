"""Lazy cloud/local Zarr dataset for TorchHydro streamflow training.

This module intentionally avoids ``BaseDataset._load_data`` because that path
eagerly loads all x/y/c arrays into NumPy.  The dataset keeps only coordinates,
sample indices, optional static attributes, and normalization statistics in
memory; dynamic windows are read from Zarr when a DataLoader batch is requested.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from torchhydro.configs.data_resolver import open_dataset
from torchhydro.datasets.data_utils import wrap_t_s_dict


class LookupTableView:
    """Tuple-compatible lookup table backed by compact NumPy arrays."""

    def __init__(self, basins, starts, lengths):
        self.basins = basins
        self.starts = starts
        self.lengths = lengths

    def __len__(self):
        return len(self.basins)

    def __getitem__(self, item: int):
        return (
            int(self.basins[item]),
            int(self.starts[item]),
            int(self.lengths[item]),
        )


@dataclass(frozen=True)
class LazyVariable:
    """Canonical-to-physical variable mapping for one Zarr variable."""

    canonical: str
    physical: str
    unit: str | None = None
    fill_value: float = 0.0
    required: bool = True
    scale: float = 1.0
    offset: float = 0.0
    configured: bool = True


@dataclass
class LazyDatasetSource:
    """One source dataset participating in lazy loading.

    The first implementation is optimized for a single source, but the shape is
    already explicit enough for multi-source indexing and dataset-level
    sampling.
    """

    dataset_id: str
    basins: list[str]
    # ``time_range`` is the source-access range.  ``target_time_range`` is the
    # experiment split range and may be narrower, allowing context history to
    # be read without turning it into a target sample.
    time_range: list[str] | str
    training_time_range: list[str] | str
    relevant_vars: list[LazyVariable]
    target_vars: list[LazyVariable]
    constant_vars: list[LazyVariable]
    target_time_range: list[str] | str = "all"
    timeseries_uri: str | None = None
    attributes_uri: str | None = None
    weight: float = 1.0


@dataclass
class LazySourceState:
    """Opened metadata and caches for one lazy source."""

    source: LazyDatasetSource
    ts_ds: xr.Dataset | None = None
    attr_ds: xr.Dataset | None = None
    basin_positions: np.ndarray | None = None
    time_offset: int = 0
    time_length: int = 0
    times: pd.DatetimeIndex | None = None
    target_time_offset: int = 0
    target_time_length: int = 0
    target_times: pd.DatetimeIndex | None = None
    time_step_ns: int | None = None
    static_values: np.ndarray | None = None
    normalized_static_values: np.ndarray | None = None
    relevant_names: list[str | None] = field(default_factory=list)
    target_names: list[str | None] = field(default_factory=list)
    constant_names: list[str | None] = field(default_factory=list)
    chunk_shape: tuple[int, int] = (1, 1)
    stats: dict[str, Any] | None = None


class CloudZarrStatCache:
    """Small mean/std cache for lazy Zarr normalization.

    This is deliberately simpler than ``DapengScaler``.  It gives lazy training
    stable normalization without loading all arrays into memory and can later be
    extended to emit TorchHydro-compatible stat dictionaries.
    """

    # Bumped when target QC became part of canonical statistics.
    version = 3

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_build(self, key_payload: dict[str, Any], builder) -> dict[str, Any]:
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if self.enabled and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("version") == self.version:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass
        stats = builder()
        stats["version"] = self.version
        stats["cache_key"] = key
        if self.enabled:
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)
        return stats


class CloudZarrLazyDataset(Dataset):
    """Read streamflow samples lazily from hydrodataset Zarr stores.

    Parameters are read from ``cfgs["data_cfgs"]`` using the same top-level keys
    as ``StreamflowDataset`` where possible.  Additional lazy-specific options
    live under ``data_cfgs["lazy_zarr"]``.
    """

    supports_prebatched_getitems = True

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        super().__init__()
        self.cfgs = cfgs
        self.data_cfgs = cfgs["data_cfgs"]
        self.training_cfgs = cfgs["training_cfgs"]
        self.evaluation_cfgs = cfgs["evaluation_cfgs"]
        self.lazy_cfg = self.data_cfgs.get("lazy_zarr") or {}
        default_target_nan_policy = (
            "drop_all_nan"
            if self.lazy_cfg.get("skip_all_nan_target_windows", True)
            else "keep"
        )
        self.target_nan_policy = self.lazy_cfg.get(
            "target_nan_policy",
            default_target_nan_policy,
        )
        if self.target_nan_policy not in ("keep", "drop_all_nan", "drop_any_nan"):
            raise ValueError(
                "lazy_zarr.target_nan_policy must be keep, drop_all_nan or drop_any_nan."
            )
        self.is_tra_val_te = is_tra_val_te
        self.train_mode = is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = int(self.training_cfgs["hindcast_length"])
        self.warmup_length = int(self.training_cfgs["warmup_length"])
        self.horizon = int(self.training_cfgs["forecast_length"])
        self._batch_level_dataloader_enabled = False
        self._open_pid = os.getpid()
        cache_mb = max(0.0, float(self.lazy_cfg.get("worker_cache_size_mb", 64)))
        self._block_cache_limit_bytes = int(cache_mb * 1024 * 1024)
        self._block_cache_bytes = 0
        self._block_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

        self._data_sources: dict[str, Any] = {}
        self._preopened_timeseries: dict[str, xr.Dataset] = {}
        self.sources = self._build_sources()
        self._align_source_schemas()
        self.source_states = [LazySourceState(source=source) for source in self.sources]
        for state in self.source_states:
            state.ts_ds = self._preopened_timeseries.pop(
                state.source.dataset_id,
                None,
            )
        for source_idx in range(len(self.source_states)):
            self._ensure_open(source_idx)
            self._prepare_time_index(source_idx)
            self._prepare_variable_layout(source_idx)
            self._prepare_static_attributes(source_idx)
        self._validate_source_compatibility()
        self._initialize_normalization_stats()
        self._finalize_missing_schema()
        self._prepare_normalized_static_values()
        self.source = self.sources[0]
        self.basins = [
            f"{source.dataset_id}:{basin}"
            for source in self.sources
            for basin in source.basins
        ]
        self.stats = self.source_states[0].stats or {"enabled": False}
        self._create_lookup_table()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data_sources"] = {}
        state["_open_pid"] = None
        state["_block_cache"] = OrderedDict()
        state["_block_cache_bytes"] = 0
        state["source_states"] = [
            LazySourceState(
                source=source_state.source,
                basin_positions=source_state.basin_positions,
                time_offset=source_state.time_offset,
                time_length=source_state.time_length,
                times=source_state.times,
                target_time_offset=source_state.target_time_offset,
                target_time_length=source_state.target_time_length,
                target_times=source_state.target_times,
                time_step_ns=source_state.time_step_ns,
                static_values=source_state.static_values,
                normalized_static_values=source_state.normalized_static_values,
                relevant_names=source_state.relevant_names,
                target_names=source_state.target_names,
                constant_names=source_state.constant_names,
                chunk_shape=source_state.chunk_shape,
                stats=source_state.stats,
            )
            for source_state in self.source_states
        ]
        return state

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        x, y = self._load_batch([int(item)])
        return x[0], y[0]

    def __getitems__(self, items):
        if self._batch_level_dataloader_enabled:
            return self._get_batch_level_items(items)
        return [self.__getitem__(int(item)) for item in items]

    def _get_batch_level_items(self, items):
        return self._load_batch([int(item) for item in items])

    def load_final_target_values(self, items: list[int]) -> np.ndarray:
        """Read raw physical targets at each window's final forecast step.

        Evaluation must distinguish missing observations from the zero values
        used in normalized model tensors. This method therefore bypasses
        normalization and reads only one target timestamp per requested window.
        """
        item_indices = np.asarray(items, dtype=np.int64)
        if item_indices.size == 0:
            return np.empty((0, len(self.target_schema)), dtype=np.float32)
        sources = self._lookup_sources[item_indices].astype(np.int16, copy=False)
        basins = self._lookup_basins[item_indices].astype(np.int32, copy=False)
        starts = self._lookup_starts[item_indices].astype(np.int32, copy=False)
        lengths = self._lookup_lengths[item_indices].astype(np.int32, copy=False)
        output = np.empty((len(item_indices), len(self.target_schema)), dtype=np.float32)

        global_basins = np.empty(len(item_indices), dtype=np.int64)
        final_times = np.empty(len(item_indices), dtype=np.int64)
        basin_chunks = np.empty(len(item_indices), dtype=np.int64)
        time_chunks = np.empty(len(item_indices), dtype=np.int64)
        for source_idx in np.unique(sources):
            rows = np.flatnonzero(sources == source_idx)
            state = self.source_states[int(source_idx)]
            assert state.basin_positions is not None
            basin_chunk_size, time_chunk_size = state.chunk_shape
            global_basins[rows] = state.basin_positions[basins[rows]]
            final_times[rows] = (
                state.time_offset
                + starts[rows].astype(np.int64)
                + lengths[rows].astype(np.int64)
                - self.warmup_length
                - 1
            )
            basin_chunks[rows] = global_basins[rows] // basin_chunk_size
            time_chunks[rows] = final_times[rows] // time_chunk_size

        order = np.lexsort((time_chunks, basin_chunks, sources))
        sorted_sources = sources[order]
        sorted_basins = basin_chunks[order]
        sorted_times = time_chunks[order]
        changes = (
            np.flatnonzero(
                (sorted_sources[1:] != sorted_sources[:-1])
                | (sorted_basins[1:] != sorted_basins[:-1])
                | (sorted_times[1:] != sorted_times[:-1])
            )
            + 1
        )
        boundaries = np.concatenate(([0], changes, [len(order)]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            rows = order[left:right]
            source_idx = int(sources[rows[0]])
            state = self.source_states[source_idx]
            assert state.ts_ds is not None
            basin_chunk_size, time_chunk_size = state.chunk_shape
            basin_start = int(basin_chunks[rows[0]] * basin_chunk_size)
            basin_stop = min(basin_start + basin_chunk_size, int(state.ts_ds.sizes["basin"]))
            time_start = int(time_chunks[rows[0]] * time_chunk_size)
            time_stop = min(time_start + time_chunk_size, int(state.ts_ds.sizes["time"]))
            block = self._read_block_array(
                source_idx,
                state.source.target_vars,
                basin_start,
                basin_stop,
                time_start,
                time_stop,
                "target_cols",
            )
            output[rows] = block[
                global_basins[rows] - basin_start,
                final_times[rows] - time_start,
            ]
        return output

    @property
    def x_feature_count(self) -> int:
        mask_count = len(self._mask_schema) if self._append_missing_masks else 0
        return len(self.relevant_schema) + mask_count + len(self.constant_schema)

    @property
    def y_feature_count(self) -> int:
        return len(self.target_schema)

    @property
    def ngrid(self) -> int:
        return sum(len(source.basins) for source in self.sources)

    @property
    def nt(self) -> int:
        return max((state.time_length for state in self.source_states), default=0)

    @property
    def source_chunk_shapes(self) -> dict[int, tuple[int, int]]:
        return {
            source_idx: state.chunk_shape
            for source_idx, state in enumerate(self.source_states)
        }

    @property
    def source_time_steps(self) -> dict[str, str | None]:
        return {
            state.source.dataset_id: (
                str(pd.to_timedelta(state.time_step_ns, unit="ns"))
                if state.time_step_ns is not None
                else None
            )
            for state in self.source_states
        }

    @property
    def lookup_nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self._lookup_sources,
                self._lookup_basins,
                self._lookup_starts,
                self._lookup_lengths,
            )
        )

    def _build_sources(self) -> list[LazyDatasetSource]:
        dataset_entries = self.lazy_cfg.get("datasets")
        if not dataset_entries:
            source_cfgs = self.data_cfgs.get("source_cfgs") or {}
            dataset_entries = [
                {
                    "dataset_id": source_cfgs.get("dataset_id"),
                    "basins": self.data_cfgs.get("object_ids"),
                    "time_range": self.lazy_cfg.get("source_time_range", "all"),
                    "relevant_cols": self.data_cfgs.get("relevant_cols") or [],
                    "target_cols": self.data_cfgs.get("target_cols") or [],
                    "constant_cols": self.data_cfgs.get("constant_cols") or [],
                    "timeseries_uri": self.lazy_cfg.get("timeseries_uri"),
                    "attributes_uri": self.lazy_cfg.get("attributes_uri"),
                }
            ]

        sources = []
        seen_dataset_ids = set()
        for entry in dataset_entries:
            dataset_id = entry.get("dataset_id")
            if not dataset_id:
                raise ValueError("lazy_zarr dataset entry must include dataset_id.")
            if dataset_id in seen_dataset_ids:
                raise ValueError(
                    f"lazy_zarr dataset_id must be unique; duplicate {dataset_id!r}."
                )
            seen_dataset_ids.add(dataset_id)
            timeseries_uri = entry.get("timeseries_uri") or self.lazy_cfg.get(
                "timeseries_uri"
            )
            attributes_uri = entry.get("attributes_uri") or self.lazy_cfg.get(
                "attributes_uri"
            )
            reader = None
            use_reader = bool(self.lazy_cfg.get("use_hydrodataset_reader", True))
            if use_reader:
                reader = self._open_data_source(dataset_id)
            basins = entry.get("basins")
            if basins is None or basins == "all":
                if reader is not None:
                    basins = [str(value) for value in reader.read_object_ids()]
                elif timeseries_uri:
                    ts = self._open_uri_dataset(timeseries_uri)
                    basins = [str(value) for value in ts["basin"].values]
                    self._preopened_timeseries[dataset_id] = ts
                else:
                    raise ValueError(
                        "basins must be provided when hydrodataset reader is disabled."
                    )
            basins = [str(value) for value in basins]
            max_basins = entry.get("max_basins")
            if max_basins is not None:
                basins = basins[: max(0, int(max_basins))]
            if not basins:
                raise ValueError(f"{dataset_id} does not contain any selected basins.")
            time_range = self._source_time_range(entry)
            target_time_range = self._target_time_range(entry, self.is_tra_val_te)
            training_time_range = self._target_time_range(entry, "train")
            relevant_cols = entry.get(
                "relevant_cols", self.data_cfgs.get("relevant_cols") or []
            )
            target_cols = entry.get(
                "target_cols", self.data_cfgs.get("target_cols") or []
            )
            constant_cols = entry.get(
                "constant_cols", self.data_cfgs.get("constant_cols") or []
            )
            sources.append(
                LazyDatasetSource(
                    dataset_id=dataset_id,
                    basins=basins,
                    time_range=time_range,
                    target_time_range=target_time_range,
                    training_time_range=training_time_range,
                    relevant_vars=self._resolve_dynamic_variables(
                        reader, relevant_cols
                    ),
                    target_vars=self._resolve_dynamic_variables(reader, target_cols),
                    constant_vars=self._resolve_static_variables(reader, constant_cols),
                    timeseries_uri=timeseries_uri,
                    attributes_uri=attributes_uri,
                    weight=float(entry.get("weight", 1.0)),
                )
            )
        return sources

    def _source_time_range(self, entry: dict[str, Any]) -> list[str] | str:
        """Resolve the source-access range, independent of target splits."""
        configured_time_range = entry.get("source_time_range")
        if configured_time_range is None:
            configured_time_range = entry.get("time_range")
        if configured_time_range in (None, "all"):
            return "all"
        if (
            not isinstance(configured_time_range, (list, tuple))
            or len(configured_time_range) != 2
        ):
            raise ValueError(
                f"{entry.get('dataset_id', 'dataset')} source_time_range must be "
                "'all' or [start, end]."
            )
        return [str(value) for value in configured_time_range]

    def _target_time_range(
        self, entry: dict[str, Any], split: str
    ) -> list[str] | str:
        """Resolve one source's target range for a train/valid/test split."""
        entry_periods = entry.get("periods") or {}
        configured_time_range = entry_periods.get(split)
        if configured_time_range is None:
            configured_time_range = entry.get(f"{split}_period")
        if configured_time_range is None:
            configured_time_range = (self.lazy_cfg.get("split_periods") or {}).get(
                split
            )
        if configured_time_range is None:
            configured_time_range = self.data_cfgs.get(f"t_range_{split}")
        if configured_time_range is None:
            configured_time_range = entry.get("target_time_range")
        if configured_time_range in (None, "all"):
            return "all"
        if (
            not isinstance(configured_time_range, (list, tuple))
            or len(configured_time_range) != 2
        ):
            raise ValueError(
                f"{entry.get('dataset_id', 'dataset')} {split} period must be "
                "'all' or [start, end]."
            )
        return [str(value) for value in configured_time_range]

    def _open_data_source(self, dataset_id: str):
        if dataset_id in self._data_sources:
            return self._data_sources[dataset_id]
        source_cfgs = self.data_cfgs.get("source_cfgs") or {}
        other_settings = dict(source_cfgs.get("other_settings") or {})
        other_settings.update(self.lazy_cfg.get("reader_kwargs") or {})
        self._data_sources[dataset_id] = open_dataset(dataset_id, **other_settings)
        return self._data_sources[dataset_id]

    def _align_source_schemas(self) -> None:
        """Align each source to a global canonical feature schema."""

        def schema_from_config(config_key: str, source_attr: str) -> list[str]:
            configured = self.lazy_cfg.get(config_key)
            if configured:
                return [str(value) for value in configured]
            seen = []
            for source in self.sources:
                for var in getattr(source, source_attr):
                    if var.canonical not in seen:
                        seen.append(var.canonical)
            return seen

        self.relevant_schema = schema_from_config("input_schema", "relevant_vars")
        self.target_schema = schema_from_config("target_schema", "target_vars")
        self.constant_schema = schema_from_config("constant_schema", "constant_vars")

        def align(
            variables: list[LazyVariable],
            schema: list[str],
            required_missing: bool,
        ) -> list[LazyVariable]:
            by_name = {var.canonical: var for var in variables}
            aligned = []
            for canonical in schema:
                if canonical in by_name:
                    aligned.append(by_name[canonical])
                else:
                    aligned.append(
                        LazyVariable(
                            canonical=canonical,
                            physical=canonical,
                            fill_value=0.0,
                            required=required_missing,
                            configured=False,
                        )
                    )
            return aligned

        for source in self.sources:
            source.relevant_vars = align(
                source.relevant_vars,
                self.relevant_schema,
                required_missing=False,
            )
            source.target_vars = align(
                source.target_vars,
                self.target_schema,
                required_missing=True,
            )
            source.constant_vars = align(
                source.constant_vars,
                self.constant_schema,
                required_missing=False,
            )

        self._mask_schema: list[str] = []

    def _resolve_dynamic_variables(self, reader, cols: Any) -> list[LazyVariable]:
        if isinstance(cols, dict):
            items = cols.items()
        else:
            items = ((col, col) for col in (cols or []))
        return [
            self._resolve_one_dynamic_variable(reader, str(canonical), spec)
            for canonical, spec in items
        ]

    def _resolve_one_dynamic_variable(
        self, reader, canonical: str, spec: Any
    ) -> LazyVariable:
        physical = canonical
        unit = None
        required = True
        fill_value = 0.0
        scale = 1.0
        offset = 0.0
        if isinstance(spec, dict):
            physical = str(spec.get("physical") or spec.get("name") or canonical)
            unit = spec.get("unit")
            required = bool(spec.get("required", True))
            fill_value = float(spec.get("fill_value", 0.0))
            scale = float(spec.get("scale", 1.0))
            offset = float(spec.get("offset", 0.0))
        elif spec is not None:
            physical = str(spec)

        mapping = getattr(reader, "_dynamic_variable_mapping", {}) or {}
        if canonical in mapping and not isinstance(spec, dict):
            mapping_info = mapping[canonical]
            source = mapping_info.get("default_source")
            physical = mapping_info["sources"][source]["specific_name"]
            unit = mapping_info["sources"][source].get("unit")

        return LazyVariable(
            canonical=canonical,
            physical=physical,
            unit=unit,
            fill_value=fill_value,
            required=required,
            scale=scale,
            offset=offset,
        )

    def _resolve_static_variables(self, reader, cols: Any) -> list[LazyVariable]:
        if isinstance(cols, dict):
            items = cols.items()
        else:
            items = ((col, col) for col in (cols or []))
        definitions = getattr(reader, "_static_variable_definitions", {}) or {}
        variables = []
        for canonical, spec in items:
            canonical = str(canonical)
            physical = canonical
            unit = None
            required = True
            fill_value = 0.0
            scale = 1.0
            offset = 0.0
            if isinstance(spec, dict):
                physical = str(spec.get("physical") or spec.get("name") or canonical)
                unit = spec.get("unit")
                required = bool(spec.get("required", True))
                fill_value = float(spec.get("fill_value", 0.0))
                scale = float(spec.get("scale", 1.0))
                offset = float(spec.get("offset", 0.0))
            elif spec is not None:
                physical = str(spec)
            if canonical in definitions and not isinstance(spec, dict):
                physical = definitions[canonical].get("specific_name", canonical)
                unit = definitions[canonical].get("unit")
            variables.append(
                LazyVariable(
                    canonical=canonical,
                    physical=physical,
                    unit=unit,
                    fill_value=fill_value,
                    required=required,
                    scale=scale,
                    offset=offset,
                )
            )
        return variables

    def _reset_open_handles_after_fork(self) -> None:
        current_pid = os.getpid()
        if getattr(self, "_open_pid", None) == current_pid:
            return
        self._close_open_datasets()
        self._data_sources = {}
        self._clear_block_cache()
        self._open_pid = current_pid

    def _close_open_datasets(self) -> None:
        for source_state in self.source_states:
            for attr in ("ts_ds", "attr_ds"):
                ds = getattr(source_state, attr)
                if ds is None:
                    continue
                try:
                    ds.close()
                except Exception:
                    pass
                setattr(source_state, attr, None)

    def _clear_block_cache(self) -> None:
        self._block_cache.clear()
        self._block_cache_bytes = 0

    def close(self) -> None:
        """Close Zarr/NetCDF handles owned by the current process."""

        self._close_open_datasets()
        self._clear_block_cache()
        self._data_sources = {}

    def _ensure_open(self, source_idx: int | None = None) -> None:
        self._reset_open_handles_after_fork()
        if source_idx is None:
            for idx in range(len(self.source_states)):
                self._ensure_open(idx)
            return
        state = self.source_states[source_idx]
        source = state.source
        if state.ts_ds is None:
            state.ts_ds = self._open_dataset_store(
                source,
                source.timeseries_uri,
                fallback_loader="_load_ts_dataset",
            )
        if (
            any(var.configured for var in source.constant_vars)
            and state.attr_ds is None
        ):
            state.attr_ds = self._open_dataset_store(
                source,
                source.attributes_uri,
                fallback_loader="_load_attr_dataset",
            )

    def _open_dataset_store(
        self,
        source: LazyDatasetSource,
        uri: str | None,
        fallback_loader: str,
    ) -> xr.Dataset:
        if uri:
            return self._open_uri_dataset(uri)

        reader = self._open_data_source(source.dataset_id)
        if not hasattr(reader, fallback_loader):
            raise ValueError(
                f"{source.dataset_id} reader does not expose {fallback_loader}; "
                "pass lazy_zarr.timeseries_uri/attributes_uri explicitly."
            )
        opened = getattr(reader, fallback_loader)()
        store_uri = opened.encoding.get("source")
        if not store_uri or not str(store_uri).rstrip("/").endswith(".zarr"):
            return opened

        storage_options = self.lazy_cfg.get("storage_options")
        if storage_options is None and hasattr(reader, "_zarr_path_and_opts"):
            try:
                _, storage_options = reader._zarr_path_and_opts("unused.zarr")
            except Exception:
                storage_options = None
        try:
            reopened = self._open_uri_dataset(
                str(store_uri),
                storage_options=storage_options,
            )
        except Exception:
            return opened
        opened.close()
        return reopened

    def _open_uri_dataset(
        self,
        uri: str,
        storage_options: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        if storage_options is None:
            storage_options = self.lazy_cfg.get("storage_options")
        if uri.startswith("s3://") or uri.endswith(".zarr"):
            return xr.open_zarr(
                uri,
                storage_options=storage_options,
                consolidated=False,
                mask_and_scale=False,
                chunks=None,
            )
        return xr.open_dataset(uri, mask_and_scale=False)

    def _prepare_time_index(self, source_idx: int) -> None:
        self._ensure_open(source_idx)
        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        if "basin" not in state.ts_ds.coords or "time" not in state.ts_ds.coords:
            raise ValueError(
                "Zarr timeseries store must have basin and time coordinates."
            )

        zarr_basins = {
            str(value): pos for pos, value in enumerate(state.ts_ds["basin"].values)
        }
        missing_basins = [basin for basin in source.basins if basin not in zarr_basins]
        if missing_basins:
            raise ValueError(
                f"{source.dataset_id} has {len(missing_basins)} requested basins "
                "missing from the Zarr store."
            )
        state.basin_positions = np.asarray(
            [zarr_basins[basin] for basin in source.basins], dtype=np.int32
        )

        full_times = pd.DatetimeIndex(pd.to_datetime(state.ts_ds["time"].values))
        if source.time_range == "all":
            positions = np.arange(len(full_times), dtype=np.int64)
        else:
            start = pd.to_datetime(source.time_range[0])
            end = pd.to_datetime(source.time_range[1])
            if full_times[0] > start or full_times[-1] < end:
                raise ValueError(
                    f"{source.dataset_id} does not cover the requested "
                    f"{self.is_tra_val_te} period {source.time_range}. Its Zarr "
                    f"time coordinate covers [{full_times[0].isoformat()}, "
                    f"{full_times[-1].isoformat()}]. Adjust the split, remove this "
                    "dataset explicitly, or configure a dataset-specific period."
                )
            positions = np.flatnonzero((full_times >= start) & (full_times <= end))
        if positions.size == 0:
            raise ValueError(
                f"No time points found for {source.dataset_id} in {source.time_range}."
            )
        state.time_offset = int(positions[0])
        state.time_length = int(positions.size)
        state.times = full_times[positions]
        target_range = source.target_time_range
        if target_range == "all":
            target_positions = positions
        else:
            target_start = pd.to_datetime(target_range[0])
            target_end = pd.to_datetime(target_range[1])
            if full_times[0] > target_start or full_times[-1] < target_end:
                raise ValueError(
                    f"{source.dataset_id} does not cover the requested "
                    f"{self.is_tra_val_te} period {target_range}. Its Zarr time "
                    f"coordinate covers [{full_times[0].isoformat()}, "
                    f"{full_times[-1].isoformat()}]."
                )
            target_positions = np.flatnonzero(
                (full_times >= target_start) & (full_times <= target_end)
            )
            if target_positions.size == 0:
                raise ValueError(
                    f"No target time points found for {source.dataset_id} in "
                    f"{target_range}."
                )
            if (
                target_positions[0] < positions[0]
                or target_positions[-1] > positions[-1]
            ):
                raise ValueError(
                    f"{source.dataset_id} source range {source.time_range} does not "
                    f"cover target range {target_range}."
                )
        state.target_time_offset = int(target_positions[0] - positions[0])
        state.target_time_length = int(target_positions.size)
        state.target_times = full_times[target_positions]
        if positions.size > 1:
            time_deltas = np.diff(state.times.asi8)
            if not np.all(time_deltas == time_deltas[0]):
                raise ValueError(
                    f"{source.dataset_id} has an irregular time coordinate in "
                    f"{source.time_range}; regular windows are required."
                )
            state.time_step_ns = int(time_deltas[0])
        if source_idx == 0:
            self.times = state.times

    def _validate_source_compatibility(self) -> None:
        time_steps = {
            state.time_step_ns
            for state in self.source_states
            if state.time_step_ns is not None
        }
        if len(time_steps) > 1 and not self.lazy_cfg.get(
            "allow_mixed_time_steps", False
        ):
            details = ", ".join(
                f"{dataset_id}={time_step}"
                for dataset_id, time_step in self.source_time_steps.items()
            )
            raise ValueError(
                "Joint lazy training requires one common time step. "
                f"Found {details}. Resample the stores first or set "
                "allow_mixed_time_steps=true only if the model handles this explicitly."
            )

        policy = self.lazy_cfg.get("unit_mismatch_policy", "warn")
        if policy not in ("ignore", "warn", "error"):
            raise ValueError(
                "lazy_zarr.unit_mismatch_policy must be ignore, warn or error."
            )
        if policy == "ignore":
            return
        mismatches = []
        for variables_attr, names_attr, schema in (
            ("relevant_vars", "relevant_names", self.relevant_schema),
            ("target_vars", "target_names", self.target_schema),
            ("constant_vars", "constant_names", self.constant_schema),
        ):
            for col, canonical in enumerate(schema):
                units = {
                    state.source.dataset_id: getattr(state.source, variables_attr)[
                        col
                    ].unit
                    for state in self.source_states
                    if getattr(state, names_attr)[col] is not None
                    and getattr(state.source, variables_attr)[col].unit
                }
                if len(set(units.values())) > 1:
                    mismatches.append(f"{canonical}: {units}")
        if not mismatches:
            return
        message = (
            "Canonical variables use different declared units across datasets. "
            "Configure per-dataset scale/offset/unit before joint training: "
            + "; ".join(mismatches)
        )
        if policy == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    def _prepare_variable_layout(self, source_idx: int) -> None:
        """Resolve physical names once and infer the store's basin/time chunks."""

        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        state.relevant_names = self._resolve_existing_names(
            state.ts_ds,
            source.relevant_vars,
            "input",
            source.dataset_id,
        )
        state.target_names = self._resolve_existing_names(
            state.ts_ds,
            source.target_vars,
            "target",
            source.dataset_id,
        )
        if source.constant_vars and state.attr_ds is not None:
            state.constant_names = self._resolve_existing_names(
                state.attr_ds,
                source.constant_vars,
                "static",
                source.dataset_id,
            )
        else:
            state.constant_names = [None] * len(source.constant_vars)

        chunk_shapes = []
        for physical in state.relevant_names + state.target_names:
            if physical is None:
                continue
            chunk_shape = self._data_array_chunk_shape(state.ts_ds[physical])
            if chunk_shape is not None:
                chunk_shapes.append(chunk_shape)
        if chunk_shapes:
            state.chunk_shape = (
                min(shape[0] for shape in chunk_shapes),
                min(shape[1] for shape in chunk_shapes),
            )
        else:
            state.chunk_shape = (
                max(1, int(self.lazy_cfg.get("fallback_basin_chunk_size", 1))),
                max(
                    self.rho + self.horizon,
                    int(self.training_cfgs.get("batch_size", 1)),
                    int(self.lazy_cfg.get("fallback_time_chunk_size", 1)),
                ),
            )

    @staticmethod
    def _data_array_chunk_shape(da: xr.DataArray) -> tuple[int, int] | None:
        preferred = da.encoding.get("preferred_chunks") or {}
        if "basin" in preferred and "time" in preferred:
            return int(preferred["basin"]), int(preferred["time"])
        chunks = da.encoding.get("chunks")
        if chunks and len(chunks) == len(da.dims):
            by_dim = {dim: int(chunks[pos]) for pos, dim in enumerate(da.dims)}
            if "basin" in by_dim and "time" in by_dim:
                return by_dim["basin"], by_dim["time"]
        return None

    def _resolve_existing_names(
        self,
        ds: xr.Dataset,
        variables: list[LazyVariable],
        kind: str,
        dataset_id: str,
    ) -> list[str | None]:
        names = []
        for var in variables:
            physical = self._existing_var_name(ds, var) if var.configured else None
            if physical is None and var.required:
                raise ValueError(
                    f"Dataset {dataset_id!r}: required {kind} variable "
                    f"{var.canonical} ({var.physical}) is missing from its Zarr "
                    f"store. Available variables: {sorted(ds.data_vars)}"
                )
            names.append(physical)
        return names

    def _finalize_missing_schema(self) -> None:
        self._relevant_mask_schema = [
            canonical
            for col, canonical in enumerate(self.relevant_schema)
            if any(state.relevant_names[col] is None for state in self.source_states)
        ]
        self._constant_mask_schema = [
            canonical
            for col, canonical in enumerate(self.constant_schema)
            if any(state.constant_names[col] is None for state in self.source_states)
        ]
        self._mask_schema = self._relevant_mask_schema + self._constant_mask_schema

    def _prepare_static_attributes(self, source_idx: int) -> None:
        state = self.source_states[source_idx]
        source = state.source
        state.static_values = None
        if not source.constant_vars:
            return
        values = np.empty(
            (len(source.basins), len(source.constant_vars)),
            dtype=np.float32,
        )
        if state.attr_ds is None:
            for col, var in enumerate(source.constant_vars):
                values[:, col] = var.fill_value
            state.static_values = values
            return

        self._ensure_open(source_idx)
        attr_basins = {
            str(value): pos for pos, value in enumerate(state.attr_ds["basin"].values)
        }
        missing_attr_basins = [
            basin for basin in source.basins if basin not in attr_basins
        ]
        if missing_attr_basins:
            raise ValueError(
                f"{source.dataset_id} has {len(missing_attr_basins)} selected basins "
                "missing from its attributes store."
            )
        basin_positions = [attr_basins[basin] for basin in source.basins]
        for col, (var, physical) in enumerate(
            zip(source.constant_vars, state.constant_names)
        ):
            if physical is None:
                values[:, col] = var.fill_value
                continue
            arr = state.attr_ds[physical].isel(basin=basin_positions).to_numpy()
            values[:, col] = np.asarray(arr, dtype=np.float32) * var.scale + var.offset
        state.static_values = values

    def _load_or_build_stats(self, source_idx: int) -> dict[str, Any]:
        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        normalization = self.lazy_cfg.get("normalization", "standard")
        if normalization in (None, "none", False):
            return {"enabled": False}

        cache_dir = self.lazy_cfg.get("stat_cache_dir")
        if cache_dir is None:
            cache_dir = Path(self.data_cfgs["case_dir"]) / "lazy_zarr_stat_cache"
        stat_cache = CloudZarrStatCache(
            cache_dir,
            enabled=bool(self.lazy_cfg.get("cache_stats", True)),
        )
        key_payload = {
            "cache_version": CloudZarrStatCache.version,
            "dataset_id": source.dataset_id,
            "timeseries_uri": source.timeseries_uri,
            "attributes_uri": source.attributes_uri,
            "reader_kwargs": self.lazy_cfg.get("reader_kwargs"),
            "basins": source.basins,
            "time_range": source.time_range,
            "target_time_range": source.target_time_range,
            "relevant": [var.__dict__ for var in source.relevant_vars],
            "target": [var.__dict__ for var in source.target_vars],
            "target_missing_values": [
                (
                    self._metadata_missing_values(state.ts_ds[physical])
                    if physical is not None
                    else ()
                )
                for physical in state.target_names
            ],
            "constant": [var.__dict__ for var in source.constant_vars],
            "normalization": normalization,
        }
        return stat_cache.load_or_build(
            key_payload,
            lambda: self._build_standard_stats(source_idx),
        )

    def _initialize_normalization_stats(self) -> None:
        """Build train statistics once, then reuse them for validation and test.

        Dynamic normalization is intentionally split-aware: deriving statistics
        from validation or test windows would leak future observations into the
        model input transformation.  The persisted payload also lets an
        evaluation-only process fail clearly when its matching train statistics
        have not been created.
        """
        normalization = self.lazy_cfg.get("normalization", "standard")
        if normalization in (None, "none", False):
            for state in self.source_states:
                state.stats = {"enabled": False}
            return

        if self.train_mode:
            for source_idx in range(len(self.source_states)):
                self.source_states[source_idx].stats = self._load_or_build_stats(
                    source_idx
                )
            self._apply_normalization_scope()
            self._save_training_stats()
            return

        training_stats = self._load_training_stats()
        stats_by_dataset = training_stats["sources"]
        for state in self.source_states:
            dataset_id = state.source.dataset_id
            if dataset_id not in stats_by_dataset:
                raise ValueError(
                    f"Training normalization statistics do not contain {dataset_id!r}. "
                    "Run the matching training split first or use a new case_dir."
                )
            state.stats = stats_by_dataset[dataset_id]

    def _training_stats_path(self) -> Path:
        configured = self.lazy_cfg.get("training_stats_path")
        if configured:
            return Path(configured)
        return Path(self.data_cfgs["case_dir"]) / "lazy_zarr_training_stats.json"

    def _training_stats_signature(self) -> str:
        payload = {
            "version": 2,
            "normalization": self.lazy_cfg.get("normalization", "standard"),
            "normalization_scope": self.lazy_cfg.get("normalization_scope", "global"),
            "input_schema": self.relevant_schema,
            "target_schema": self.target_schema,
            "constant_schema": self.constant_schema,
            "sources": [
                {
                    "dataset_id": source.dataset_id,
                    "timeseries_uri": source.timeseries_uri,
                    "attributes_uri": source.attributes_uri,
                    "basins": source.basins,
                    "source_time_range": source.time_range,
                    "training_time_range": source.training_time_range,
                    "relevant": [var.__dict__ for var in source.relevant_vars],
                    "target": [var.__dict__ for var in source.target_vars],
                    "target_missing_values": [
                        (
                            self._metadata_missing_values(
                                self.source_states[source_idx].ts_ds[physical]
                            )
                            if physical is not None
                            else ()
                        )
                        for physical in self.source_states[source_idx].target_names
                    ],
                    "constant": [var.__dict__ for var in source.constant_vars],
                }
                for source_idx, source in enumerate(self.sources)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _save_training_stats(self) -> None:
        path = self._training_stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "signature": self._training_stats_signature(),
            "sources": {
                state.source.dataset_id: state.stats for state in self.source_states
            },
        }
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _load_training_stats(self) -> dict[str, Any]:
        path = self._training_stats_path()
        if not path.exists():
            raise ValueError(
                f"Training normalization statistics are required for the "
                f"{self.is_tra_val_te} split but were not found at {path}. "
                "Construct or run the matching train split first."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read training normalization statistics from {path}."
            ) from exc
        if payload.get("version") != 2 or not isinstance(payload.get("sources"), dict):
            raise ValueError(
                f"Training normalization statistics at {path} have an unsupported format."
            )
        if payload.get("signature") != self._training_stats_signature():
            raise ValueError(
                f"Training normalization statistics at {path} do not match this "
                "dataset configuration or train_period. Use a separate case_dir "
                "or rebuild the training split."
            )
        return payload

    def _build_standard_stats(self, source_idx: int) -> dict[str, Any]:
        self._ensure_open(source_idx)
        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        assert state.basin_positions is not None
        stats = {
            "enabled": True,
            "normalization": "standard",
            "dataset_id": source.dataset_id,
            "relevant_cols": {},
            "target_cols": {},
            "constant_cols": {},
        }
        for kind, variables in (
            ("relevant_cols", source.relevant_vars),
            ("target_cols", source.target_vars),
        ):
            physical_names = (
                state.relevant_names if kind == "relevant_cols" else state.target_names
            )
            for var, physical in zip(variables, physical_names):
                if physical is None:
                    stats[kind][var.canonical] = {
                        "mean": 0.0,
                        "std": 1.0,
                        "count": 0,
                    }
                    continue
                da = state.ts_ds[physical].isel(
                    time=slice(
                        state.time_offset, state.time_offset + state.time_length
                    ),
                )
                stats[kind][var.canonical] = self._mean_std_dynamic(
                    da,
                    state.basin_positions,
                    state.target_time_length,
                    state.chunk_shape[0],
                    start_offset=state.time_offset + state.target_time_offset,
                    scale=var.scale,
                    offset=var.offset,
                    variable=var,
                    physical=physical,
                    target_qc=kind == "target_cols",
                )
        if state.static_values is not None:
            for col, (var, physical) in enumerate(
                zip(source.constant_vars, state.constant_names)
            ):
                stats["constant_cols"][var.canonical] = (
                    self._mean_std(state.static_values[:, col])
                    if physical is not None
                    else {"mean": 0.0, "std": 1.0, "count": 0}
                )
        return stats

    def _mean_std_dynamic(
        self,
        da: xr.DataArray,
        basin_positions: np.ndarray,
        time_length: int,
        storage_basin_chunk_size: int,
        *,
        start_offset: int = 0,
        scale: float = 1.0,
        offset: float = 0.0,
        variable: LazyVariable | None = None,
        physical: str | None = None,
        target_qc: bool = False,
    ) -> dict[str, float]:
        time_chunk_size = int(self.lazy_cfg.get("stat_time_chunk_size", 3650))
        time_chunk_size = max(1, time_chunk_size)
        count = 0
        total = 0.0
        total_sq = 0.0
        basin_groups: dict[int, list[int]] = {}
        for basin_position in basin_positions:
            chunk_id = int(basin_position) // storage_basin_chunk_size
            basin_groups.setdefault(chunk_id, []).append(int(basin_position))
        for chunk_id, selected_basins in basin_groups.items():
            basin_start = chunk_id * storage_basin_chunk_size
            basin_stop = min(basin_start + storage_basin_chunk_size, da.sizes["basin"])
            selected_offsets = np.asarray(selected_basins, dtype=np.int64) - basin_start
            for time_start in range(0, time_length, time_chunk_size):
                time_stop = min(time_start + time_chunk_size, time_length)
                block = da.isel(
                    basin=slice(basin_start, basin_stop),
                    time=slice(
                        start_offset + time_start,
                        start_offset + time_stop,
                    ),
                ).to_numpy()
                arr = np.asarray(block, dtype=np.float64)[selected_offsets]
                # Fit on canonical values, matching the transform used at read time.
                if target_qc:
                    if variable is None or physical is None:
                        raise ValueError(
                            "Target statistics require the configured target variable."
                        )
                    arr = self._canonicalize_target_values(
                        arr,
                        variable,
                        physical,
                        da,
                    )
                elif scale != 1.0 or offset != 0.0:
                    arr = arr * scale + offset
                valid = np.isfinite(arr)
                if not valid.any():
                    continue
                values = arr[valid]
                count += int(values.size)
                total += float(values.sum())
                total_sq += float(np.square(values).sum())
        if count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        if not math.isfinite(mean):
            mean = 0.0
        return {"mean": float(mean), "std": float(std), "count": count}

    @staticmethod
    def _mean_std(value: Any) -> dict[str, float]:
        arr = np.asarray(value, dtype=np.float64)
        valid = np.isfinite(arr)
        count = int(valid.sum())
        if count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        values = arr[valid]
        mean = float(values.mean())
        std = float(values.std())
        if not math.isfinite(mean):
            mean = 0.0
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        return {"mean": mean, "std": std, "count": count}

    def _apply_normalization_scope(self) -> None:
        normalization = self.lazy_cfg.get("normalization", "standard")
        if normalization in (None, "none", False) or len(self.source_states) <= 1:
            return
        scope = self.lazy_cfg.get("normalization_scope", "global")
        if scope in ("per_dataset", "per_source"):
            return
        if scope != "global":
            raise ValueError(
                "lazy_zarr.normalization_scope must be 'global' or 'per_dataset'."
            )

        global_stats: dict[str, Any] = {
            "enabled": True,
            "normalization": "standard",
            "dataset_id": "__global__",
            "normalization_scope": "global",
            "relevant_cols": {},
            "target_cols": {},
            "constant_cols": {},
        }
        for kind, schema in (
            ("relevant_cols", self.relevant_schema),
            ("target_cols", self.target_schema),
            ("constant_cols", self.constant_schema),
        ):
            for canonical in schema:
                parts = [
                    state.stats.get(kind, {}).get(canonical)
                    for state in self.source_states
                    if state.stats and state.stats.get("enabled")
                ]
                global_stats[kind][canonical] = self._combine_stats(parts)
        for state in self.source_states:
            state.stats = global_stats

    @staticmethod
    def _combine_stats(parts: list[dict[str, Any] | None]) -> dict[str, float | int]:
        usable = [part for part in parts if part and int(part.get("count", 0)) > 0]
        count = sum(int(part["count"]) for part in usable)
        if count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        mean = sum(float(part["mean"]) * int(part["count"]) for part in usable) / count
        second_moment = (
            sum(
                (float(part["std"]) ** 2 + float(part["mean"]) ** 2)
                * int(part["count"])
                for part in usable
            )
            / count
        )
        std = math.sqrt(max(second_moment - mean * mean, 0.0))
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        return {"mean": float(mean), "std": float(std), "count": count}

    def _prepare_normalized_static_values(self) -> None:
        for source_idx, state in enumerate(self.source_states):
            if state.static_values is None:
                state.normalized_static_values = None
                continue
            state.normalized_static_values = self._normalize_static(
                state.static_values,
                source_idx,
            )

    def _create_lookup_table(self) -> None:
        warmup = self.warmup_length
        seq_len = self.rho + self.horizon
        source_chunks = []
        basin_chunks = []
        start_chunks = []
        length_chunks = []
        filter_targets = self.target_nan_policy != "keep"
        for source_idx, state in enumerate(self.source_states):
            target_start = state.target_time_offset
            target_end = target_start + state.target_time_length
            target_length = seq_len - warmup
            first_start = max(warmup, target_start)
            max_start = target_end - target_length + 1
            if max_start <= first_start:
                continue
            starts_template = np.arange(first_start, max_start, dtype=np.int32)
            packed_target_validity = (
                self._load_or_build_target_validity(source_idx)
                if filter_targets and self.train_mode
                else None
            )
            for basin_pos in range(len(state.source.basins)):
                starts = starts_template
                if packed_target_validity is not None:
                    time_has_value = np.unpackbits(
                        packed_target_validity[basin_pos],
                        count=state.time_length,
                    ).astype(bool, copy=False)
                    starts = self._filter_starts_from_target_validity(
                        time_has_value,
                        starts,
                        seq_len - self.warmup_length,
                        self.target_nan_policy,
                    )
                if starts.size:
                    source_chunks.append(
                        np.full(starts.shape, source_idx, dtype=np.int16)
                    )
                    basin_chunks.append(
                        np.full(starts.shape, basin_pos, dtype=np.int32)
                    )
                    start_chunks.append(starts.astype(np.int32, copy=False))
                    length_chunks.append(np.full(starts.shape, seq_len, dtype=np.int32))
        if basin_chunks:
            sources = np.concatenate(source_chunks)
            basins = np.concatenate(basin_chunks)
            starts = np.concatenate(start_chunks)
            lengths = np.concatenate(length_chunks)
        else:
            sources = np.empty(0, dtype=np.int16)
            basins = np.empty(0, dtype=np.int32)
            starts = np.empty(0, dtype=np.int32)
            lengths = np.empty(0, dtype=np.int32)
        self._set_lookup_arrays(sources, basins, starts, lengths)

    def _load_or_build_target_validity(self, source_idx: int) -> np.ndarray:
        state = self.source_states[source_idx]
        source = state.source
        cache_dir = self.lazy_cfg.get("index_cache_dir")
        if cache_dir is None:
            cache_dir = Path(self.data_cfgs["case_dir"]) / "lazy_zarr_index_cache"
        cache_dir = Path(cache_dir)
        cache_enabled = bool(self.lazy_cfg.get("cache_target_index", True))
        assert state.ts_ds is not None
        payload = {
            "version": 2,
            "dataset_id": source.dataset_id,
            "timeseries_uri": source.timeseries_uri,
            "reader_kwargs": self.lazy_cfg.get("reader_kwargs"),
            "basins": source.basins,
            "time_range": source.time_range,
            "target": [var.__dict__ for var in source.target_vars],
            "target_missing_values": [
                (
                    self._metadata_missing_values(state.ts_ds[physical])
                    if physical is not None
                    else ()
                )
                for physical in state.target_names
            ],
            "time_length": state.time_length,
            "target_nan_policy": self.target_nan_policy,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        path = cache_dir / f"{key}.npz"
        if cache_enabled and path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    packed = np.asarray(cached["packed"], dtype=np.uint8)
                    cached_time_length = int(cached["time_length"])
                expected_shape = (
                    len(source.basins),
                    math.ceil(state.time_length / 8),
                )
                if (
                    cached_time_length == state.time_length
                    and packed.shape == expected_shape
                ):
                    return packed
            except (OSError, ValueError, EOFError, KeyError):
                pass

        packed = self._scan_target_validity(source_idx)
        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(temporary, packed=packed, time_length=state.time_length)
            os.replace(temporary, path)
        return packed

    def _scan_target_validity(self, source_idx: int) -> np.ndarray:
        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        assert state.basin_positions is not None
        packed = np.zeros(
            (len(source.basins), math.ceil(state.time_length / 8)),
            dtype=np.uint8,
        )
        basin_chunk_size = state.chunk_shape[0]
        basin_groups: dict[int, list[tuple[int, int]]] = {}
        for local_basin, global_basin in enumerate(state.basin_positions):
            chunk_id = int(global_basin) // basin_chunk_size
            basin_groups.setdefault(chunk_id, []).append(
                (local_basin, int(global_basin))
            )

        for chunk_id, selected in basin_groups.items():
            basin_start = chunk_id * basin_chunk_size
            basin_stop = min(
                basin_start + basin_chunk_size,
                int(state.ts_ds.sizes["basin"]),
            )
            configured_time_chunk = int(
                self.lazy_cfg.get(
                    "index_scan_time_chunk_size",
                    state.chunk_shape[1] * 8,
                )
            )
            scan_time_chunk = max(8, configured_time_chunk)
            scan_time_chunk -= scan_time_chunk % 8
            for relative_start in range(0, state.time_length, scan_time_chunk):
                relative_stop = min(
                    relative_start + scan_time_chunk,
                    state.time_length,
                )
                block = self._read_block_array(
                    source_idx,
                    source.target_vars,
                    basin_start,
                    basin_stop,
                    state.time_offset + relative_start,
                    state.time_offset + relative_stop,
                    "target_cols",
                )
                byte_start = relative_start // 8
                for local_basin, global_basin in selected:
                    target_values = block[global_basin - basin_start]
                    if self.target_nan_policy == "drop_any_nan":
                        time_has_value = np.all(np.isfinite(target_values), axis=1)
                    else:
                        time_has_value = np.any(np.isfinite(target_values), axis=1)
                    encoded = np.packbits(time_has_value)
                    packed[
                        local_basin,
                        byte_start : byte_start + len(encoded),
                    ] = encoded
        return packed

    @staticmethod
    def _filter_starts_from_target_validity(
        time_has_value: np.ndarray,
        starts: np.ndarray,
        target_length: int,
        target_nan_policy: str,
    ) -> np.ndarray:
        if starts.size == 0:
            return starts
        counts = np.concatenate(([0], np.cumsum(time_has_value, dtype=np.int32)))
        valid_counts = counts[starts + target_length] - counts[starts]
        valid = (
            valid_counts == target_length
            if target_nan_policy == "drop_any_nan"
            else valid_counts > 0
        )
        return starts[valid]

    def _set_lookup_arrays(self, sources, basins, starts, lengths) -> None:
        self._lookup_sources = self._as_compact_unsigned(sources)
        self._lookup_basins = self._as_compact_unsigned(basins)
        self._lookup_starts = self._as_compact_unsigned(starts)
        self._lookup_lengths = self._as_compact_unsigned(lengths)
        self.lookup_table = LookupTableView(
            self._lookup_basins,
            self._lookup_starts,
            self._lookup_lengths,
        )
        self.num_samples = len(self.lookup_table)

    @staticmethod
    def _as_compact_unsigned(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
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

    def _load_batch(self, indices: list[int]):
        item_indices = np.asarray(indices, dtype=np.int64)
        if item_indices.size == 0:
            return torch.empty(0), torch.empty(0)
        sources = self._lookup_sources[item_indices].astype(np.int16, copy=False)
        basins = self._lookup_basins[item_indices].astype(np.int32, copy=False)
        starts = self._lookup_starts[item_indices].astype(np.int32, copy=False)
        lengths = self._lookup_lengths[item_indices].astype(np.int32, copy=False)
        length = int(lengths[0])
        if not np.all(lengths == length):
            raise ValueError(
                "CloudZarrLazyDataset currently supports fixed-length batches."
            )

        y_length = length - self.warmup_length
        batch_size = len(item_indices)
        dyn_feature_count = len(self.relevant_schema)
        static_feature_count = len(self.constant_schema)
        mask_count = len(self._mask_schema) if self._append_missing_masks else 0
        xs = np.empty(
            (batch_size, length, dyn_feature_count + mask_count + static_feature_count),
            dtype=np.float32,
        )
        ys = np.empty((batch_size, y_length, len(self.target_schema)), dtype=np.float32)

        for source_idx, basin_chunk, time_chunk, row_array in self._batch_chunk_groups(
            sources, basins, starts
        ):
            source_idx = int(source_idx)
            state = self.source_states[source_idx]
            source = state.source
            self._ensure_open(source_idx)
            assert state.ts_ds is not None
            assert state.basin_positions is not None

            group_basins = basins[row_array].astype(np.int64, copy=False)
            group_starts = starts[row_array].astype(np.int32, copy=False)
            global_basins = state.basin_positions[group_basins].astype(
                np.int64,
                copy=False,
            )
            global_x_starts = state.time_offset + group_starts - self.warmup_length
            global_y_starts = state.time_offset + group_starts
            basin_chunk_size, time_chunk_size = state.chunk_shape
            basin_block_start = basin_chunk * basin_chunk_size
            basin_block_stop = min(
                basin_block_start + basin_chunk_size,
                int(state.ts_ds.sizes["basin"]),
            )
            time_block_start = time_chunk * time_chunk_size
            required_time_stop = max(
                int((global_x_starts + length).max()),
                int((global_y_starts + y_length).max()),
            )
            time_block_stop = min(
                int(math.ceil(required_time_stop / time_chunk_size) * time_chunk_size),
                int(state.ts_ds.sizes["time"]),
            )

            x_blocks = self._read_cached_normalized_blocks(
                source_idx,
                source.relevant_vars,
                basin_block_start,
                basin_block_stop,
                time_block_start,
                time_block_stop,
                kind="relevant_cols",
            )
            y_blocks = self._read_cached_normalized_blocks(
                source_idx,
                source.target_vars,
                basin_block_start,
                basin_block_stop,
                time_block_start,
                time_block_stop,
                kind="target_cols",
            )
            missing_mask = (
                self._missing_mask_values(source_idx, length)
                if self._append_missing_masks
                else None
            )

            basin_offsets = global_basins - basin_block_start
            xs[row_array, :, :dyn_feature_count] = self._gather_cached_windows(
                x_blocks,
                basin_offsets,
                global_x_starts,
                length,
            )
            next_col = dyn_feature_count
            if missing_mask is not None:
                xs[row_array, :, next_col : next_col + mask_count] = missing_mask
                next_col += mask_count
            if static_feature_count:
                assert state.normalized_static_values is not None
                xs[row_array, :, next_col:] = state.normalized_static_values[
                    group_basins
                ][:, None, :]
            ys[row_array, :, :] = self._gather_cached_windows(
                y_blocks,
                basin_offsets,
                global_y_starts,
                y_length,
            )

        return torch.from_numpy(xs), torch.from_numpy(ys)

    def _batch_chunk_groups(
        self,
        sources: np.ndarray,
        basins: np.ndarray,
        starts: np.ndarray,
    ) -> list[tuple[int, int, int, np.ndarray]]:
        basin_chunks = np.empty(len(sources), dtype=np.int64)
        time_chunks = np.empty(len(sources), dtype=np.int64)
        for source_idx in np.unique(sources):
            rows = np.flatnonzero(sources == source_idx)
            state = self.source_states[int(source_idx)]
            assert state.basin_positions is not None
            global_basins = state.basin_positions[basins[rows].astype(np.int64)]
            global_starts = (
                state.time_offset + starts[rows].astype(np.int64) - self.warmup_length
            )
            basin_chunk_size, time_chunk_size = state.chunk_shape
            basin_chunks[rows] = global_basins // basin_chunk_size
            time_chunks[rows] = global_starts // time_chunk_size

        order = np.lexsort((time_chunks, basin_chunks, sources))
        sorted_sources = sources[order]
        sorted_basins = basin_chunks[order]
        sorted_times = time_chunks[order]
        changes = (
            np.flatnonzero(
                (sorted_sources[1:] != sorted_sources[:-1])
                | (sorted_basins[1:] != sorted_basins[:-1])
                | (sorted_times[1:] != sorted_times[:-1])
            )
            + 1
        )
        boundaries = np.concatenate(([0], changes, [len(order)]))
        return [
            (
                int(sorted_sources[left]),
                int(sorted_basins[left]),
                int(sorted_times[left]),
                order[left:right],
            )
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ]

    def _read_cached_normalized_blocks(
        self,
        source_idx: int,
        variables: list[LazyVariable],
        basin_start: int,
        basin_stop: int,
        time_start: int,
        time_stop: int,
        kind: str,
    ) -> list[tuple[int, np.ndarray]]:
        state = self.source_states[source_idx]
        time_chunk_size = state.chunk_shape[1]
        blocks = []
        for chunk_start in range(time_start, time_stop, time_chunk_size):
            chunk_stop = min(chunk_start + time_chunk_size, time_stop)
            blocks.append(
                (
                    chunk_start,
                    self._read_cached_single_block(
                        source_idx,
                        variables,
                        basin_start,
                        basin_stop,
                        chunk_start,
                        chunk_stop,
                        kind,
                    ),
                )
            )
        return blocks

    @staticmethod
    def _slice_cached_blocks(
        blocks: list[tuple[int, np.ndarray]],
        basin_offset: int,
        time_start: int,
        length: int,
    ) -> np.ndarray:
        feature_count = blocks[0][1].shape[-1]
        time_stop = time_start + length
        for block_start, block in blocks:
            block_stop = block_start + block.shape[1]
            if time_start >= block_start and time_stop <= block_stop:
                source_start = time_start - block_start
                return block[
                    basin_offset,
                    source_start : source_start + length,
                ]

        output = np.empty((length, feature_count), dtype=np.float32)
        destination = 0
        for block_start, block in blocks:
            block_stop = block_start + block.shape[1]
            overlap_start = max(time_start, block_start)
            overlap_stop = min(time_stop, block_stop)
            if overlap_start >= overlap_stop:
                continue
            count = overlap_stop - overlap_start
            source_start = overlap_start - block_start
            output[destination : destination + count] = block[
                basin_offset,
                source_start : source_start + count,
            ]
            destination += count
        if destination != length:
            raise RuntimeError(
                f"Cached Zarr chunks covered {destination} of {length} requested steps."
            )
        return output

    @classmethod
    def _gather_cached_windows(
        cls,
        blocks: list[tuple[int, np.ndarray]],
        basin_offsets: np.ndarray,
        time_starts: np.ndarray,
        length: int,
    ) -> np.ndarray:
        if len(basin_offsets) < 32:
            return np.stack(
                [
                    cls._slice_cached_blocks(
                        blocks,
                        int(basin_offset),
                        int(time_start),
                        length,
                    )
                    for basin_offset, time_start in zip(basin_offsets, time_starts)
                ]
            )
        feature_count = blocks[0][1].shape[-1]
        output = np.empty((len(basin_offsets), length, feature_count), dtype=np.float32)
        pending = np.ones(len(basin_offsets), dtype=bool)
        steps = np.arange(length, dtype=np.int64)
        for block_start, block in blocks:
            relative_starts = time_starts - block_start
            fits = (
                pending
                & (relative_starts >= 0)
                & (relative_starts + length <= block.shape[1])
            )
            if not fits.any():
                continue
            rows = np.flatnonzero(fits)
            time_indices = relative_starts[rows, None] + steps[None, :]
            output[rows] = block[
                basin_offsets[rows, None],
                time_indices,
            ]
            pending[rows] = False
        for row in np.flatnonzero(pending):
            output[row] = cls._slice_cached_blocks(
                blocks,
                int(basin_offsets[row]),
                int(time_starts[row]),
                length,
            )
        return output

    def _read_cached_single_block(
        self,
        source_idx: int,
        variables: list[LazyVariable],
        basin_start: int,
        basin_stop: int,
        time_start: int,
        time_stop: int,
        kind: str,
    ) -> np.ndarray:
        key = (
            source_idx,
            kind,
            basin_start,
            basin_stop,
            time_start,
            time_stop,
        )
        cached = self._block_cache.pop(key, None)
        if cached is not None:
            self._block_cache[key] = cached
            return cached

        state = self.source_states[source_idx]
        block = self._read_block_array(
            source_idx,
            variables,
            basin_start,
            basin_stop,
            time_start,
            time_stop,
            kind,
        )
        block = self._normalize_array(
            block,
            variables,
            kind,
            state.stats or {"enabled": False},
            (state.relevant_names if kind == "relevant_cols" else state.target_names),
        )
        if (
            self._block_cache_limit_bytes <= 0
            or block.nbytes > self._block_cache_limit_bytes
        ):
            return block
        while (
            self._block_cache
            and self._block_cache_bytes + block.nbytes > self._block_cache_limit_bytes
        ):
            _, evicted = self._block_cache.popitem(last=False)
            self._block_cache_bytes -= evicted.nbytes
        self._block_cache[key] = block
        self._block_cache_bytes += block.nbytes
        return block

    def _read_block_array(
        self,
        source_idx: int,
        variables: list[LazyVariable],
        basin_start: int,
        basin_stop: int,
        time_start: int,
        time_stop: int,
        kind: str,
    ) -> np.ndarray:
        self._ensure_open(source_idx)
        state = self.source_states[source_idx]
        assert state.ts_ds is not None
        names = state.relevant_names if kind == "relevant_cols" else state.target_names
        values = np.empty(
            (basin_stop - basin_start, time_stop - time_start, len(variables)),
            dtype=np.float32,
        )
        for col, (var, physical) in enumerate(zip(variables, names)):
            if physical is None:
                values[:, :, col] = np.nan if kind == "target_cols" else var.fill_value
                continue
            raw_values = self._read_physical_block_with_retry(
                source_idx,
                physical,
                basin_start,
                basin_stop,
                time_start,
                time_stop,
            )
            if kind == "target_cols":
                state = self.source_states[source_idx]
                assert state.ts_ds is not None
                values[:, :, col] = self._canonicalize_target_values(
                    raw_values,
                    var,
                    physical,
                    state.ts_ds[physical],
                )
            else:
                values[:, :, col] = raw_values
                if var.scale != 1.0 or var.offset != 0.0:
                    values[:, :, col] = values[:, :, col] * var.scale + var.offset
        return values

    @staticmethod
    def _metadata_missing_values(data_array: xr.DataArray) -> tuple[float, ...]:
        values: list[float] = []
        for metadata in (data_array.attrs, data_array.encoding):
            for key in ("_FillValue", "missing_value"):
                value = metadata.get(key)
                if value is None:
                    continue
                try:
                    candidates = np.asarray(value).reshape(-1)
                except (TypeError, ValueError):
                    continue
                for candidate in candidates:
                    try:
                        candidate_float = float(candidate)
                    except (TypeError, ValueError):
                        continue
                    # Zarr exposes its default storage fill value (0) through
                    # xarray even when the source did not declare zero as a
                    # missing observation. Preserve physical zero streamflow.
                    if key == "_FillValue" and candidate_float == 0.0:
                        continue
                    if math.isfinite(candidate_float) and candidate_float not in values:
                        values.append(candidate_float)
        return tuple(values)

    @staticmethod
    def _is_streamflow_target(variable: LazyVariable, physical: str) -> bool:
        names = (variable.canonical, physical)
        return any(
            token in name.lower()
            for name in names
            for token in ("streamflow", "discharge", "q_cms", "qobs", "q_obs")
        )

    def _canonicalize_target_values(
        self,
        values: np.ndarray,
        variable: LazyVariable,
        physical: str,
        data_array: xr.DataArray,
    ) -> np.ndarray:
        """Decode target missing values and apply physical-domain QC.

        Missing metadata is interpreted in the raw storage domain.  Only values
        that survive that check receive the configured scale/offset transform;
        streamflow domain QC then rejects negative canonical values.
        """
        raw = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(raw)
        for missing_value in self._metadata_missing_values(data_array):
            invalid |= raw == missing_value

        canonical = raw.copy()
        valid = ~invalid
        canonical[valid] = canonical[valid] * variable.scale + variable.offset
        if self._is_streamflow_target(variable, physical):
            invalid |= canonical < 0.0
        canonical[invalid] = np.nan
        return canonical.astype(np.float32, copy=False)

    def _read_physical_block_with_retry(
        self,
        source_idx: int,
        physical: str,
        basin_start: int,
        basin_stop: int,
        time_start: int,
        time_stop: int,
    ) -> np.ndarray:
        max_retries = max(0, int(self.lazy_cfg.get("cloud_read_max_retries", 8)))
        base_delay = max(
            0.0, float(self.lazy_cfg.get("cloud_read_retry_base_seconds", 1.0))
        )
        max_delay = max(
            base_delay,
            float(self.lazy_cfg.get("cloud_read_retry_max_seconds", 30.0)),
        )
        dataset_id = self.source_states[source_idx].source.dataset_id
        for attempt in range(max_retries + 1):
            try:
                self._ensure_open(source_idx)
                state = self.source_states[source_idx]
                assert state.ts_ds is not None
                da = state.ts_ds[physical]
                if "basin" not in da.dims or "time" not in da.dims or len(da.dims) != 2:
                    raise ValueError(
                        f"{dataset_id}:{physical} must have exactly basin/time dimensions."
                    )
                block = da.isel(
                    basin=slice(basin_start, basin_stop),
                    time=slice(time_start, time_stop),
                ).transpose("basin", "time")
                return np.asarray(block.to_numpy(), dtype=np.float32)
            except Exception as exc:
                if attempt >= max_retries or not self._is_retryable_cloud_error(exc):
                    raise
                self._reset_timeseries_connection(source_idx)
                delay = min(max_delay, base_delay * (2**attempt))
                if delay:
                    delay *= random.uniform(0.8, 1.2)
                warnings.warn(
                    f"{dataset_id}:{physical} cloud read failed with "
                    f"{type(exc).__name__}; retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f}s.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time.sleep(delay)
        raise RuntimeError("Unreachable cloud read retry state.")

    def _reset_timeseries_connection(self, source_idx: int) -> None:
        state = self.source_states[source_idx]
        if state.ts_ds is not None:
            try:
                state.ts_ds.close()
            except Exception:
                pass
            state.ts_ds = None
        self._data_sources.pop(state.source.dataset_id, None)

    @staticmethod
    def _is_retryable_cloud_error(exc: Exception) -> bool:
        retryable_names = {
            "ClientConnectorDNSError",
            "ClientConnectorError",
            "ConnectionClosedError",
            "ConnectTimeoutError",
            "EndpointConnectionError",
            "ReadTimeoutError",
            "ResponseStreamingError",
        }
        current: BaseException | None = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (TimeoutError, ConnectionError)):
                return True
            if type(current).__name__ in retryable_names:
                return True
            message = str(current).lower()
            if any(
                marker in message
                for marker in (
                    "dns",
                    "endpoint url",
                    "name resolution",
                    "temporary failure in name resolution",
                    "timeout while contacting dns servers",
                )
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _normalize_array(
        self,
        values: np.ndarray,
        variables: list[LazyVariable],
        kind: str,
        stats: dict[str, Any],
        physical_names: list[str | None],
    ) -> np.ndarray:
        if not stats.get("enabled"):
            return values.astype(np.float32, copy=False)
        out = values.copy()
        for col, (var, physical) in enumerate(zip(variables, physical_names)):
            if physical is None:
                out[..., col] = np.nan if kind == "target_cols" else var.fill_value
                continue
            stat = stats.get(kind, {}).get(var.canonical, {"mean": 0.0, "std": 1.0})
            out[..., col] = (out[..., col] - float(stat["mean"])) / float(stat["std"])
        if kind == "target_cols":
            return out.astype(np.float32, copy=False)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_static(self, values: np.ndarray, source_idx: int) -> np.ndarray:
        state = self.source_states[source_idx]
        stats = state.stats or {"enabled": False}
        if not stats.get("enabled"):
            return np.nan_to_num(values.astype(np.float32), nan=0.0)
        out = values.astype(np.float32, copy=True)
        for col, (var, physical) in enumerate(
            zip(state.source.constant_vars, state.constant_names)
        ):
            if physical is None:
                out[..., col] = var.fill_value
                continue
            stat = stats.get("constant_cols", {}).get(
                var.canonical, {"mean": 0.0, "std": 1.0}
            )
            out[..., col] = (out[..., col] - float(stat["mean"])) / float(stat["std"])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _existing_var_name(self, ds: xr.Dataset, var: LazyVariable) -> str | None:
        if var.physical in ds.data_vars:
            return var.physical
        if var.canonical in ds.data_vars:
            return var.canonical
        return None

    @property
    def _append_missing_masks(self) -> bool:
        return self.lazy_cfg.get("missing_policy") == "fill_and_mask"

    def _missing_mask_values(self, source_idx: int, length: int) -> np.ndarray:
        if not self._mask_schema:
            return np.empty((length, 0), dtype=np.float32)
        source = self.source_states[source_idx].source
        state = self.source_states[source_idx]
        relevant_names = {
            var.canonical: physical
            for var, physical in zip(source.relevant_vars, state.relevant_names)
        }
        constant_names = {
            var.canonical: physical
            for var, physical in zip(source.constant_vars, state.constant_names)
        }
        row = np.zeros((len(self._mask_schema),), dtype=np.float32)
        for pos, canonical in enumerate(self._relevant_mask_schema):
            if relevant_names.get(canonical) is None:
                row[pos] = 1.0
        offset = len(self._relevant_mask_schema)
        for pos, canonical in enumerate(self._constant_mask_schema):
            if constant_names.get(canonical) is None:
                row[offset + pos] = 1.0
        return np.repeat(row[None, :], length, axis=0)
