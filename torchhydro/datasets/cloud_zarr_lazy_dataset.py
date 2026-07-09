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
from dataclasses import dataclass
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


@dataclass
class LazyDatasetSource:
    """One source dataset participating in lazy loading.

    The first implementation is optimized for a single source, but the shape is
    already explicit enough for multi-source indexing and dataset-level
    sampling.
    """

    dataset_id: str
    basins: list[str]
    time_range: list[str]
    relevant_vars: list[LazyVariable]
    target_vars: list[LazyVariable]
    constant_vars: list[LazyVariable]
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
    static_values: np.ndarray | None = None
    stats: dict[str, Any] | None = None


class CloudZarrStatCache:
    """Small mean/std cache for lazy Zarr normalization.

    This is deliberately simpler than ``DapengScaler``.  It gives lazy training
    stable normalization without loading all arrays into memory and can later be
    extended to emit TorchHydro-compatible stat dictionaries.
    """

    version = 1

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
            return json.loads(path.read_text(encoding="utf-8"))
        stats = builder()
        stats["version"] = self.version
        stats["cache_key"] = key
        if self.enabled:
            path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
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
        self.is_tra_val_te = is_tra_val_te
        self.train_mode = is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = int(self.training_cfgs["hindcast_length"])
        self.warmup_length = int(self.training_cfgs["warmup_length"])
        self.horizon = int(self.training_cfgs["forecast_length"])
        self._batch_level_dataloader_enabled = False
        self._open_pid = os.getpid()

        self._data_sources: dict[str, Any] = {}
        self.sources = self._build_sources()
        self._align_source_schemas()
        self.source_states = [LazySourceState(source=source) for source in self.sources]
        for source_idx in range(len(self.source_states)):
            self._ensure_open(source_idx)
            self._prepare_time_index(source_idx)
            self._prepare_static_attributes(source_idx)
            self.source_states[source_idx].stats = self._load_or_build_stats(source_idx)
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
        state["source_states"] = [
            LazySourceState(
                source=source_state.source,
                basin_positions=source_state.basin_positions,
                time_offset=source_state.time_offset,
                time_length=source_state.time_length,
                times=source_state.times,
                static_values=source_state.static_values,
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

    def _build_sources(self) -> list[LazyDatasetSource]:
        dataset_entries = self.lazy_cfg.get("datasets")
        if not dataset_entries:
            source_cfgs = self.data_cfgs.get("source_cfgs") or {}
            dataset_entries = [
                {
                    "dataset_id": source_cfgs.get("dataset_id"),
                    "basins": self.data_cfgs.get("object_ids"),
                    "time_range": self.t_s_dict["t_final_range"],
                    "relevant_cols": self.data_cfgs.get("relevant_cols") or [],
                    "target_cols": self.data_cfgs.get("target_cols") or [],
                    "constant_cols": self.data_cfgs.get("constant_cols") or [],
                    "timeseries_uri": self.lazy_cfg.get("timeseries_uri"),
                    "attributes_uri": self.lazy_cfg.get("attributes_uri"),
                }
            ]

        sources = []
        for entry in dataset_entries:
            dataset_id = entry.get("dataset_id")
            if not dataset_id:
                raise ValueError("lazy_zarr dataset entry must include dataset_id.")
            timeseries_uri = entry.get("timeseries_uri") or self.lazy_cfg.get("timeseries_uri")
            attributes_uri = entry.get("attributes_uri") or self.lazy_cfg.get("attributes_uri")
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
                    try:
                        basins = [str(value) for value in ts["basin"].values]
                    finally:
                        ts.close()
                else:
                    raise ValueError(
                        "basins must be provided when hydrodataset reader is disabled."
                    )
            basins = [str(value) for value in basins]
            time_range = entry.get("time_range") or self.t_s_dict["t_final_range"]
            relevant_cols = entry.get("relevant_cols", self.data_cfgs.get("relevant_cols") or [])
            target_cols = entry.get("target_cols", self.data_cfgs.get("target_cols") or [])
            constant_cols = entry.get(
                "constant_cols", self.data_cfgs.get("constant_cols") or []
            )
            sources.append(
                LazyDatasetSource(
                    dataset_id=dataset_id,
                    basins=basins,
                    time_range=list(time_range),
                    relevant_vars=self._resolve_dynamic_variables(reader, relevant_cols),
                    target_vars=self._resolve_dynamic_variables(reader, target_cols),
                    constant_vars=self._resolve_static_variables(reader, constant_cols),
                    timeseries_uri=timeseries_uri,
                    attributes_uri=attributes_uri,
                    weight=float(entry.get("weight", 1.0)),
                )
            )
        return sources

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

        mask_schema = []
        for canonical in self.relevant_schema:
            if any(
                not any(
                    var.canonical == canonical and var.required
                    for var in source.relevant_vars
                )
                for source in self.sources
            ):
                mask_schema.append(canonical)
        self._mask_schema = mask_schema

    def _resolve_dynamic_variables(self, reader, cols: Any) -> list[LazyVariable]:
        if isinstance(cols, dict):
            items = cols.items()
        else:
            items = ((col, col) for col in (cols or []))
        return [
            self._resolve_one_dynamic_variable(reader, str(canonical), spec)
            for canonical, spec in items
        ]

    def _resolve_one_dynamic_variable(self, reader, canonical: str, spec: Any) -> LazyVariable:
        physical = canonical
        unit = None
        required = True
        fill_value = 0.0
        if isinstance(spec, dict):
            physical = str(spec.get("physical") or spec.get("name") or canonical)
            unit = spec.get("unit")
            required = bool(spec.get("required", True))
            fill_value = float(spec.get("fill_value", 0.0))
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
            if isinstance(spec, dict):
                physical = str(spec.get("physical") or spec.get("name") or canonical)
                unit = spec.get("unit")
                required = bool(spec.get("required", True))
                fill_value = float(spec.get("fill_value", 0.0))
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
                )
        )
        return variables

    def _reset_open_handles_after_fork(self) -> None:
        current_pid = os.getpid()
        if getattr(self, "_open_pid", None) == current_pid:
            return
        for source_state in self.source_states:
            for attr in ("ts_ds", "attr_ds"):
                ds = getattr(source_state, attr)
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
                    setattr(source_state, attr, None)
        self._data_sources = {}
        self._open_pid = current_pid

    def _ensure_open(self, source_idx: int | None = None) -> None:
        self._reset_open_handles_after_fork()
        if source_idx is None:
            for idx in range(len(self.source_states)):
                self._ensure_open(idx)
            return
        state = self.source_states[source_idx]
        source = state.source
        if state.ts_ds is not None:
            return
        state.ts_ds = self._open_dataset_store(
            source,
            source.timeseries_uri,
            fallback_loader="_load_ts_dataset",
        )
        if source.constant_vars:
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
        return getattr(reader, fallback_loader)()

    def _open_uri_dataset(self, uri: str) -> xr.Dataset:
        storage_options = self.lazy_cfg.get("storage_options")
        if uri.startswith("s3://") or uri.endswith(".zarr"):
            return xr.open_zarr(
                uri,
                storage_options=storage_options,
                consolidated=False,
                mask_and_scale=False,
            )
        return xr.open_dataset(uri, mask_and_scale=False)

    def _prepare_time_index(self, source_idx: int) -> None:
        self._ensure_open(source_idx)
        state = self.source_states[source_idx]
        source = state.source
        assert state.ts_ds is not None
        if "basin" not in state.ts_ds.coords or "time" not in state.ts_ds.coords:
            raise ValueError("Zarr timeseries store must have basin and time coordinates.")

        zarr_basins = {str(value): pos for pos, value in enumerate(state.ts_ds["basin"].values)}
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
        start = pd.to_datetime(source.time_range[0])
        end = pd.to_datetime(source.time_range[1])
        mask = (full_times >= start) & (full_times <= end)
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            raise ValueError(
                f"No time points found for {source.dataset_id} in {source.time_range}."
            )
        state.time_offset = int(positions[0])
        state.time_length = int(positions.size)
        state.times = full_times[positions]
        if source_idx == 0:
            self.times = state.times

    def _prepare_static_attributes(self, source_idx: int) -> None:
        state = self.source_states[source_idx]
        source = state.source
        state.static_values = None
        if not source.constant_vars:
            return
        self._ensure_open(source_idx)
        assert state.attr_ds is not None
        values = np.empty(
            (len(source.basins), len(source.constant_vars)),
            dtype=np.float32,
        )
        attr_basins = {str(value): pos for pos, value in enumerate(state.attr_ds["basin"].values)}
        for col, var in enumerate(source.constant_vars):
            physical = self._existing_var_name(state.attr_ds, var)
            if physical is None:
                if var.required:
                    raise ValueError(
                        f"Required static variable {var.canonical} ({var.physical}) "
                        "is missing from attributes store."
                    )
                values[:, col] = var.fill_value
                continue
            basin_positions = [attr_basins[basin] for basin in source.basins]
            arr = state.attr_ds[physical].isel(basin=basin_positions).to_numpy()
            values[:, col] = np.asarray(arr, dtype=np.float32)
        state.static_values = values

    def _load_or_build_stats(self, source_idx: int) -> dict[str, Any]:
        state = self.source_states[source_idx]
        source = state.source
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
            "dataset_id": source.dataset_id,
            "basins": source.basins,
            "time_range": source.time_range,
            "relevant": [var.__dict__ for var in source.relevant_vars],
            "target": [var.__dict__ for var in source.target_vars],
            "constant": [var.__dict__ for var in source.constant_vars],
            "normalization": normalization,
        }
        return stat_cache.load_or_build(
            key_payload,
            lambda: self._build_standard_stats(source_idx),
        )

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
            for var in variables:
                physical = self._existing_var_name(state.ts_ds, var)
                if physical is None:
                    stats[kind][var.canonical] = {"mean": 0.0, "std": 1.0}
                    continue
                da = state.ts_ds[physical].isel(
                    time=slice(state.time_offset, state.time_offset + state.time_length),
                )
                stats[kind][var.canonical] = self._mean_std_dynamic(
                    da,
                    state.basin_positions,
                    state.time_length,
                )
        if state.static_values is not None:
            for col, var in enumerate(source.constant_vars):
                stats["constant_cols"][var.canonical] = self._mean_std(state.static_values[:, col])
        return stats

    def _mean_std_dynamic(
        self,
        da: xr.DataArray,
        basin_positions: np.ndarray,
        time_length: int,
    ) -> dict[str, float]:
        basin_chunk_size = int(self.lazy_cfg.get("stat_basin_chunk_size", 128))
        time_chunk_size = int(self.lazy_cfg.get("stat_time_chunk_size", 3650))
        basin_chunk_size = max(1, basin_chunk_size)
        time_chunk_size = max(1, time_chunk_size)
        count = 0
        total = 0.0
        total_sq = 0.0
        for basin_start in range(0, len(basin_positions), basin_chunk_size):
            basin_index = basin_positions[basin_start : basin_start + basin_chunk_size]
            for time_start in range(0, time_length, time_chunk_size):
                time_stop = min(time_start + time_chunk_size, time_length)
                block = da.isel(
                    basin=basin_index,
                    time=slice(time_start, time_stop),
                ).to_numpy()
                arr = np.asarray(block, dtype=np.float64)
                valid = np.isfinite(arr)
                if not valid.any():
                    continue
                values = arr[valid]
                count += int(values.size)
                total += float(values.sum())
                total_sq += float(np.square(values).sum())
        if count == 0:
            return {"mean": 0.0, "std": 1.0}
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        if not math.isfinite(mean):
            mean = 0.0
        return {"mean": float(mean), "std": float(std)}

    @staticmethod
    def _mean_std(value: Any) -> dict[str, float]:
        if hasattr(value, "mean") and hasattr(value, "std") and not isinstance(value, np.ndarray):
            mean_value = value.mean(skipna=True).to_numpy()
            std_value = value.std(skipna=True).to_numpy()
            mean = float(np.asarray(mean_value))
            std = float(np.asarray(std_value))
        else:
            arr = np.asarray(value)
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
        if not math.isfinite(mean):
            mean = 0.0
        if not math.isfinite(std) or std == 0.0:
            std = 1.0
        return {"mean": mean, "std": std}

    def _create_lookup_table(self) -> None:
        warmup = self.warmup_length
        seq_len = self.rho + self.horizon
        source_chunks = []
        basin_chunks = []
        start_chunks = []
        length_chunks = []
        filter_targets = bool(self.lazy_cfg.get("skip_all_nan_target_windows", True))
        for source_idx, state in enumerate(self.source_states):
            max_start = state.time_length - seq_len + 1
            if max_start <= warmup:
                continue
            starts_template = np.arange(warmup, max_start, dtype=np.int32)
            for basin_pos in range(len(state.source.basins)):
                starts = starts_template
                if filter_targets and self.train_mode:
                    starts = self._filter_valid_target_starts(
                        source_idx,
                        basin_pos,
                        starts,
                        seq_len,
                    )
                if starts.size:
                    source_chunks.append(np.full(starts.shape, source_idx, dtype=np.int16))
                    basin_chunks.append(np.full(starts.shape, basin_pos, dtype=np.int32))
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

    def _filter_valid_target_starts(
        self,
        source_idx: int,
        basin_pos: int,
        starts: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        if starts.size == 0:
            return starts
        target = self._read_window_array(
            source_idx,
            self.source_states[source_idx].source.target_vars,
            basin_pos,
            x_start=0,
            length=self.source_states[source_idx].time_length,
            kind="target_cols",
        )
        nan_array = np.isnan(target)
        y_window = seq_len - self.warmup_length
        time_has_value = ~np.all(nan_array, axis=1)
        counts = np.concatenate(([0], np.cumsum(time_has_value, dtype=np.int32)))
        valid = (counts[starts + y_window] - counts[starts]) > 0
        return starts[valid]

    def _set_lookup_arrays(self, sources, basins, starts, lengths) -> None:
        self._lookup_sources = sources.astype(np.int16, copy=False)
        self._lookup_basins = basins.astype(np.int32, copy=False)
        self._lookup_starts = starts.astype(np.int32, copy=False)
        self._lookup_lengths = lengths.astype(np.int32, copy=False)
        self.lookup_table = LookupTableView(
            self._lookup_basins,
            self._lookup_starts,
            self._lookup_lengths,
        )
        self.num_samples = len(self.lookup_table)

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
            raise ValueError("CloudZarrLazyDataset currently supports fixed-length batches.")

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

        group_keys: dict[tuple[int, int], list[int]] = {}
        for row, (source_idx, basin_pos) in enumerate(zip(sources, basins)):
            group_keys.setdefault((int(source_idx), int(basin_pos)), []).append(row)

        for (source_idx, basin_pos), rows in group_keys.items():
            source_idx = int(source_idx)
            state = self.source_states[source_idx]
            source = state.source
            self._ensure_open(source_idx)

            row_array = np.asarray(rows, dtype=np.int64)
            group_starts = starts[row_array].astype(np.int32, copy=False)
            x_starts = group_starts - self.warmup_length
            y_starts = group_starts
            x_block_start = int(x_starts.min())
            x_block_stop = int((x_starts + length).max())
            y_block_start = int(y_starts.min())
            y_block_stop = int((y_starts + y_length).max())

            x_block = self._read_window_array(
                source_idx,
                source.relevant_vars,
                basin_pos,
                x_start=x_block_start,
                length=x_block_stop - x_block_start,
                kind="relevant_cols",
            )
            y_block = self._read_window_array(
                source_idx,
                source.target_vars,
                basin_pos,
                x_start=y_block_start,
                length=y_block_stop - y_block_start,
                kind="target_cols",
            )
            x_block = self._normalize_array(
                x_block,
                source.relevant_vars,
                "relevant_cols",
                state.stats or {"enabled": False},
            )
            y_block = self._normalize_array(
                y_block,
                source.target_vars,
                "target_cols",
                state.stats or {"enabled": False},
            )

            missing_mask = (
                self._missing_mask_values(source_idx, length)
                if self._append_missing_masks
                else None
            )
            static_values = None
            if static_feature_count:
                assert state.static_values is not None
                static_values = self._normalize_static(
                    state.static_values[basin_pos, :],
                    source_idx,
                )

            for row, start in zip(rows, group_starts):
                x_offset = int(start - self.warmup_length - x_block_start)
                y_offset = int(start - y_block_start)
                xs[row, :, :dyn_feature_count] = x_block[
                    x_offset : x_offset + length,
                    :,
                ]
                next_col = dyn_feature_count
                if missing_mask is not None:
                    xs[row, :, next_col : next_col + mask_count] = missing_mask
                    next_col += mask_count
                if static_values is not None:
                    xs[row, :, next_col:] = static_values[None, :]
                ys[row, :, :] = y_block[y_offset : y_offset + y_length, :]

        return torch.from_numpy(xs), torch.from_numpy(ys)

    def _read_window_array(
        self,
        source_idx: int,
        variables: list[LazyVariable],
        basin_pos: int,
        x_start: int,
        length: int,
        kind: str,
    ) -> np.ndarray:
        self._ensure_open(source_idx)
        state = self.source_states[source_idx]
        assert state.ts_ds is not None
        assert state.basin_positions is not None
        values = np.empty((length, len(variables)), dtype=np.float32)
        global_basin_pos = int(state.basin_positions[basin_pos])
        global_time_start = state.time_offset + int(x_start)
        time_slice = slice(global_time_start, global_time_start + length)
        for col, var in enumerate(variables):
            physical = self._existing_var_name(state.ts_ds, var)
            if physical is None:
                if var.required:
                    raise ValueError(
                        f"Required {kind} variable {var.canonical} ({var.physical}) "
                        "is missing from timeseries store."
                    )
                values[:, col] = var.fill_value
                continue
            da = state.ts_ds[physical].isel(basin=global_basin_pos, time=time_slice)
            values[:, col] = np.asarray(da.to_numpy(), dtype=np.float32)
        return values

    def _normalize_array(
        self,
        values: np.ndarray,
        variables: list[LazyVariable],
        kind: str,
        stats: dict[str, Any],
    ) -> np.ndarray:
        if not stats.get("enabled"):
            return np.nan_to_num(values, nan=0.0)
        out = values.copy()
        for col, var in enumerate(variables):
            stat = stats.get(kind, {}).get(var.canonical, {"mean": 0.0, "std": 1.0})
            out[:, col] = (out[:, col] - float(stat["mean"])) / float(stat["std"])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_static(self, values: np.ndarray, source_idx: int) -> np.ndarray:
        state = self.source_states[source_idx]
        stats = state.stats or {"enabled": False}
        if not stats.get("enabled"):
            return np.nan_to_num(values.astype(np.float32), nan=0.0)
        out = values.astype(np.float32, copy=True)
        for col, var in enumerate(state.source.constant_vars):
            stat = stats.get("constant_cols", {}).get(
                var.canonical, {"mean": 0.0, "std": 1.0}
            )
            out[col] = (out[col] - float(stat["mean"])) / float(stat["std"])
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

    @property
    def _missing_relevant_vars(self) -> list[LazyVariable]:
        return [
            LazyVariable(canonical=name, physical=name, required=False)
            for name in self._mask_schema
        ]

    def _missing_mask_values(self, source_idx: int, length: int) -> np.ndarray:
        if not self._mask_schema:
            return np.empty((length, 0), dtype=np.float32)
        source = self.source_states[source_idx].source
        vars_by_name = {var.canonical: var for var in source.relevant_vars}
        row = np.zeros((len(self._mask_schema),), dtype=np.float32)
        for pos, canonical in enumerate(self._mask_schema):
            var = vars_by_name.get(canonical)
            if var is None or not var.required:
                row[pos] = 1.0
        return np.repeat(row[None, :], length, axis=0)
