"""Train TorchHydro with CloudZarrLazyDataset.

Server-oriented entry point for single-dataset CAMELS-US lazy training and
multi-dataset joint lazy training.  The script builds a TorchHydro config,
instantiates CloudZarrLazyDataset once to infer model input/output sizes, then
runs the normal DeepHydro training path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from torchhydro.configs.config import default_config_file
from torchhydro.datasets.cloud_zarr_lazy_dataset import CloudZarrLazyDataset


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_variable_items(value: str | None) -> list[tuple[str, str]]:
    """Parse CLI variable specs.

    Each item may be either ``canonical`` or ``canonical:physical``.  ``=`` is
    accepted as an alias for ``:`` to make shell usage less ambiguous.
    """

    items: list[tuple[str, str]] = []
    for item in parse_csv(value):
        if ":" in item:
            canonical, physical = item.split(":", 1)
        elif "=" in item:
            canonical, physical = item.split("=", 1)
        else:
            canonical, physical = item, item
        canonical = canonical.strip()
        physical = physical.strip()
        if not canonical or not physical:
            raise ValueError(f"Invalid variable mapping: {item!r}")
        items.append((canonical, physical))
    return items


def parse_variable_spec(value: str | None) -> list[str] | dict[str, str]:
    items = parse_variable_items(value)
    if not items:
        return []
    if any(canonical != physical for canonical, physical in items):
        return {canonical: physical for canonical, physical in items}
    return [canonical for canonical, _ in items]


def parse_schema(value: str | None, fallback_specs: str | None) -> list[str]:
    explicit = parse_csv(value)
    if explicit:
        return explicit
    return [canonical for canonical, _ in parse_variable_items(fallback_specs)]


def load_dataset_manifest(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        import yaml

        payload = yaml.safe_load(text)
    if isinstance(payload, list):
        payload = {"datasets": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError("Dataset config must contain a 'datasets' list.")
    return payload


def _format_dataset_uri(
    template: str | None, dataset_id: str, count: int
) -> str | None:
    if not template:
        return None
    if count > 1 and "{dataset_id}" not in template:
        raise ValueError(
            "A shared URI for multiple datasets is ambiguous. Use {dataset_id} "
            "in the URI template or provide per-dataset URIs in --dataset-config."
        )
    return template.format(dataset_id=dataset_id)


def dataset_entries(
    args: argparse.Namespace,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    manifest = manifest or {}
    selected = parse_csv(args.datasets)
    if manifest:
        dataset_defaults = manifest.get("dataset_defaults") or {}
        if not isinstance(dataset_defaults, dict):
            raise ValueError("dataset_defaults must be a mapping.")
        raw_entries = [
            {**dataset_defaults, **dict(entry)} for entry in manifest["datasets"]
        ]
        if selected and selected != ["all"]:
            selected_set = set(selected)
            raw_entries = [
                entry
                for entry in raw_entries
                if entry.get("dataset_id") in selected_set
            ]
            missing = selected_set - {
                str(entry.get("dataset_id")) for entry in raw_entries
            }
            if missing:
                raise ValueError(
                    "Datasets missing from dataset config: "
                    + ", ".join(sorted(missing))
                )
    else:
        if selected == ["all"]:
            raise ValueError("--datasets all requires --dataset-config.")
        raw_entries = [{"dataset_id": dataset_id} for dataset_id in selected]

    entries = []
    for raw_entry in raw_entries:
        dataset_id = str(raw_entry.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValueError("Each dataset config entry requires dataset_id.")
        variables = raw_entry.get("variables") or {}
        configured_time_range = raw_entry.get(
            "time_range", [args.time_start, args.time_end]
        )
        if configured_time_range in (None, "all"):
            time_range: str | list[str] = "all"
        else:
            if (
                not isinstance(configured_time_range, (list, tuple))
                or len(configured_time_range) != 2
            ):
                raise ValueError(
                    f"{dataset_id} time_range must be 'all' or [start, end]."
                )
            time_range = [str(value) for value in configured_time_range]
        entry: dict[str, Any] = {
            **raw_entry,
            "dataset_id": dataset_id,
            "basins": raw_entry.get("basins", "all"),
            "time_range": time_range,
            "relevant_cols": raw_entry.get(
                "relevant_cols",
                variables.get("inputs", parse_variable_spec(args.relevant_cols)),
            ),
            "target_cols": raw_entry.get(
                "target_cols",
                variables.get("targets", parse_variable_spec(args.target_cols)),
            ),
            "constant_cols": raw_entry.get(
                "constant_cols",
                variables.get("constants", parse_variable_spec(args.constant_cols)),
            ),
            "weight": float(raw_entry.get("weight", 1.0)),
        }
        if "timeseries_uri" not in entry:
            uri = _format_dataset_uri(
                args.timeseries_uri,
                dataset_id,
                len(raw_entries),
            )
            if uri:
                entry["timeseries_uri"] = uri
        if "attributes_uri" not in entry:
            uri = _format_dataset_uri(
                args.attributes_uri,
                dataset_id,
                len(raw_entries),
            )
            if uri:
                entry["attributes_uri"] = uri
        if entry.get("max_basins") is None and args.max_basins is not None:
            entry["max_basins"] = int(args.max_basins)
        entry.pop("variables", None)
        entries.append(entry)
    if not entries:
        raise ValueError("No datasets were selected for joint training.")
    return entries


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_config_file()
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_dataset_manifest(getattr(args, "dataset_config", None))
    entries = dataset_entries(args, manifest)
    top_object_ids = (
        entries[0]["basins"]
        if isinstance(entries[0].get("basins"), list)
        else ["00000000"]
    )

    lazy_zarr = {
        "datasets": entries,
        "input_schema": manifest.get("input_schema")
        or parse_schema(args.input_schema, args.relevant_cols),
        "target_schema": manifest.get("target_schema")
        or parse_schema(args.target_schema, args.target_cols),
        "constant_schema": manifest.get("constant_schema")
        or parse_schema(args.constant_schema, args.constant_cols),
        "normalization": args.normalization,
        "normalization_scope": getattr(args, "normalization_scope", "global"),
        "stat_cache_dir": str(case_dir / "lazy_zarr_stat_cache"),
        "cache_stats": not args.no_cache_stats,
        "skip_all_nan_target_windows": not args.no_valid_target_filter,
        "target_nan_policy": (
            "keep"
            if args.no_valid_target_filter
            else getattr(args, "target_nan_policy", "drop_any_nan")
        ),
        "missing_policy": args.missing_policy,
        "worker_cache_size_mb": getattr(args, "worker_cache_size_mb", 64),
        "unit_mismatch_policy": manifest.get(
            "unit_mismatch_policy", getattr(args, "unit_mismatch_policy", "warn")
        ),
        "allow_mixed_time_steps": manifest.get(
            "allow_mixed_time_steps",
            getattr(args, "allow_mixed_time_steps", False),
        ),
        "reader_kwargs": {"source": args.source},
        "use_hydrodataset_reader": args.use_hydrodataset_reader,
        "sampling": args.sampling,
    }
    cfg["data_cfgs"].update(
        {
            "dataset": "CloudZarrLazyDataset",
            "case_dir": str(case_dir),
            "source_cfgs": {
                "dataset_id": parse_csv(args.datasets)[0],
                "other_settings": {"source": args.source},
            },
            "object_ids": top_object_ids,
            "t_range_train": [args.time_start, args.time_end],
            "t_range_valid": None,
            "t_range_test": [args.time_start, args.time_end],
            "target_cols": lazy_zarr["target_schema"],
            "relevant_cols": lazy_zarr["input_schema"],
            "constant_cols": lazy_zarr["constant_schema"],
            "target_rm_nan": False,
            "relevant_rm_nan": False,
            "constant_rm_nan": False,
            "scaler": "StandardScaler",
            "stat_dict_file": None,
            "dataset_id": None,
            "sampler": "CloudZarrChunkBatchSampler" if args.chunk_sampler else None,
            "sampler_hyperparam": {
                "time_chunk_size": args.time_chunk_size,
                "basin_chunk_size": args.basin_chunk_size,
                "sampling": args.sampling,
                "max_batches_per_epoch": args.max_batches_per_epoch,
                "drop_last": getattr(args, "drop_last", False),
            },
            "min_time_unit": "D",
            "min_time_interval": 1,
            "lazy_zarr": lazy_zarr,
        }
    )
    cfg["training_cfgs"].update(
        {
            "batch_size": args.batch_size,
            "hindcast_length": args.hindcast_length,
            "forecast_length": args.forecast_length,
            "warmup_length": args.warmup_length,
            "epochs": args.epochs,
            "start_epoch": 1,
            "train_mode": True,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "dataloader_timeout": getattr(args, "dataloader_timeout", 0.0),
            "dataloader_in_order": getattr(args, "dataloader_in_order", False),
            "batch_level_dataloader": True,
            "non_blocking_transfer": args.pin_memory,
            "cuda_prefetch": args.cuda_prefetch,
            "cuda_prefetch_batches": getattr(args, "cuda_prefetch_batches", 2),
            "performance_monitor": getattr(args, "performance_monitor", False),
            "performance_gpu_sample_interval": getattr(
                args, "performance_gpu_sample_interval", 1.0
            ),
            "performance_gpu_idle_threshold": getattr(
                args, "performance_gpu_idle_threshold", 5.0
            ),
            "which_first_tensor": "sequence",
            "device": [args.device],
            "criterion": "RMSE",
            "optimizer": "Adam",
            "optim_params": {"lr": args.learning_rate},
            "lr_scheduler": {0: 1.0},
            "early_stopping": False,
            "save_epoch": args.save_epoch,
            "save_iter": 0,
            "variable_length_cfgs": {"use_variable_length": False},
        }
    )
    cfg["evaluation_cfgs"].update(
        {
            "rolling": 0,
            "hrwin": None,
            "frwin": None,
            "evaluator": {"recover_mode": "bybasins"},
            "metrics": ["RMSE"],
            "fill_nan": "no",
            "model_loader": {"load_way": "specified", "test_epoch": args.epochs},
        }
    )
    cfg["model_cfgs"].update(
        {
            "model_type": "Normal",
            "model_name": "SimpleLSTM",
            "weight_path": None,
            "continue_train": False,
            "model_hyperparam": {
                "input_size": 1,
                "output_size": 1,
                "hidden_size": args.hidden_size,
                "dr": args.dropout,
            },
        }
    )
    return cfg


def inspect_lazy_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    probe = CloudZarrLazyDataset(cfg, "train")
    try:
        sample_counts = np.bincount(
            np.asarray(probe._lookup_sources, dtype=np.int64),
            minlength=len(probe.sources),
        )
        sources = []
        for source_idx, state in enumerate(probe.source_states):
            sources.append(
                {
                    "dataset_id": state.source.dataset_id,
                    "basins": len(state.source.basins),
                    "time_steps": state.time_length,
                    "time_range": [
                        state.times[0].isoformat(),
                        state.times[-1].isoformat(),
                    ],
                    "time_step": probe.source_time_steps[state.source.dataset_id],
                    "samples": int(sample_counts[source_idx]),
                    "weight": state.source.weight,
                    "zarr_chunk_shape": list(state.chunk_shape),
                    "input_variables": dict(
                        zip(probe.relevant_schema, state.relevant_names)
                    ),
                    "target_variables": dict(
                        zip(probe.target_schema, state.target_names)
                    ),
                    "constant_variables": dict(
                        zip(probe.constant_schema, state.constant_names)
                    ),
                }
            )
        return {
            "input_size": probe.x_feature_count,
            "output_size": probe.y_feature_count,
            "dataset_length": len(probe),
            "lookup_bytes": probe.lookup_nbytes,
            "missing_mask_variables": probe._mask_schema,
            "sources": sources,
        }
    finally:
        probe.close()


def infer_model_sizes(cfg: dict[str, Any]) -> tuple[int, int, int]:
    inspection = inspect_lazy_dataset(cfg)
    return (
        inspection["input_size"],
        inspection["output_size"],
        inspection["dataset_length"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="camels_us")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="JSON/YAML file with per-dataset basins, time range, variables, URIs and weight.",
    )
    parser.add_argument("--source", default="cloud")
    parser.add_argument("--timeseries-uri", default=None)
    parser.add_argument("--attributes-uri", default=None)
    parser.add_argument("--use-hydrodataset-reader", action="store_true", default=True)
    parser.add_argument(
        "--no-hydrodataset-reader",
        dest="use_hydrodataset_reader",
        action="store_false",
    )
    parser.add_argument("--case-dir", default="/tmp/torchhydro_lazy_zarr_train")
    parser.add_argument("--time-start", default="2000-01-01")
    parser.add_argument("--time-end", default="2002-12-31")
    parser.add_argument("--relevant-cols", default="precipitation")
    parser.add_argument("--target-cols", default="streamflow")
    parser.add_argument("--constant-cols", default="")
    parser.add_argument("--input-schema", default="")
    parser.add_argument("--target-schema", default="")
    parser.add_argument("--constant-schema", default="")
    parser.add_argument("--missing-policy", default="fill_zero")
    parser.add_argument("--normalization", default="standard")
    parser.add_argument(
        "--normalization-scope",
        choices=("global", "per_dataset"),
        default="global",
    )
    parser.add_argument("--no-cache-stats", action="store_true")
    parser.add_argument("--no-valid-target-filter", action="store_true")
    parser.add_argument(
        "--target-nan-policy",
        choices=("keep", "drop_all_nan", "drop_any_nan"),
        default="drop_any_nan",
        help="drop_any_nan avoids silently training against zero-filled missing targets.",
    )
    parser.add_argument(
        "--max-basins",
        type=int,
        default=None,
        help="Optional per-dataset basin limit. Omit for full joint training.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hindcast-length", type=int, default=30)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--warmup-length", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--worker-cache-size-mb", type=float, default=64)
    parser.add_argument(
        "--unit-mismatch-policy",
        choices=("ignore", "warn", "error"),
        default="warn",
    )
    parser.add_argument("--allow-mixed-time-steps", action="store_true")
    parser.add_argument("--dataloader-timeout", type=float, default=0.0)
    parser.add_argument(
        "--dataloader-in-order",
        action="store_true",
        help="Force DataLoader to wait for batches in submission order.",
    )
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--cuda-prefetch", action="store_true")
    parser.add_argument("--cuda-prefetch-batches", type=int, default=2)
    parser.add_argument(
        "--performance-monitor",
        action="store_true",
        help="Write per-epoch loader, compute, throughput and GPU metrics as JSON.",
    )
    parser.add_argument("--performance-gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--performance-gpu-idle-threshold", type=float, default=5.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--no-chunk-sampler", dest="chunk_sampler", action="store_false"
    )
    parser.add_argument(
        "--sampling",
        choices=("weighted_by_sample", "balanced_by_dataset", "weighted_by_dataset"),
        default="weighted_by_sample",
    )
    parser.add_argument("--time-chunk-size", type=int, default=None)
    parser.add_argument("--basin-chunk-size", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the final incomplete batch. Disabled by default so a full epoch keeps every sample.",
    )
    parser.add_argument("--save-epoch", type=int, default=0)
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_cfg(args)
    inspection = inspect_lazy_dataset(cfg)
    input_size = inspection["input_size"]
    output_size = inspection["output_size"]
    dataset_length = inspection["dataset_length"]
    cfg["model_cfgs"]["model_hyperparam"]["input_size"] = input_size
    cfg["model_cfgs"]["model_hyperparam"]["output_size"] = output_size
    print(
        json.dumps(
            {
                "datasets": args.datasets,
                "dataset_length": dataset_length,
                "input_size": input_size,
                "output_size": output_size,
                "case_dir": args.case_dir,
                "chunk_sampler": args.chunk_sampler,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "worker_cache_size_mb": args.worker_cache_size_mb,
                "cuda_prefetch": args.cuda_prefetch,
                "lookup_bytes": inspection["lookup_bytes"],
                "missing_mask_variables": inspection["missing_mask_variables"],
                "sources": inspection["sources"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.dry_run:
        return 0
    from torchhydro.trainers.deep_hydro import DeepHydro

    model = DeepHydro(cfg)
    model.model_train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
