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


def dataset_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = []
    for dataset_id in parse_csv(args.datasets):
        entry: dict[str, Any] = {
            "dataset_id": dataset_id,
            "basins": "all",
            "time_range": [args.time_start, args.time_end],
            "relevant_cols": parse_variable_spec(args.relevant_cols),
            "target_cols": parse_variable_spec(args.target_cols),
            "constant_cols": parse_variable_spec(args.constant_cols),
            "weight": 1.0,
        }
        if args.timeseries_uri:
            entry["timeseries_uri"] = args.timeseries_uri
        if args.attributes_uri:
            entry["attributes_uri"] = args.attributes_uri
        if args.max_basins is not None:
            if args.timeseries_uri and not args.use_hydrodataset_reader:
                import xarray as xr

                ds = xr.open_zarr(args.timeseries_uri, consolidated=False)
                try:
                    entry["basins"] = [
                        str(value) for value in ds["basin"].values[: args.max_basins]
                    ]
                finally:
                    ds.close()
            else:
                from torchhydro.configs.data_resolver import open_dataset

                reader = open_dataset(dataset_id, source=args.source)
                entry["basins"] = [
                    str(value) for value in reader.read_object_ids()[: args.max_basins]
                ]
        entries.append(entry)
    if not entries:
        raise ValueError("--datasets must contain at least one dataset id.")
    return entries


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_config_file()
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    entries = dataset_entries(args)
    top_object_ids = (
        entries[0]["basins"]
        if isinstance(entries[0].get("basins"), list)
        else ["00000000"]
    )

    lazy_zarr = {
        "datasets": entries,
        "input_schema": parse_schema(args.input_schema, args.relevant_cols),
        "target_schema": parse_schema(args.target_schema, args.target_cols),
        "constant_schema": parse_schema(args.constant_schema, args.constant_cols),
        "normalization": args.normalization,
        "stat_cache_dir": str(case_dir / "lazy_zarr_stat_cache"),
        "cache_stats": not args.no_cache_stats,
        "skip_all_nan_target_windows": not args.no_valid_target_filter,
        "missing_policy": args.missing_policy,
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
                "drop_last": True,
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
            "batch_level_dataloader": True,
            "non_blocking_transfer": args.pin_memory,
            "cuda_prefetch": args.cuda_prefetch,
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


def infer_model_sizes(cfg: dict[str, Any]) -> tuple[int, int, int]:
    probe = CloudZarrLazyDataset(cfg, "train")
    return probe.x_feature_count, probe.y_feature_count, len(probe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="camels_us")
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
    parser.add_argument("--no-cache-stats", action="store_true")
    parser.add_argument("--no-valid-target-filter", action="store_true")
    parser.add_argument("--max-basins", type=int, default=64)
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
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--cuda-prefetch", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-chunk-sampler", dest="chunk_sampler", action="store_false")
    parser.add_argument("--sampling", default="weighted_by_sample")
    parser.add_argument("--time-chunk-size", type=int, default=None)
    parser.add_argument("--basin-chunk-size", type=int, default=1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--save-epoch", type=int, default=0)
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_cfg(args)
    input_size, output_size, dataset_length = infer_model_sizes(cfg)
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
                "cuda_prefetch": args.cuda_prefetch,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        return 0
    from torchhydro.trainers.deep_hydro import DeepHydro

    model = DeepHydro(cfg)
    model.model_train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
