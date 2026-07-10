"""Benchmark CloudZarrLazyDataset data loading and light GPU training.

This script is intended for the server with hydrodataset cloud settings.  It
does not eager-load the full Zarr store; batches are read through
CloudZarrLazyDataset.__getitems__.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from torchhydro.configs.config import default_config_file
from torchhydro.datasets.cloud_zarr_lazy_dataset import CloudZarrLazyDataset
from torchhydro.datasets.cloud_zarr_sampler import CloudZarrChunkBatchSampler
from torchhydro.trainers.cuda_prefetcher import CudaPrefetcher


def prebatched_collate_fn(batch):
    return batch


def log_progress(args: argparse.Namespace, message: str) -> None:
    if args.quiet:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def process_rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def query_gpu() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    first = out.splitlines()[0]
    util, used, total = [item.strip() for item in first.split(",")]
    return {
        "gpu_util_percent": float(util),
        "gpu_memory_used_mib": float(used),
        "gpu_memory_total_mib": float(total),
    }


class GpuMonitor:
    """Collect nvidia-smi samples without blocking the benchmark loop."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="torchhydro-gpu-monitor",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._thread.join(timeout=3)
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_variable_items(value: str | None) -> list[tuple[str, str]]:
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


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_config_file()
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    dataset_entry = {
        "dataset_id": args.dataset_id,
        "basins": "all",
        "max_basins": args.max_basins,
        "time_range": [args.time_start, args.time_end],
        "target_cols": parse_variable_spec(args.target_cols),
        "relevant_cols": parse_variable_spec(args.relevant_cols),
        "constant_cols": parse_variable_spec(args.constant_cols),
    }
    if args.timeseries_uri:
        dataset_entry["timeseries_uri"] = args.timeseries_uri
    if args.attributes_uri:
        dataset_entry["attributes_uri"] = args.attributes_uri
    cfg["data_cfgs"].update(
        {
            "dataset": "CloudZarrLazyDataset",
            "case_dir": str(case_dir),
            "source_cfgs": {
                "dataset_id": args.dataset_id,
                "other_settings": {"source": args.source},
            },
            # CloudZarrLazyDataset reads real basin lists from lazy_zarr.datasets.
            # The placeholder only satisfies TorchHydro's shared config wrapper.
            "object_ids": ["00000000"],
            "t_range_train": [args.time_start, args.time_end],
            "t_range_valid": None,
            "t_range_test": [args.time_start, args.time_end],
            "target_cols": parse_variable_spec(args.target_cols),
            "relevant_cols": parse_variable_spec(args.relevant_cols),
            "constant_cols": parse_variable_spec(args.constant_cols),
            "target_rm_nan": False,
            "relevant_rm_nan": False,
            "constant_rm_nan": False,
            "scaler": "StandardScaler",
            "stat_dict_file": None,
            "min_time_unit": "D",
            "min_time_interval": 1,
            "lazy_zarr": {
                "datasets": [dataset_entry],
                "normalization": args.normalization,
                "stat_cache_dir": str(case_dir / "lazy_zarr_stat_cache"),
                "cache_stats": not args.no_cache_stats,
                "skip_all_nan_target_windows": args.valid_target_filter,
                "target_nan_policy": (
                    "drop_all_nan" if args.valid_target_filter else "keep"
                ),
                "missing_policy": args.missing_policy,
                "worker_cache_size_mb": args.worker_cache_size_mb,
                "use_hydrodataset_reader": args.use_hydrodataset_reader,
                "reader_kwargs": {"source": args.source},
            },
        }
    )
    cfg["training_cfgs"].update(
        {
            "batch_size": args.batch_size,
            "hindcast_length": args.hindcast_length,
            "forecast_length": args.forecast_length,
            "warmup_length": args.warmup_length,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "batch_level_dataloader": True,
            "valid_batch_mode": "valid",
            "variable_length_cfgs": {"use_variable_length": False},
        }
    )
    cfg["evaluation_cfgs"].update(
        {
            "rolling": 0,
            "hrwin": None,
            "frwin": None,
            "evaluator": {"recover_mode": "bybasins"},
        }
    )
    return cfg


def make_loader(dataset: CloudZarrLazyDataset, args: argparse.Namespace) -> DataLoader:
    dataset._batch_level_dataloader_enabled = True
    kwargs = {}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
        if "in_order" in inspect.signature(DataLoader).parameters:
            kwargs["in_order"] = args.dataloader_in_order
    batch_sampler = None
    if args.chunk_sampler:
        batch_sampler = CloudZarrChunkBatchSampler(
            dataset,
            batch_size=args.batch_size,
            time_chunk_size=args.time_chunk_size,
            basin_chunk_size=args.basin_chunk_size,
            sampling=args.sampling,
            max_batches_per_epoch=args.max_batches_per_epoch,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            timeout=args.dataloader_timeout,
            collate_fn=prebatched_collate_fn,
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        timeout=args.dataloader_timeout,
        collate_fn=prebatched_collate_fn,
        **kwargs,
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    log_progress(args, "building config")
    cfg = build_cfg(args)
    log_progress(args, "initializing CloudZarrLazyDataset")
    init_start = time.perf_counter()
    dataset = CloudZarrLazyDataset(cfg, "train")
    init_seconds = time.perf_counter() - init_start
    log_progress(
        args,
        (
            "dataset initialized: "
            f"length={len(dataset)}, basins={dataset.ngrid}, "
            f"nt={dataset.nt}, seconds={init_seconds:.2f}"
        ),
    )
    log_progress(args, "building DataLoader")
    loader = make_loader(dataset, args)
    log_progress(args, f"DataLoader ready: batches={len(loader)}")
    device = torch.device(args.device)
    use_cuda_prefetch = bool(args.cuda_prefetch and device.type == "cuda")
    model = None
    optimizer = None
    if not args.no_train:
        log_progress(args, f"initializing benchmark model on {device}")
        model = torch.nn.LSTM(
            input_size=dataset.x_feature_count,
            hidden_size=args.hidden_size,
            batch_first=True,
        ).to(device)
        head = torch.nn.Linear(args.hidden_size, dataset.y_feature_count).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=1e-3
        )
    else:
        head = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wait_times = []
    compute_times = []
    batch_sizes = []
    gpu_monitor = (
        GpuMonitor(args.gpu_monitor_interval) if device.type == "cuda" else None
    )
    if gpu_monitor is not None:
        gpu_monitor.start()
    peak_rss = process_rss()
    iterable = (
        CudaPrefetcher(
            loader,
            device,
            non_blocking=args.pin_memory,
            prefetch_batches=args.cuda_prefetch_batches,
        )
        if use_cuda_prefetch
        else loader
    )
    iterator = iter(iterable)
    log_progress(args, f"starting benchmark loop for up to {args.steps} steps")
    started = time.perf_counter()
    for step in range(args.steps):
        wait_start = time.perf_counter()
        try:
            x, y = next(iterator)
        except StopIteration:
            break
        wait_times.append(time.perf_counter() - wait_start)
        if step == 0:
            log_progress(
                args,
                (
                    "first batch loaded: "
                    f"x={tuple(x.shape)}, y={tuple(y.shape)}, "
                    f"wait={wait_times[-1]:.2f}s"
                ),
            )
        batch_sizes.append(int(x.shape[0]))

        compute_start = time.perf_counter()
        if not use_cuda_prefetch:
            x = x.to(device, non_blocking=args.pin_memory)
            y = y.to(device, non_blocking=args.pin_memory)
        if model is not None and optimizer is not None and head is not None:
            optimizer.zero_grad(set_to_none=True)
            out, _ = model(x.float())
            pred = head(out[:, -y.shape[1] :, :])
            loss = torch.mean((pred - y.float()) ** 2)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_times.append(time.perf_counter() - compute_start)
        current_rss = process_rss()
        if current_rss is not None:
            peak_rss = current_rss if peak_rss is None else max(peak_rss, current_rss)

    elapsed = time.perf_counter() - started
    gpu_samples = gpu_monitor.stop() if gpu_monitor is not None else []
    total_samples = sum(batch_sizes)
    result = {
        "dataset_id": args.dataset_id,
        "dataset_length": len(dataset),
        "dataset_init_seconds": init_seconds,
        "steps": len(batch_sizes),
        "total_samples": total_samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed if elapsed else 0.0,
        "avg_dataloader_wait_seconds": (
            float(sum(wait_times) / len(wait_times)) if wait_times else None
        ),
        "avg_compute_seconds": (
            float(sum(compute_times) / len(compute_times)) if compute_times else None
        ),
        "max_dataloader_wait_seconds": max(wait_times) if wait_times else None,
        "max_compute_seconds": max(compute_times) if compute_times else None,
        "rss": human_bytes(process_rss()),
        "peak_rss": human_bytes(peak_rss),
        "x_feature_count": dataset.x_feature_count,
        "y_feature_count": dataset.y_feature_count,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "cuda_prefetch": use_cuda_prefetch,
        "cuda_prefetch_batches": args.cuda_prefetch_batches,
        "normalization": args.normalization,
        "valid_target_filter": args.valid_target_filter,
        "dataloader_timeout": args.dataloader_timeout,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else None,
        "dataloader_in_order": (
            args.dataloader_in_order if args.num_workers > 0 else None
        ),
        "worker_cache_size_mb": args.worker_cache_size_mb,
        "chunk_sampler": args.chunk_sampler,
        "sampling": args.sampling if args.chunk_sampler else None,
        "time_chunk_size": args.time_chunk_size if args.chunk_sampler else None,
        "effective_chunk_shapes": (
            getattr(loader.batch_sampler, "effective_chunk_shapes", None)
            if args.chunk_sampler
            else None
        ),
        "basin_chunk_size": args.basin_chunk_size if args.chunk_sampler else None,
        "sampler_index_bytes": (
            getattr(loader.batch_sampler, "index_storage_nbytes", None)
            if args.chunk_sampler
            else None
        ),
        "dataset_lookup_bytes": dataset.lookup_nbytes,
        "loader_batches": len(loader),
    }
    if len(batch_sizes) == 0:
        result["warning"] = (
            "No batches were produced. Check batch_size, time_chunk_size, "
            "basin_chunk_size and drop_last."
        )
    if device.type == "cuda":
        result["torch_cuda_peak_allocated"] = human_bytes(
            torch.cuda.max_memory_allocated(device)
        )
    if gpu_samples:
        result["avg_gpu_util_percent"] = sum(
            item["gpu_util_percent"] for item in gpu_samples
        ) / len(gpu_samples)
        result["last_gpu"] = gpu_samples[-1]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="camels_us")
    parser.add_argument("--source", default="cloud")
    parser.add_argument("--timeseries-uri", default=None)
    parser.add_argument("--attributes-uri", default=None)
    parser.add_argument("--use-hydrodataset-reader", action="store_true", default=True)
    parser.add_argument(
        "--no-hydrodataset-reader",
        dest="use_hydrodataset_reader",
        action="store_false",
    )
    parser.add_argument("--case-dir", default="/tmp/torchhydro_lazy_zarr_benchmark")
    parser.add_argument("--time-start", default="2000-01-01")
    parser.add_argument("--time-end", default="2002-12-31")
    parser.add_argument("--relevant-cols", default="precipitation")
    parser.add_argument("--target-cols", default="streamflow")
    parser.add_argument("--constant-cols", default="")
    parser.add_argument("--missing-policy", default="fill_zero")
    parser.add_argument(
        "--normalization",
        default="none",
        help=(
            "Normalization used by CloudZarrLazyDataset. The benchmark default "
            "is 'none' to avoid a full cloud stat scan before timing batches. "
            "Use 'standard' when measuring normalized training."
        ),
    )
    parser.add_argument("--no-cache-stats", action="store_true")
    parser.add_argument(
        "--valid-target-filter",
        action="store_true",
        help=(
            "Filter windows whose target is all NaN. Disabled by default for "
            "benchmark startup speed because it scans target data before the "
            "timed loop."
        ),
    )
    parser.add_argument("--max-basins", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hindcast-length", type=int, default=30)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--warmup-length", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--worker-cache-size-mb", type=float, default=64)
    parser.add_argument(
        "--dataloader-timeout",
        type=float,
        default=0.0,
        help="DataLoader worker timeout in seconds. Use 120 to fail fast on worker hangs.",
    )
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
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
    parser.add_argument(
        "--dataloader-in-order",
        action="store_true",
        help="Wait for worker results in submission order; lower throughput if one worker stalls.",
    )
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--cuda-prefetch", action="store_true")
    parser.add_argument("--cuda-prefetch-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.5)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--no-valid-target-filter",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
