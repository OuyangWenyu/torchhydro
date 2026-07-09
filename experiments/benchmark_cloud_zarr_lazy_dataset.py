"""Benchmark CloudZarrLazyDataset data loading and light GPU training.

This script is intended for the server with hydrodataset cloud settings.  It
does not eager-load the full Zarr store; batches are read through
CloudZarrLazyDataset.__getitems__.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    cfg["data_cfgs"].update(
        {
            "dataset": "CloudZarrLazyDataset",
            "case_dir": str(case_dir),
            "source_cfgs": {
                "dataset_id": args.dataset_id,
                "other_settings": {"source": args.source},
            },
            "object_ids": None,
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
                "normalization": "standard",
                "stat_cache_dir": str(case_dir / "lazy_zarr_stat_cache"),
                "skip_all_nan_target_windows": not args.no_valid_target_filter,
                "missing_policy": args.missing_policy,
            },
        }
    )
    if args.max_basins is not None:
        # Use the first N basins after the hydrodataset reader resolves object ids.
        # The dataset itself will validate those basin ids against the Zarr store.
        from torchhydro.configs.data_resolver import open_dataset

        reader = open_dataset(args.dataset_id, source=args.source)
        cfg["data_cfgs"]["object_ids"] = [
            str(value) for value in reader.read_object_ids()[: args.max_basins]
        ]
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
        {"rolling": 0, "hrwin": None, "frwin": None, "evaluator": {"recover_mode": "bybasins"}}
    )
    return cfg


def make_loader(dataset: CloudZarrLazyDataset, args: argparse.Namespace) -> DataLoader:
    dataset._batch_level_dataloader_enabled = True
    kwargs = {}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
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
            collate_fn=prebatched_collate_fn,
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=prebatched_collate_fn,
        **kwargs,
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_cfg(args)
    dataset = CloudZarrLazyDataset(cfg, "train")
    loader = make_loader(dataset, args)
    device = torch.device(args.device)
    use_cuda_prefetch = bool(args.cuda_prefetch and device.type == "cuda")
    model = None
    optimizer = None
    if not args.no_train:
        model = torch.nn.LSTM(
            input_size=dataset.x_feature_count,
            hidden_size=args.hidden_size,
            batch_first=True,
        ).to(device)
        head = torch.nn.Linear(args.hidden_size, dataset.y_feature_count).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    else:
        head = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wait_times = []
    compute_times = []
    batch_sizes = []
    gpu_samples = []
    peak_rss = process_rss()
    iterable = (
        CudaPrefetcher(loader, device, non_blocking=args.pin_memory)
        if use_cuda_prefetch
        else loader
    )
    iterator = iter(iterable)
    started = time.perf_counter()
    for step in range(args.steps):
        wait_start = time.perf_counter()
        try:
            x, y = next(iterator)
        except StopIteration:
            break
        wait_times.append(time.perf_counter() - wait_start)
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
        gpu = query_gpu()
        if gpu:
            gpu_samples.append(gpu)
        current_rss = process_rss()
        if current_rss is not None:
            peak_rss = current_rss if peak_rss is None else max(peak_rss, current_rss)

    elapsed = time.perf_counter() - started
    total_samples = sum(batch_sizes)
    result = {
        "dataset_id": args.dataset_id,
        "dataset_length": len(dataset),
        "steps": len(batch_sizes),
        "total_samples": total_samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed if elapsed else 0.0,
        "avg_dataloader_wait_seconds": float(sum(wait_times) / len(wait_times)) if wait_times else None,
        "avg_compute_seconds": float(sum(compute_times) / len(compute_times)) if compute_times else None,
        "max_dataloader_wait_seconds": max(wait_times) if wait_times else None,
        "max_compute_seconds": max(compute_times) if compute_times else None,
        "rss": human_bytes(process_rss()),
        "peak_rss": human_bytes(peak_rss),
        "x_feature_count": dataset.x_feature_count,
        "y_feature_count": dataset.y_feature_count,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "cuda_prefetch": use_cuda_prefetch,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else None,
        "chunk_sampler": args.chunk_sampler,
        "sampling": args.sampling if args.chunk_sampler else None,
        "time_chunk_size": args.time_chunk_size if args.chunk_sampler else None,
        "effective_time_chunk_size": (
            getattr(loader.batch_sampler, "time_chunk_size", None)
            if args.chunk_sampler
            else None
        ),
        "basin_chunk_size": args.basin_chunk_size if args.chunk_sampler else None,
        "loader_batches": len(loader),
    }
    if len(batch_sizes) == 0:
        result["warning"] = (
            "No batches were produced. Check batch_size, time_chunk_size, "
            "basin_chunk_size and drop_last."
        )
    if device.type == "cuda":
        result["torch_cuda_peak_allocated"] = human_bytes(torch.cuda.max_memory_allocated(device))
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
    parser.add_argument("--case-dir", default="/tmp/torchhydro_lazy_zarr_benchmark")
    parser.add_argument("--time-start", default="2000-01-01")
    parser.add_argument("--time-end", default="2002-12-31")
    parser.add_argument("--relevant-cols", default="precipitation")
    parser.add_argument("--target-cols", default="streamflow")
    parser.add_argument("--constant-cols", default="")
    parser.add_argument("--missing-policy", default="fill_zero")
    parser.add_argument("--max-basins", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hindcast-length", type=int, default=30)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--warmup-length", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
    parser.add_argument("--no-chunk-sampler", dest="chunk_sampler", action="store_false")
    parser.add_argument("--sampling", default="weighted_by_sample")
    parser.add_argument("--time-chunk-size", type=int, default=None)
    parser.add_argument("--basin-chunk-size", type=int, default=1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--cuda-prefetch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-valid-target-filter", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
