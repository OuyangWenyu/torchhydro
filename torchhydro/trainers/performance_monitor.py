"""Low-overhead performance monitoring for TorchHydro training."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch


def _batch_size(batch: Any) -> int:
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return int(batch[0].shape[0])
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    return int(getattr(batch, "num_graphs", 0) or 0)


class TrainingPerformanceMonitor:
    """Collect epoch timing, throughput, CUDA, GPU and sampler metrics."""

    def __init__(
        self,
        output_dir: str | Path,
        device: torch.device,
        gpu_sample_interval: float = 1.0,
        gpu_idle_threshold: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.gpu_sample_interval = max(0.2, float(gpu_sample_interval))
        self.gpu_idle_threshold = float(gpu_idle_threshold)
        self._stop = threading.Event()
        self._gpu_samples: list[dict[str, float]] = []
        self._thread: threading.Thread | None = None
        self._compute_events: deque[tuple[torch.cuda.Event, torch.cuda.Event]] = deque()
        self._transfer_events: deque[tuple[torch.cuda.Event, torch.cuda.Event]] = (
            deque()
        )
        self._lock = threading.Lock()

    def start_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.started = time.perf_counter()
        self.data_wait_seconds = 0.0
        self.compute_host_seconds = 0.0
        self.loader_fetch_seconds = 0.0
        self.batch_count = 0
        self.sample_count = 0
        self.batch_size_min: int | None = None
        self.batch_size_max = 0
        self.compute_cuda_ms = 0.0
        self.transfer_cuda_ms = 0.0
        self._gpu_samples = []
        self._compute_events.clear()
        self._transfer_events.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._sample_gpu,
                name="torchhydro-training-gpu-monitor",
                daemon=True,
            )
            self._thread.start()

    def record_data_wait(self, seconds: float) -> None:
        self.data_wait_seconds += float(seconds)

    def record_loader_fetch(self, seconds: float) -> None:
        with self._lock:
            self.loader_fetch_seconds += float(seconds)

    def record_batch(self, batch: Any, host_seconds: float) -> None:
        batch_size = _batch_size(batch)
        self.batch_count += 1
        self.sample_count += batch_size
        self.batch_size_min = (
            batch_size
            if self.batch_size_min is None
            else min(self.batch_size_min, batch_size)
        )
        self.batch_size_max = max(self.batch_size_max, batch_size)
        self.compute_host_seconds += float(host_seconds)
        self._drain_cuda_events(complete=False)

    def cuda_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

    def record_compute_events(self, events) -> None:
        if events is not None:
            self._compute_events.append(events)

    def record_transfer_events(self, events) -> None:
        if events is not None:
            with self._lock:
                self._transfer_events.append(events)

    def finish_epoch(self, sampler=None, dataset=None) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._drain_cuda_events(complete=True)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        elapsed = time.perf_counter() - self.started
        gpu_utils = [sample["gpu_util_percent"] for sample in self._gpu_samples]
        gpu_memory = [sample["gpu_memory_used_mib"] for sample in self._gpu_samples]
        result: dict[str, Any] = {
            "epoch": self.epoch,
            "epoch_seconds": elapsed,
            "batches": self.batch_count,
            "samples": self.sample_count,
            "samples_per_second": self.sample_count / elapsed if elapsed else 0.0,
            "batch_size_avg": (
                self.sample_count / self.batch_count if self.batch_count else None
            ),
            "batch_size_min": self.batch_size_min,
            "batch_size_max": self.batch_size_max if self.batch_count else None,
            "avg_dataloader_wait_seconds": (
                self.data_wait_seconds / self.batch_count if self.batch_count else None
            ),
            "total_dataloader_wait_seconds": self.data_wait_seconds,
            "loader_fetch_seconds": self.loader_fetch_seconds,
            "avg_compute_host_seconds": (
                self.compute_host_seconds / self.batch_count
                if self.batch_count
                else None
            ),
            "total_compute_host_seconds": self.compute_host_seconds,
            "total_compute_cuda_seconds": self.compute_cuda_ms / 1000.0,
            "total_host_to_device_cuda_seconds": self.transfer_cuda_ms / 1000.0,
            "gpu_sample_count": len(gpu_utils),
            "gpu_util_avg_percent": (
                sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
            ),
            "gpu_util_min_percent": min(gpu_utils) if gpu_utils else None,
            "gpu_util_max_percent": max(gpu_utils) if gpu_utils else None,
            "gpu_idle_percent": (
                100.0
                * sum(value <= self.gpu_idle_threshold for value in gpu_utils)
                / len(gpu_utils)
                if gpu_utils
                else None
            ),
            "gpu_idle_threshold_percent": self.gpu_idle_threshold,
            "gpu_memory_used_max_mib": max(gpu_memory) if gpu_memory else None,
            "torch_cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda"
                else None
            ),
            "per_dataset_samples": self._sampler_counts(sampler, dataset),
        }
        path = self.output_dir / f"performance_epoch_{self.epoch:04d}.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def _drain_cuda_events(self, complete: bool) -> None:
        for queue_, attr in (
            (self._compute_events, "compute_cuda_ms"),
            (self._transfer_events, "transfer_cuda_ms"),
        ):
            with self._lock:
                while queue_ and (complete or queue_[0][1].query()):
                    start, end = queue_.popleft()
                    setattr(self, attr, getattr(self, attr) + start.elapsed_time(end))

    def _sample_gpu(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        f"--id={self.device.index or 0}",
                    ],
                    text=True,
                    timeout=2,
                ).strip()
                util, used, total = output.splitlines()[0].split(",")
                self._gpu_samples.append(
                    {
                        "gpu_util_percent": float(util),
                        "gpu_memory_used_mib": float(used),
                        "gpu_memory_total_mib": float(total),
                    }
                )
            except Exception:
                pass
            self._stop.wait(self.gpu_sample_interval)

    @staticmethod
    def _sampler_counts(sampler, dataset) -> dict[str, int] | None:
        counts = getattr(sampler, "epoch_sample_counts", None)
        if counts is None or dataset is None:
            return None
        return {
            dataset.sources[int(source)].dataset_id: int(count)
            for source, count in counts.items()
        }
