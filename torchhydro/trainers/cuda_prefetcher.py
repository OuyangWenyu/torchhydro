"""CUDA batch prefetch utilities."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move tensors in a batch to a device."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    return value


def record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    """Recursively mark tensors as used by the current CUDA stream."""
    if torch.is_tensor(value):
        value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_stream(item, stream)


class CudaPrefetcher:
    """Overlap CPU-to-GPU batch transfer with GPU compute.

    The wrapped loader must yield CPU tensors, ideally from pinned memory.  Each
    next batch is moved on a side stream while the caller trains on the current
    batch.
    """

    def __init__(
        self,
        loader,
        device: torch.device,
        non_blocking: bool = True,
        prefetch_batches: int = 2,
        performance_monitor=None,
    ):
        if device.type != "cuda":
            raise ValueError("CudaPrefetcher requires a CUDA device.")
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.stream = torch.cuda.Stream(device=device)
        self.performance_monitor = performance_monitor

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Create DataLoader workers on the caller's main thread. Forking worker
        # processes from the producer thread is unsafe on Linux.
        iterator = iter(self.loader)
        output: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop = threading.Event()
        sentinel = object()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    output.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                with torch.cuda.device(self.device):
                    while not stop.is_set():
                        fetch_started = time.perf_counter()
                        try:
                            cpu_batch = next(iterator)
                        except StopIteration:
                            break
                        if self.performance_monitor is not None:
                            self.performance_monitor.record_loader_fetch(
                                time.perf_counter() - fetch_started
                            )
                        if stop.is_set():
                            break
                        with torch.cuda.stream(self.stream):
                            events = (
                                self.performance_monitor.cuda_event_pair()
                                if self.performance_monitor is not None
                                else None
                            )
                            if events is not None:
                                events[0].record(self.stream)
                            cuda_batch = move_to_device(
                                cpu_batch,
                                self.device,
                                non_blocking=self.non_blocking,
                            )
                            ready = torch.cuda.Event()
                            ready.record(self.stream)
                            if events is not None:
                                events[1].record(self.stream)
                                self.performance_monitor.record_transfer_events(events)
                        if not put((cuda_batch, ready, None)):
                            return
            except BaseException as exc:
                put((None, None, exc))
            finally:
                put(sentinel)

        producer = threading.Thread(
            target=produce,
            name="torchhydro-cuda-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = output.get()
                if item is sentinel:
                    break
                batch, ready, error = item
                if error is not None:
                    raise error
                current_stream = torch.cuda.current_stream(self.device)
                current_stream.wait_event(ready)
                record_stream(batch, current_stream)
                yield batch
        finally:
            stop.set()
            producer.join(timeout=0.2)
