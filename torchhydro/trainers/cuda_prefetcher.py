"""CUDA batch prefetch utilities."""

from __future__ import annotations

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
    ):
        if device.type != "cuda":
            raise ValueError("CudaPrefetcher requires a CUDA device.")
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        next_batch = None
        exhausted = False

        def preload():
            nonlocal next_batch, exhausted
            try:
                batch = next(iterator)
            except StopIteration:
                exhausted = True
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = move_to_device(
                    batch,
                    self.device,
                    non_blocking=self.non_blocking,
                )

        preload()
        while not exhausted:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.stream)
            batch = next_batch
            preload()
            record_stream(batch, current_stream)
            yield batch
