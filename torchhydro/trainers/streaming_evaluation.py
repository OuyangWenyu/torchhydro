"""Streaming metrics for lazy window datasets.

The evaluator deliberately keeps no prediction/observation arrays.  It uses
the final target step of each sliding window, which gives one forecast per
window start and avoids counting overlapping target windows repeatedly.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchhydro.trainers.train_utils import model_infer


@dataclass
class _NSEAccumulator:
    """Sufficient statistics for NSE, tracked independently per target."""

    count: np.ndarray
    observed_sum: np.ndarray
    predicted_sum: np.ndarray
    observed_square_sum: np.ndarray
    squared_error_sum: np.ndarray

    @classmethod
    def create(cls, target_count: int) -> _NSEAccumulator:
        return cls(
            count=np.zeros(target_count, dtype=np.int64),
            observed_sum=np.zeros(target_count, dtype=np.float64),
            predicted_sum=np.zeros(target_count, dtype=np.float64),
            observed_square_sum=np.zeros(target_count, dtype=np.float64),
            squared_error_sum=np.zeros(target_count, dtype=np.float64),
        )

    def update(self, observed: np.ndarray, predicted: np.ndarray) -> None:
        valid = np.isfinite(observed) & np.isfinite(predicted)
        for target_idx in range(observed.shape[-1]):
            obs = observed[:, target_idx][valid[:, target_idx]]
            pred = predicted[:, target_idx][valid[:, target_idx]]
            if not obs.size:
                continue
            self.count[target_idx] += int(obs.size)
            self.observed_sum[target_idx] += float(obs.sum(dtype=np.float64))
            self.predicted_sum[target_idx] += float(pred.sum(dtype=np.float64))
            self.observed_square_sum[target_idx] += float(
                np.square(obs, dtype=np.float64).sum(dtype=np.float64)
            )
            self.squared_error_sum[target_idx] += float(
                np.square(obs - pred, dtype=np.float64).sum(dtype=np.float64)
            )

    def as_dict(self, target_names: list[str]) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for target_idx, name in enumerate(target_names):
            count = int(self.count[target_idx])
            denominator = (
                self.observed_square_sum[target_idx]
                - self.observed_sum[target_idx] ** 2 / count
                if count
                else 0.0
            )
            nse = (
                1.0 - self.squared_error_sum[target_idx] / denominator
                if denominator > 0.0
                else None
            )
            result[name] = {
                "count": count,
                "observed_sum": float(self.observed_sum[target_idx]),
                "predicted_sum": float(self.predicted_sum[target_idx]),
                "observed_square_sum": float(self.observed_square_sum[target_idx]),
                "squared_error_sum": float(self.squared_error_sum[target_idx]),
                "nse": float(nse) if nse is not None else None,
            }
        return result


def _denormalize_targets(dataset, values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    """Convert normalized target values to physical units per source dataset."""
    output = np.asarray(values, dtype=np.float64).copy()
    for source_idx in np.unique(source_indices):
        rows = np.flatnonzero(source_indices == source_idx)
        state = dataset.source_states[int(source_idx)]
        stats = state.stats or {"enabled": False}
        if not stats.get("enabled"):
            continue
        for target_idx, canonical in enumerate(dataset.target_schema):
            stat = stats.get("target_cols", {}).get(canonical)
            if stat is None:
                continue
            output[rows, target_idx] = (
                output[rows, target_idx] * float(stat["std"])
                + float(stat["mean"])
            )
    return output


def _write_metrics_csv(result: dict[str, Any], path: Path) -> None:
    """Write nested global/dataset/basin metrics as one analysis-friendly CSV."""
    fieldnames = [
        "scope",
        "dataset_id",
        "basin_id",
        "target",
        "count",
        "observed_sum",
        "predicted_sum",
        "observed_square_sum",
        "squared_error_sum",
        "nse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        def write_rows(scope: str, metrics, dataset_id: str = "", basin_id: str = ""):
            for target, values in metrics.items():
                writer.writerow(
                    {
                        "scope": scope,
                        "dataset_id": dataset_id,
                        "basin_id": basin_id,
                        "target": target,
                        **values,
                    }
                )

        write_rows("global", result["global"])
        for dataset_id, metrics in result["per_dataset"].items():
            write_rows("dataset", metrics, dataset_id=dataset_id)
        for dataset_id, basins in result["per_basin"].items():
            for basin_id, metrics in basins.items():
                write_rows(
                    "basin",
                    metrics,
                    dataset_id=dataset_id,
                    basin_id=basin_id,
                )


def evaluate_cloud_zarr_lazy_nse(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str | None,
    *,
    batch_size: int,
    output_path: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a lazy test dataset with constant-memory NSE accumulators.

    The dataset is read in sequential index batches so each test window is
    visited once. Only the final output/target step is used because it maps to
    a unique forecast time for a given window start.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    target_names = list(dataset.target_schema)
    overall = _NSEAccumulator.create(len(target_names))
    by_source = {
        source.dataset_id: _NSEAccumulator.create(len(target_names))
        for source in dataset.sources
    }
    by_basin: dict[str, dict[str, _NSEAccumulator]] = {
        source.dataset_id: {} for source in dataset.sources
    }
    started = time.perf_counter()
    batch_count = 0
    window_count = 0
    model.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and batch_count >= max_batches:
                break
            stop = min(start + batch_size, len(dataset))
            item_indices = np.arange(start, stop, dtype=np.int64)
            x, y = dataset._load_batch(item_indices.tolist())
            targets, predictions = model_infer(
                True,
                torch_device,
                model,
                (x, y),
                dataset.training_cfgs.get("variable_length_cfgs"),
            )
            if not isinstance(predictions, torch.Tensor):
                raise TypeError("Streaming NSE evaluation requires tensor model outputs.")
            if targets is None:
                raise ValueError("Streaming NSE evaluation requires target values.")
            if predictions.ndim != 3 or targets.ndim != 3:
                raise ValueError(
                    "Streaming NSE evaluation expects [batch, time, target] tensors."
                )

            observed = dataset.load_final_target_values(item_indices.tolist())
            predicted = predictions[:, -1, :].detach().cpu().numpy()
            source_indices = np.asarray(dataset._lookup_sources[item_indices], dtype=np.int64)
            basin_indices = np.asarray(dataset._lookup_basins[item_indices], dtype=np.int64)
            predicted = _denormalize_targets(dataset, predicted, source_indices)
            overall.update(observed, predicted)
            for source_idx in np.unique(source_indices):
                rows = np.flatnonzero(source_indices == source_idx)
                source = dataset.sources[int(source_idx)]
                dataset_id = source.dataset_id
                by_source[dataset_id].update(observed[rows], predicted[rows])
                for basin_idx in np.unique(basin_indices[rows]):
                    basin_rows = rows[basin_indices[rows] == basin_idx]
                    basin_id = source.basins[int(basin_idx)]
                    accumulator = by_basin[dataset_id].setdefault(
                        basin_id,
                        _NSEAccumulator.create(len(target_names)),
                    )
                    accumulator.update(observed[basin_rows], predicted[basin_rows])

            batch_count += 1
            window_count += len(item_indices)

    result: dict[str, Any] = {
        "split": getattr(dataset, "is_tra_val_te", "test"),
        "prediction_policy": "last_step_per_window",
        "batches": batch_count,
        "windows": window_count,
        "elapsed_seconds": time.perf_counter() - started,
        "global": overall.as_dict(target_names),
        "per_dataset": {
            dataset_id: accumulator.as_dict(target_names)
            for dataset_id, accumulator in by_source.items()
        },
        "per_basin": {
            dataset_id: {
                basin_id: accumulator.as_dict(target_names)
                for basin_id, accumulator in basins.items()
            }
            for dataset_id, basins in by_basin.items()
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_metrics_csv(result, path.with_suffix(".csv"))
    return result
