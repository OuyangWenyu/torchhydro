"""
Data path resolver for torchhydro.

Thin wrapper around hydrodatasource's resolve_data_path() which itself
wraps hydrodataset's resolver + injects hydrodatasource-specific registries.

This gives torchhydro a single entry point for resolving any dataset id
(public CAMELS datasets + hydrodatasource custom datasets) to an absolute URI.

Usage:
    from torchhydro.configs.data_resolver import resolve_dataset_uri

    uri = resolve_dataset_uri("camels_us")
    ds = CamelsUs(uri)
"""

from __future__ import annotations

from typing import Dict, Optional

from hydrodatasource.configs.data_resolver import (
    resolve_data_path,
    DatasetResolutionError,
)


def resolve_dataset_uri(
    dataset_id: str,
    *,
    source: str = "local",
    local_root: Optional[str] = None,
) -> str:
    """Resolve a dataset id to an absolute URI.

    Uses the storage.local.root from ~/hydro_setting.yml and the
    dataset registry (built-in + hydrodatasource custom datasets).

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g. 'camels_us', 'camels_aus', 'songliao_event').
    source : str
        Storage backend: 'local' or 'cloud'. Default 'local'.
    local_root : str, optional
        Override storage.local.root. When provided, skips reading settings.

    Returns
    -------
    str
        Absolute path (local) or S3 URI (cloud) to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If dataset_id is unknown or the resolved path does not exist.
    """
    return resolve_data_path(
        dataset_id,
        source=source,
        local_root=local_root,
    )


def normalize_source_cfgs(source_cfgs: Dict) -> Dict:
    """Normalize source_cfgs: resolve dataset_id to a concrete source_path.

    The new format is simply ``{"dataset_id": "camels_us"}``.
    An explicit ``source_path`` override is also supported:
    ``{"dataset_id": "camels_us", "source_path": "/custom/path"}``.

    Parameters
    ----------
    source_cfgs : dict
        Source configuration dict. Must contain 'dataset_id' or explicit 'source_path'.

    Returns
    -------
    dict
        Dict with 'source_name' and 'source_path' keys, ready for data_sources_dict lookup.
    """
    # Explicit path override — use directly (skips resolver)
    if "source_path" in source_cfgs:
        # source_name is the key for data_sources_dict lookup; it takes priority
        # over dataset_id when both are present with an explicit source_path
        source_name = source_cfgs.get("source_name") or source_cfgs.get("dataset_id")
        if not source_name:
            raise DatasetResolutionError(
                "source_cfgs with explicit 'source_path' must also have 'dataset_id' or 'source_name'"
            )
        return {"source_name": source_name, "source_path": source_cfgs["source_path"]}

    # New clean path: resolve via dataset_id
    dataset_id = source_cfgs.get("dataset_id")
    if dataset_id:
        resolved_path = resolve_dataset_uri(dataset_id)
        return {"source_name": dataset_id, "source_path": resolved_path}

    raise DatasetResolutionError(
        "source_cfgs must contain 'dataset_id' (e.g. {'dataset_id': 'camels_us'}). "
        "Known hydrodataset ids: camels_us, camels_aus, camels_br, camels_gb, "
        "camels_cl, camels_col, camelsh, caravan, grdc_caravan, camelsh_kr."
    )
