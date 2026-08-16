"""
Data source entry point for torchhydro.

torchhydro is data-agnostic — all data source resolution and instantiation
is delegated to hydrodatasource's registry.  Use ``open_dataset`` to get
an instantiated reader from a dataset identifier.

Usage::

    from torchhydro.configs.data_resolver import open_dataset

    ds = open_dataset("camels_us")
    ds = open_dataset("selfmadehydrodataset", time_unit=["1D"])

For explicit path resolution without instantiation::

    from torchhydro.configs.data_resolver import resolve_data_path

    uri = resolve_data_path("camels_us")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from hydrodataset.configs.settings import get_default_source, load_settings
from hydrodatasource.configs.data_resolver import (
    DatasetResolutionError,
    ResolverContext,
    open_dataset as _hds_open_dataset,
    resolve_data_path as _hds_resolve_data_path,
)


_IGNORED_SOURCE_CFG_KEYS = {
    "dataset_id",
    "source_name",
    "source_path",
    "source_paths",
    "path",
    "local_path",
    "root",
    # legacy reader keys replaced by the unified uri= interface
    "dataset_name",
    "data_path",
    "data_folder",
}


def _default_source_from_settings() -> str:
    """Return storage.default_source from ~/hydro_setting.yml."""
    return get_default_source(load_settings())


def _resolve_source(source: Optional[str]) -> str:
    """Use hydro_setting.yml when source is not explicitly provided."""
    return source or _default_source_from_settings()


def normalize_source_cfgs(
    source_cfgs: Dict[str, Any],
) -> tuple[str, Optional[str], Dict[str, Any]]:
    """Normalize TorchHydro source_cfgs to resolver inputs.

    Supported formats:

    New format::

        {"dataset_id": "camels_us", "other_settings": {"source": "cloud"}}

    Legacy format::

        {"source_name": "camels_us", "source_path": "/old/path"}

    ``source_path`` and other local path keys are intentionally ignored here.
    Dataset locations are resolved from ``~/hydro_setting.yml`` by the
    hydrodataset/hydrodatasource resolver.
    """
    source_cfgs = source_cfgs or {}
    dataset_id = source_cfgs.get("dataset_id") or source_cfgs.get("source_name")
    if dataset_id is None:
        raise DatasetResolutionError(
            "source_cfgs must contain 'dataset_id' or legacy 'source_name' "
            "(e.g. {'dataset_id': 'camels_us'})."
        )

    other_settings = dict(source_cfgs.get("other_settings") or {})
    source = other_settings.pop("source", None) or source_cfgs.get("source")

    # Keep reader-specific options such as download=False or region='US', but
    # never forward legacy path fields to readers. Paths are resolved centrally
    # from hydro_setting.yml.
    reader_kwargs = {
        key: value
        for key, value in other_settings.items()
        if key not in _IGNORED_SOURCE_CFG_KEYS
    }
    return str(dataset_id), source, reader_kwargs


def resolve_data_path(
    dataset_id: str,
    *,
    source: Optional[str] = None,
    ctx: Optional[ResolverContext] = None,
) -> str:
    """Resolve a dataset path using hydro_setting.yml by default.

    Unlike hydrodatasource's wrapper, ``source=None`` here means "read
    storage.default_source from ~/hydro_setting.yml" instead of defaulting to
    local. This prevents cloud runs from validating local paths.
    """
    return _hds_resolve_data_path(dataset_id, source=_resolve_source(source), ctx=ctx)


def open_dataset(
    dataset_id: str,
    *,
    source: Optional[str] = None,
    ctx: Optional[ResolverContext] = None,
    **reader_kwargs: Any,
):
    """Open a dataset using hydro_setting.yml by default."""
    return _hds_open_dataset(
        dataset_id,
        source=_resolve_source(source),
        ctx=ctx,
        **reader_kwargs,
    )


def open_dataset_from_source_cfgs(
    source_cfgs: Dict[str, Any],
    **extra_reader_kwargs: Any,
):
    """Open a dataset from TorchHydro source_cfgs.

    This is the preferred entry point for TorchHydro datasets because it
    supports both new and legacy config formats while centralizing path
    resolution in ``hydro_setting.yml``.
    """
    dataset_id, source, reader_kwargs = normalize_source_cfgs(source_cfgs)
    reader_kwargs.update(extra_reader_kwargs)
    try:
        return open_dataset(dataset_id, source=source, **reader_kwargs)
    except DatasetResolutionError as e:
        # Only fall back for a genuinely unknown dataset id.  Storage/path
        # errors (root not configured, s3 bucket missing, ...) must propagate.
        if "Unknown dataset id" not in str(e):
            raise
        # torchhydro-specific sources (self-made datasets, reservoirs, ...) are
        # not in the hydrodatasource resolver registry; fall back to the
        # torchhydro data_sources_dict registry.
        return _open_from_data_sources_dict(source_cfgs, dataset_id, reader_kwargs)


def _open_from_data_sources_dict(
    source_cfgs: Dict[str, Any],
    dataset_id: str,
    reader_kwargs: Dict[str, Any],
):
    from torchhydro.datasets.data_sources import data_sources_dict

    if dataset_id not in data_sources_dict:
        raise DatasetResolutionError(
            f"Unknown dataset id {dataset_id!r}. Neither the hydrodataset/"
            "hydrodatasource resolver nor torchhydro's data_sources_dict "
            "recognises it."
        ) from None
    constructor = data_sources_dict[dataset_id]
    # self-made sources locate their data via the legacy source_path/path key
    path = source_cfgs.get("source_path") or source_cfgs.get("path")
    if path is not None:
        return constructor(path, **reader_kwargs)
    return constructor(**reader_kwargs)


def resolve_dataset(dataset_id: str):
    """Return a callable that opens *dataset_id* via the resolver.

    The returned factory accepts the legacy ``data_sources_dict[name](path)``
    call convention: positional path arguments are intentionally ignored
    because dataset locations are resolved centrally from ``hydro_setting.yml``.
    """

    def _factory(*args: Any, **kwargs: Any):
        return open_dataset(dataset_id, **kwargs)

    return _factory
