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

from hydrodatasource.configs.data_resolver import (  # noqa: F401
    DatasetResolutionError,
    ResolverContext,
    open_dataset,
    resolve_data_path,
)
