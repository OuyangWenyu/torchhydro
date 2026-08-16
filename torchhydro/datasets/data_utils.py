"""
Author: Wenyu Ouyang
Date: 2023-09-21 15:37:58
LastEditTime: 2025-07-13 15:46:09
LastEditors: Wenyu Ouyang
Description: Some basic funtions for dealing with data
FilePath: \torchhydro\torchhydro\datasets\data_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from collections import OrderedDict
import numpy as np
import xarray as xr
import pint_xarray  # noqa: F401
import warnings
import pandas as pd


def warn_if_nan(dataarray, max_display=5, nan_mode="any", data_name=""):
    """
    Issue a warning if the dataarray contains any NaN values and display their locations.

    Parameters
    -----------
    dataarray: xr.DataArray
        Input dataarray to check for NaN values.
    max_display: int
        Maximum number of NaN locations to display in the warning.
    nan_mode: str
        Mode of NaN checking:
        'any' means if any NaNs exist return True, if all values are NaNs raise ValueError
        'all' means if all values are NaNs return True
    data_name: str
        Name of the dataarray to be displayed in the warning message.
    """
    if dataarray is None:
        raise ValueError("The dataarray is None!")
    if nan_mode not in ["any", "all"]:
        raise ValueError("nan_mode must be 'any' or 'all'")

    # Build the NaN mask once. The previous implementation evaluated
    # ``np.isnan`` repeatedly and materialized every NaN coordinate with
    # ``argwhere``, even though detailed warning output is disabled. That is
    # expensive for basin x time x variable arrays and can allocate a large
    # temporary integer array.
    nan_mask = np.isnan(dataarray.values)
    if not np.any(nan_mask):
        return False

    if np.all(nan_mask):
        if nan_mode == "any":
            raise ValueError("The dataarray contains only NaN values!")
        else:
            return True

    # Warn with a bounded list of NaN locations (max_display caps the output
    # so this stays cheap even for basin x time x variable arrays).
    nan_indices = np.argwhere(nan_mask)
    total_nan = nan_indices.shape[0]
    display_count = min(max_display, total_nan)
    indices_str = ", ".join(str(tuple(idx)) for idx in nan_indices[:display_count])
    warnings.warn(
        f"The {data_name} dataarray contains {total_nan} NaN values! "
        f"Here are the indices of the first {display_count} NaNs: {indices_str}",
        UserWarning,
    )
    return True


def unify_streamflow_unit(ds: xr.Dataset, area=None, inverse=False):
    """Unify the unit of xr_dataset to be mm/day in a basin or inverse

    Parameters
    ----------
    ds: xarray dataset
        _description_
    area:
        area of each basin

    Returns
    -------
    _type_
        _description_
    """
    # use pint to convert unit
    if not inverse:
        target_unit = "mm/d"
        q = ds.pint.quantify()
        a = area.pint.quantify()
        r = q[list(q.keys())[0]] / a[list(a.keys())[0]]
        result = r.pint.to(target_unit).to_dataset(name=list(q.keys())[0])
    else:
        target_unit = "m^3/s"
        r = ds.pint.quantify()
        a = area.pint.quantify()
        q = r[list(r.keys())[0]] * a[list(a.keys())[0]]
        # q = q.pint.quantify()
        result = q.pint.to(target_unit).to_dataset(name=list(r.keys())[0])
    # dequantify to get normal xr_dataset
    return result.pint.dequantify()


def wrap_t_s_dict(data_cfgs: dict, is_tra_val_te: str) -> OrderedDict:
    """
    Basins and periods

    Parameters
    ----------
    data_cfgs
        configs for reading from data source
    is_tra_val_te
        train, valid or test

    Returns
    -------
    OrderedDict
        OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    """
    basins_id = data_cfgs["object_ids"]
    if type(basins_id) is str and basins_id == "ALL":
        raise ValueError("Please specify the basins_id in configs!")
    if any(x >= y for x, y in zip(basins_id, basins_id[1:])):
        # raise a warning if the basins_id is not sorted
        warnings.warn("The basins_id is not sorted!")
    if f"t_range_{is_tra_val_te}" in data_cfgs:
        t_range_list = data_cfgs[f"t_range_{is_tra_val_te}"]
    else:
        raise KeyError(f"Error! The mode {is_tra_val_te} was not found. Please add it.")
    return OrderedDict(sites_id=basins_id, t_final_range=t_range_list)


def validate_split_ranges(config: dict) -> dict:
    """Validate train/validation/test target intervals.

    ``t_range_*`` values describe target timestamps under TorchHydro's
    left-closed right-open convention: adjacent periods that share a boundary
    timestamp (train ends 2001-10-01, valid starts 2001-10-01) are **not** an
    overlap.  Only a strictly nested period (one split starting before the
    previous split's end) is flagged.

    A strict overlap is reported as an advisory ``UserWarning`` and the result
    is labelled ``in_sample`` — it never blocks a run, because overlap commonly
    comes from reusing train data as warmup/hindcast input history, which is
    legitimate.  Set ``data_cfgs.allow_split_overlap`` to true to suppress the
    advisory warning when the in-sample intent is explicit.

    The function is deliberately small and side-effect free apart from adding
    the derived evaluation metadata to the supplied normalized config when it
    has an ``evaluation_cfgs`` section.
    """

    data_cfgs = config.get("data_cfgs", config)
    if not isinstance(data_cfgs, dict):
        raise TypeError("split configuration must be a mapping")

    ranges = {}
    for split in ("train", "valid", "test"):
        value = data_cfgs.get(f"t_range_{split}")
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"t_range_{split} must be [start, end] or None; got {value!r}"
            )
        start = pd.Timestamp(value[0])
        end = pd.Timestamp(value[1])
        if start > end:
            raise ValueError(f"t_range_{split} start must not exceed end")
        ranges[split] = (start, end)

    overlaps = []
    split_order = ("train", "valid", "test")
    for left_idx, left in enumerate(split_order):
        if left not in ranges:
            continue
        for right in split_order[left_idx + 1 :]:
            if right not in ranges:
                continue
            left_start, left_end = ranges[left]
            right_start, right_end = ranges[right]
            # Left-closed right-open semantics: only a strictly nested period
            # (one starting before the previous split's end) counts as overlap.
            if max(left_start, right_start) < min(left_end, right_end):
                overlaps.append((left, right))

    allow_overlap = bool(data_cfgs.get("allow_split_overlap", False))
    semantics = "in_sample" if overlaps else "out_of_sample"
    if overlaps and not allow_overlap:
        overlap_text = ", ".join(f"{left}/{right}" for left, right in overlaps)
        warnings.warn(
            f"Overlapping target split ranges detected for {overlap_text}. "
            "Results will be treated as in-sample and must not be interpreted "
            "as out-of-sample generalization performance.",
            UserWarning,
            stacklevel=2,
        )

    result = {
        "overlaps": tuple(overlaps),
        "evaluation_semantics": semantics,
        "target_ranges": {
            split: (start.isoformat(), end.isoformat())
            for split, (start, end) in ranges.items()
        },
    }
    if "evaluation_cfgs" in config:
        config["evaluation_cfgs"]["split_evaluation_semantics"] = semantics
        config["evaluation_cfgs"]["overlapping_splits"] = [
            list(pair) for pair in overlaps
        ]
    return result


def _trans_norm(
    x: xr.DataArray,
    var_lst: list,
    stat_dict: dict,
    log_norm_cols: list = None,
    maxmin_norm_cols: list = None,
    to_norm: bool = True,
    **kwargs,
) -> np.array:
    """
    Normalization or inverse normalization

    There are two normalization formulas:

    .. math:: normalized_x = (x - mean) / std

    and

     .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

    and:
        .. math:: normalized_x = (x-min) / (max-min)

     The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

    Parameters
    ----------
    x
        data to be normalized or denormalized
    var_lst
        the type of variables
    stat_dict
        statistics of all variables
    log_norm_cols
        which cols use the second norm method
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    if x is None:
        return None
    if log_norm_cols is None:
        log_norm_cols = []
    if maxmin_norm_cols is None:
        maxmin_norm_cols = []
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = xr.full_like(x, np.nan)
    for item in var_lst:
        stat = stat_dict[item]
        if kwargs.get("debug", False):
            arr = x.sel(variable=item).to_numpy()
            print(
                f"[DEBUG] {item}: mean={stat[2]}, std={stat[3]}, "
                f"arr_sample={arr[:10]}"
            )
        if to_norm:
            if item in maxmin_norm_cols:
                out.loc[dict(variable=item)] = (x.sel(variable=item) - stat[0]) / (stat[1] - stat[0])
            elif item in log_norm_cols:
                out.loc[dict(variable=item)] = (np.log10(np.sqrt(np.abs(x.sel(variable=item))) + 0.1) - stat[2]) / stat[3]
            else:
                out.loc[dict(variable=item)] = (x.sel(variable=item) - stat[2]) / stat[3]
        else:
            if item in maxmin_norm_cols:
                out.loc[dict(variable=item)] = x.sel(variable=item) * (stat[1] - stat[0]) + stat[0]
            elif item in log_norm_cols:
                out.loc[dict(variable=item)] = (np.power(10, x.sel(variable=item) * stat[3] + stat[2]) - 0.1) ** 2
            else:
                out.loc[dict(variable=item)] = x.sel(variable=item) * stat[3] + stat[2]
    if to_norm:
        out.attrs = {}
    else:
        if "recover_units" in kwargs.keys() and kwargs["recover_units"] is not None:
            recover_units = kwargs["recover_units"]
            for item in var_lst:
                out.attrs["units"][item] = recover_units[item]
    return out
    #     stat = stat_dict[item]
    #     if to_norm:
    #         out.loc[dict(variable=item)] = (
    #             (np.log10(np.sqrt(np.abs(x.sel(variable=item))) + 0.1) - stat[2])
    #             / stat[3]
    #             if item in log_norm_cols
    #             else (x.sel(variable=item) - stat[2]) / stat[3]
    #         )
    #     elif item in log_norm_cols:
    #         out.loc[dict(variable=item)] = (
    #             np.power(10, x.sel(variable=item) * stat[3] + stat[2]) - 0.1
    #         ) ** 2
    #     else:
    #         out.loc[dict(variable=item)] = x.sel(variable=item) * stat[3] + stat[2]
    # if to_norm:
    #     # after normalization, all units are dimensionless
    #     out.attrs = {}
    # # after denormalization, recover units
    # else:
    #     if "recover_units" in kwargs.keys() and kwargs["recover_units"] is not None:
    #         recover_units = kwargs["recover_units"]
    #         for item in var_lst:
    #             out.attrs["units"][item] = recover_units[item]
    # return out


def _prcp_norm(x: np.array, mean_prep: np.array, to_norm: bool) -> np.array:
    """
    Normalize or denormalize data with mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{precipitation}

    Parameters
    ----------
    x
        data to be normalized or denormalized
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    prep = mean_prep
    while prep.ndim < x.ndim:
        prep = np.expand_dims(prep, axis=-1)
    return x / prep if to_norm else x * prep



def set_unit_to_var(ds):
    """returned xa.Dataset need has units for each variable -- xr.DataArray
    or the dataset cannot be saved to netCDF file

    Parameters
    ----------
    ds : xr.Dataset
        the dataset with units as attributes

    Returns
    -------
    ds : xr.Dataset
        unit attrs are for each variable dataarray
    """
    units_dict = ds.attrs["units"]
    for var_name, units in units_dict.items():
        if var_name in ds:
            ds[var_name].attrs["units"] = units
    if "units" in ds.attrs:
        del ds.attrs["units"]
    return ds
