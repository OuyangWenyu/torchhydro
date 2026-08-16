import warnings

import pandas as pd
import pytest

from torchhydro.configs.config import default_config_file
from torchhydro.datasets.data_utils import validate_split_ranges
from torchhydro.datasets.cloud_zarr_lazy_dataset import CloudZarrLazyDataset


@pytest.fixture()
def tiny_zarr_stores(tmp_path):
    pytest.importorskip("zarr")
    from tests.test_cloud_zarr_lazy_dataset import _write_tiny_zarr

    return _write_tiny_zarr(
        tmp_path,
        "split",
        ["01013500"],
        include_temperature=False,
    )


def _ranges(train, valid, test):
    return {
        "t_range_train": train,
        "t_range_valid": valid,
        "t_range_test": test,
    }


def test_train_test_oos_overlap_is_rejected():
    with pytest.raises(ValueError, match="train.*test"):
        validate_split_ranges(
            _ranges(["2000-01-01", "2000-01-10"], None, ["2000-01-10", "2000-01-20"])
        )


def test_train_valid_oos_overlap_is_rejected():
    with pytest.raises(ValueError, match="train.*valid"):
        validate_split_ranges(
            _ranges(
                ["2000-01-01", "2000-01-10"],
                ["2000-01-10", "2000-01-20"],
                ["2000-01-21", "2000-01-30"],
            )
        )


def test_valid_test_oos_overlap_is_rejected():
    with pytest.raises(ValueError, match="valid.*test"):
        validate_split_ranges(
            _ranges(
                ["2000-01-01", "2000-01-10"],
                ["2000-01-11", "2000-01-20"],
                ["2000-01-20", "2000-01-30"],
            )
        )


def test_touching_inclusive_boundaries_are_detected():
    with pytest.raises(ValueError, match="train.*test"):
        validate_split_ranges(
            _ranges(["2000-01-01", "2000-01-10"], None, ["2000-01-10", "2000-01-20"])
        )


def test_non_overlapping_oos_splits_are_accepted():
    result = validate_split_ranges(
        _ranges(
            ["2000-01-01", "2000-01-10"],
            ["2000-01-11", "2000-01-20"],
            ["2000-01-21", "2000-01-30"],
        )
    )
    assert result["evaluation_semantics"] == "out_of_sample"
    assert result["overlaps"] == ()


def test_explicit_in_sample_evaluation_is_allowed():
    cfg = _ranges(
        ["2000-01-01", "2000-01-10"],
        None,
        ["2000-01-01", "2000-01-10"],
    )
    cfg["allow_split_overlap"] = True
    cfg["evaluation_cfgs"] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_split_ranges(cfg)
    assert result["evaluation_semantics"] == "in_sample"
    assert result["overlaps"] == (("train", "test"),)
    assert any("in-sample" in str(item.message) for item in caught)
    assert cfg["evaluation_cfgs"]["split_evaluation_semantics"] == "in_sample"


def test_default_config_disables_split_overlap():
    cfg = default_config_file()
    assert cfg["data_cfgs"]["allow_split_overlap"] is False


def test_cloud_manifest_time_range_is_source_range_not_target_split(
    tmp_path, tiny_zarr_stores
):
    from tests.test_cloud_zarr_lazy_dataset import _base_cfg

    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["object_ids"] = ["01013500"]
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["t_range_train"] = ["2000-01-01", "2000-01-05"]
    cfg["data_cfgs"]["t_range_valid"] = ["2000-01-06", "2000-01-07"]
    cfg["data_cfgs"]["t_range_test"] = ["2000-01-08", "2000-01-10"]
    cfg["data_cfgs"]["lazy_zarr"]["datasets"] = [
        {
            "dataset_id": "camels_us",
            "timeseries_uri": str(ts_uri),
            "attributes_uri": str(attr_uri),
            "basins": ["01013500"],
            "time_range": "all",
            "relevant_cols": {"precipitation": "pcp_mm"},
            "target_cols": {"streamflow": "q_cms_obs"},
            "constant_cols": {"area": "area_km2"},
        }
    ]
    train = CloudZarrLazyDataset(cfg, "train")
    valid = CloudZarrLazyDataset(cfg, "valid")
    test = CloudZarrLazyDataset(cfg, "test")
    try:
        assert train.source_states[0].times[0] == pd.Timestamp("2000-01-01")
        assert valid.source_states[0].times[0] == pd.Timestamp("2000-01-01")
        assert valid.source_states[0].target_times[0] == pd.Timestamp("2000-01-06")
        assert test.source_states[0].target_times[0] == pd.Timestamp("2000-01-08")
    finally:
        train.close()
        valid.close()
        test.close()


def test_test_history_can_precede_target_split(tmp_path, tiny_zarr_stores):
    from tests.test_cloud_zarr_lazy_dataset import _base_cfg

    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["lazy_zarr"]["normalization"] = "none"
    cfg["data_cfgs"]["t_range_train"] = ["2000-01-01", "2000-01-04"]
    cfg["data_cfgs"]["t_range_valid"] = None
    cfg["data_cfgs"]["t_range_test"] = ["2000-01-06", "2000-01-10"]
    cfg["training_cfgs"]["warmup_length"] = 2
    cfg["data_cfgs"]["lazy_zarr"]["datasets"] = [
        {
            "dataset_id": "camels_us",
            "timeseries_uri": str(ts_uri),
            "attributes_uri": str(attr_uri),
            "basins": ["01013500"],
            "time_range": "all",
            "relevant_cols": {"precipitation": "pcp_mm"},
            "target_cols": {"streamflow": "q_cms_obs"},
            "constant_cols": {"area": "area_km2"},
        }
    ]
    dataset = CloudZarrLazyDataset(cfg, "test")
    try:
        assert dataset.source_states[0].times[0] == pd.Timestamp("2000-01-01")
        assert dataset.source_states[0].target_times[0] == pd.Timestamp("2000-01-06")
        assert dataset.lookup_table[0][1] >= 5
    finally:
        dataset.close()


def test_scaler_signature_changes_when_train_period_changes(
    tmp_path, tiny_zarr_stores
):
    from tests.test_cloud_zarr_lazy_dataset import _base_cfg

    ts_uri, attr_uri = tiny_zarr_stores
    cfg = _base_cfg(tmp_path, ts_uri, attr_uri)
    cfg["data_cfgs"]["object_ids"] = ["01013500"]
    cfg["data_cfgs"]["t_range_train"] = ["2000-01-01", "2000-01-05"]
    cfg["data_cfgs"]["t_range_valid"] = ["2000-01-06", "2000-01-10"]
    cfg["data_cfgs"]["t_range_test"] = ["2000-01-06", "2000-01-10"]
    train = CloudZarrLazyDataset(cfg, "train")
    train.close()

    changed_cfg = {
        **cfg,
        "data_cfgs": {
            **cfg["data_cfgs"],
            "t_range_train": ["2000-01-01", "2000-01-04"],
        },
    }
    with pytest.raises(ValueError, match="do not match.*train_period"):
        CloudZarrLazyDataset(changed_cfg, "valid")
