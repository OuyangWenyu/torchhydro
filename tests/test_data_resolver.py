"""Tests for torchhydro's data resolver entry points."""

import pytest

from torchhydro.configs.data_resolver import open_dataset_from_source_cfgs
import torchhydro.datasets.data_sources as ds_mod


@pytest.mark.requires_data
def test_open_standard_dataset_via_resolver():
    """camels_us is in the hydrodataset registry -> resolved, not raised."""
    ds = open_dataset_from_source_cfgs({"dataset_id": "camels_us"})
    assert ds is not None


def test_open_selfmade_source_falls_back_to_data_sources_dict(mocker):
    """self-made sources live in torchhydro's data_sources_dict, not the
    hydrodatasource resolver registry; open_dataset_from_source_cfgs must fall
    back instead of raising DatasetResolutionError.  Legacy keys (dataset_name)
    are dropped, not forwarded to the reader."""
    fake_source = mocker.Mock(return_value=object())
    mocker.patch.dict(ds_mod.data_sources_dict, {"selfmadehydrodataset": fake_source})
    result = open_dataset_from_source_cfgs(
        {
            "source_name": "selfmadehydrodataset",
            "source_path": "/some/path",
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "FDSources",
            },
        }
    )
    fake_source.assert_called_once_with("/some/path", time_unit=["1D"])
    assert result is fake_source.return_value
