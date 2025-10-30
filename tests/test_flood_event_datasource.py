"""
Author: Wenyu Ouyang
Date: 2025-08-05 20:00:00
LastEditTime: 2025-08-05 10:20:21
LastEditors: Wenyu Ouyang
Description: Test module for FloodEventDatasource functionality
FilePath: \torchhydro\tests\test_flood_event_datasource.py
Copyright (c) 2025-2025 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import pandas as pd
import numpy as np

from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_flood_event_datasource_in_dict():
    """Test that FloodEventDatasource is properly registered in data_sources_dict"""
    assert "floodeventdatasource" in data_sources_dict
    from hydrodatasource.reader.floodevent import FloodEventDatasource

    assert data_sources_dict["floodeventdatasource"] == FloodEventDatasource


def test_flood_event_datasource_initialization(flood_event_datasource_args):
    """Test FloodEventDatasource initialization with configuration"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Test that we can create a DeepHydro instance with FloodEventDatasource
    deep_hydro = DeepHydro(config)

    # Verify that the data source is properly configured
    assert (
        deep_hydro.cfgs["data_cfgs"]["source_cfgs"]["source_name"]
        == "floodeventdatasource"
    )
    assert (
        "songliaorrevents"
        in deep_hydro.cfgs["data_cfgs"]["source_cfgs"]["other_settings"]["dataset_name"]
    )


def test_flood_event_datasource_properties(flood_event_datasource_args):
    """Test FloodEventDatasource specific properties and methods"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    deep_hydro = DeepHydro(config)

    # Test data source creation
    data_source_cfg = deep_hydro.cfgs["data_cfgs"]["source_cfgs"]
    source_name = data_source_cfg["source_name"]
    source_path = data_source_cfg["source_path"]
    other_settings = data_source_cfg.get("other_settings", {})

    # Create data source instance
    data_source = data_sources_dict[source_name](source_path, **other_settings)

    # Test FloodEventDatasource specific methods
    assert hasattr(data_source, "get_constants")
    assert hasattr(data_source, "extract_flood_events")
    assert hasattr(data_source, "load_1basin_flood_events")
    assert hasattr(data_source, "read_ts_xrdataset")

    # Test constants
    constants = data_source.get_constants()
    assert "net_rain_key" in constants
    assert "obs_flow_key" in constants
    assert "delta_t_hours" in constants
    assert "delta_t_seconds" in constants

    # Verify configured values
    assert constants["net_rain_key"] == "net_rain"
    assert constants["obs_flow_key"] == "inflow"
    assert constants["delta_t_hours"] == 3.0
    assert constants["delta_t_seconds"] == 3.0 * 3600


def test_flood_event_datasource_read_ts_xrdataset(
    monkeypatch, flood_event_datasource_args
):
    """Test FloodEventDatasource read_ts_xrdataset method"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Mock data structure similar to what FloodEventDatasource would return
    class MockXrData:
        pass

    mock_xr_data = MockXrData()

    def mock_read_ts(*args, **kwargs):
        return {"3h": mock_xr_data}

    # Use monkeypatch to replace the method
    monkeypatch.setattr(
        "hydrodatasource.reader.floodevent.FloodEventDatasource.read_ts_xrdataset",
        mock_read_ts,
    )

    deep_hydro = DeepHydro(config)
    data_source = deep_hydro.data_source

    # Test read_ts_xrdataset call
    result = data_source.read_ts_xrdataset(
        gage_id_lst=["songliao_21401550"],
        t_range=["2016-01-01", "2020-12-31"],
        var_lst=["net_rain", "inflow", "flood_event"],
    )

    # Verify the result structure
    assert "3h" in result


def test_flood_event_dataset_integration(flood_event_datasource_args):
    """Test integration with FloodEventDataset"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Ensure we're using the correct dataset class
    assert config["data_cfgs"]["dataset"] == "FloodEventDataset"

    # Create DeepHydro instance
    deep_hydro = DeepHydro(config)

    # Verify configuration consistency
    assert (
        deep_hydro.cfgs["data_cfgs"]["source_cfgs"]["source_name"]
        == "floodeventdatasource"
    )
    assert deep_hydro.cfgs["data_cfgs"]["dataset"] == "FloodEventDataset"

    # Test that FloodEventDataset can work with FloodEventDatasource
    # This tests the compatibility between the datasource and dataset
    assert hasattr(deep_hydro, "data_source")


def test_flood_event_loading(monkeypatch, flood_event_datasource_args):
    """Test flood event loading functionality"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Mock flood events data
    mock_events = [
        {
            "net_rain": np.array([1.0, 2.0, 3.0]),
            "inflow": np.array([0.5, 1.0, 1.5]),
            "filepath": "event_test.csv",
            "peak_obs": 1.5,
            "m_eff": 3,
            "n_specific": 3,
        }
    ]

    def mock_load_events(*args, **kwargs):
        return mock_events

    # Use monkeypatch to replace the method
    monkeypatch.setattr(
        "hydrodatasource.reader.floodevent.FloodEventDatasource.load_1basin_flood_events",
        mock_load_events,
    )

    deep_hydro = DeepHydro(config)
    data_source = deep_hydro.data_source

    # Test flood event loading
    events = data_source.load_1basin_flood_events(
        station_id="songliao_21401550", flow_unit="mm/3h", include_peak_obs=True
    )

    # Verify the returned structure
    assert events == mock_events
    assert len(events) == 1
    assert "net_rain" in events[0]
    assert "inflow" in events[0]


def test_enhanced_data_configuration(flood_event_datasource_args):
    """Test that enhanced data configuration is properly set"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    other_settings = config["data_cfgs"]["source_cfgs"]["other_settings"]

    # Verify enhanced data configuration
    assert other_settings["dataset_name"] == "songliaorrevents"
    assert other_settings["net_rain_key"] == "net_rain"
    assert other_settings["obs_flow_key"] == "inflow"
    assert other_settings["delta_t_hours"] == 3.0
    assert other_settings["time_unit"] == ["3h"]


def test_variable_configuration_compatibility(flood_event_datasource_args):
    """Test that variable configuration is compatible with FloodEventDatasource"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Test variable configuration
    assert "net_rain" in config["data_cfgs"]["relevant_cols"]
    assert "inflow" in config["data_cfgs"]["target_cols"]
    assert "flood_event" in config["data_cfgs"]["target_cols"]

    # Test time configuration
    assert config["data_cfgs"]["min_time_unit"] == "h"
    assert config["data_cfgs"]["min_time_interval"] == "3"


def test_loss_function_configuration(flood_event_datasource_args):
    """Test that FloodLoss is properly configured for flood events"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Verify FloodLoss configuration
    assert config["training_cfgs"]["criterion"] == "FloodLoss"
    assert config["training_cfgs"]["criterion_params"]["flood_weight"] == 2.0
    assert config["training_cfgs"]["criterion_params"]["flood_strategy"] == "weight"


def test_evaluator_configuration(flood_event_datasource_args):
    """Test that flood event evaluator is properly configured"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    # Verify evaluator configuration for flood events
    assert config["evaluation_cfgs"]["evaluator"]["eval_way"] == "floodevent"
    assert config["evaluation_cfgs"]["rolling"] == -1  # Use flood event sequences


@pytest.mark.parametrize("time_unit", ["3h", "1h", "6h"])
def test_different_time_units(flood_event_datasource_args, time_unit):
    """Test FloodEventDatasource with different time units"""
    config = default_config_file()

    # Modify time unit configuration
    flood_event_datasource_args.source_cfgs["other_settings"]["time_unit"] = [time_unit]
    update_cfg(config, flood_event_datasource_args)

    deep_hydro = DeepHydro(config)
    data_source = deep_hydro.data_source

    # Verify time unit configuration
    assert time_unit in data_source.time_unit


def test_data_source_description(flood_event_datasource_args):
    """Test that FloodEventDatasource has proper description"""
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)

    deep_hydro = DeepHydro(config)
    data_source = deep_hydro.data_source

    # Test that data source has expected attributes
    assert hasattr(data_source, "dataset_name")
    assert data_source.dataset_name == "songliaorrevents"

def test_train_evaluate(flood_event_datasource_args):
    config = default_config_file()
    update_cfg(config, flood_event_datasource_args)
    train_and_evaluate(config)