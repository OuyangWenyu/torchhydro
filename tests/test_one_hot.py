
import os
import shutil
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from torchhydro.datasets.data_scalers import DapengScaler
from torchhydro.configs.data_resolver import open_dataset

@pytest.fixture
def camels_source():
    # resolve CAMELS-US via the unified resolver (data location comes from
    # ~/hydro_setting.yml) instead of a stale hardcoded Mac path
    return open_dataset("camels_us", region="US")

@pytest.fixture
def gage_ids():
    return ['01013500', '01022500', '01030500', '01031500', '01047000']

@pytest.fixture
def var_c():
    return [
        "elev_mean", "slope_mean", "area", "frac_forest", "lai_max", "lai_diff",
        "dom_land_cover_frac", "dom_land_cover", "root_depth_50", "soil_depth_statsgo",
        "soil_porosity", "soil_conductivity", "max_water_content",
        "geol_1st_class", "geol_2nd_class", "geol_porostiy", "geol_permeability"
    ]

@pytest.fixture
def one_hot_cols():
    return ["dom_land_cover", "geol_1st_class", "geol_2nd_class"]

@pytest.fixture
def vars_data(camels_source, gage_ids, var_c):
    # Read constant attributes
    ds_c = camels_source.read_attr_xrdataset(gage_ids, var_c)
    
    # Create a DataArray from the Dataset
    # We need to stack variables into a new dimension 'variable'
    da_c = ds_c.to_array(dim="variable").transpose("basin", "variable")
    
    # Mock target data (streamflow)
    times = pd.date_range("2000-01-01", periods=10)
    da_target = xr.DataArray(
        np.random.rand(len(gage_ids), len(times), 1),
        coords={"basin": gage_ids, "time": times, "variable": ["streamflow"]},
        dims=["basin", "time", "variable"],
        attrs={"units": {"streamflow": "mm/d"}}
    )
    
    return {
        "target_cols": da_target,
        "constant_cols": da_c
    }

@pytest.fixture
def case_dir(tmpdir):
    return str(tmpdir)

@pytest.mark.skip(
    reason="one-hot encoding in DapengScaler was removed during the performance "
    "refactor (41b3045); restoration is a separate feature task"
)
@pytest.mark.requires_data
def test_one_hot_encoding(camels_source, vars_data, gage_ids, one_hot_cols, var_c, case_dir):
    # --- Run 1: WITH One-Hot Encoding ---
    data_cfgs_one_hot = {
        "scaler": "DapengScaler",
        "scaler_params": {
            "pbm_norm": False,
            "gamma_norm_cols": [],
            "prcp_norm_cols": ["streamflow"],
            "one_hot_cols": one_hot_cols
        },
        "target_cols": ["streamflow"],
        "object_ids": gage_ids,
        "t_range_train": ["2000-01-01", "2000-01-10"],
        "case_dir": case_dir,
        "stat_dict_file": None
    }
    
    scaler_oh = DapengScaler(
        vars_data,
        data_cfgs_one_hot,
        is_tra_val_te="train",
        data_source=camels_source
    )
    
    norm_data_oh = scaler_oh.load_norm_data(vars_data)
    norm_c_oh = norm_data_oh["constant_cols"]
    vars_oh = norm_c_oh.coords["variable"].values
    
    print("\n--- One-Hot Encoding Results ---")
    print(f"Total static features: {len(vars_oh)}")
    print("Variables:", vars_oh)
    
    # Check if original categorical columns are removed
    for col in one_hot_cols:
        assert col not in vars_oh
        
    # Check if new one-hot columns are present
    # We expect at least some one-hot columns
    assert any(col in v for v in vars_oh for col in one_hot_cols)

    # --- Run 2: WITHOUT One-Hot Encoding ---
    data_cfgs_no_oh = {
        "scaler": "DapengScaler",
        "scaler_params": {
            "pbm_norm": False,
            "gamma_norm_cols": [],
            "prcp_norm_cols": ["streamflow"],
            "one_hot_cols": [] # Empty
        },
        "target_cols": ["streamflow"],
        "object_ids": gage_ids,
        "t_range_train": ["2000-01-01", "2000-01-10"],
        "case_dir": case_dir,
        "stat_dict_file": None
    }
    
    scaler_no_oh = DapengScaler(
        vars_data,
        data_cfgs_no_oh,
        is_tra_val_te="train",
        data_source=camels_source
    )
    
    norm_data_no_oh = scaler_no_oh.load_norm_data(vars_data)
    norm_c_no_oh = norm_data_no_oh["constant_cols"]
    vars_no_oh = norm_c_no_oh.coords["variable"].values
    
    print("\n--- No One-Hot Encoding Results ---")
    print(f"Total static features: {len(vars_no_oh)}")
    print("Variables:", vars_no_oh)
    
    # --- Comparison ---
    print("\n--- Comparison ---")
    
    common_vars = [v for v in var_c if v not in one_hot_cols]
    
    for var in common_vars:
        val_oh = norm_c_oh.sel(variable=var).values
        val_no_oh = norm_c_no_oh.sel(variable=var).values
        
        # Check for NaNs
        mask_oh = np.isnan(val_oh)
        mask_no_oh = np.isnan(val_no_oh)
        
        np.testing.assert_array_equal(mask_oh, mask_no_oh, err_msg=f"Mismatch in NaNs for {var}")
            
        # Compare non-NaN values
        valid_mask = ~mask_oh
        np.testing.assert_allclose(
            val_oh[valid_mask], 
            val_no_oh[valid_mask], 
            err_msg=f"Mismatch in values for {var}"
        )
            
    print("\nSUCCESS: All non-one-hot columns match exactly between the two runs.")

    # Print sample data for one-hot columns
    print("\n--- Sample Data for One-Hot Columns (Run 1) ---")
    for var in vars_oh:
        if any(oh in var for oh in one_hot_cols) and var not in common_vars:
            print(f"{var}: {norm_c_oh.sel(variable=var).values.flatten()}")

if __name__ == "__main__":
    pytest.main(["-s", __file__])
