
import pytest
import os
import pandas as pd
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.datasets.narxdataset import NarxDataset, StlDataset


def test_NarxDataset():
    """

    Returns
    -------

    """
    temp_test_path = "results/test_camels/narxdataset_camelsfr"
    os.makedirs(temp_test_path, exist_ok=True)
    data_cfgs = {
        "source_cfgs": {
            "source_name": "camels_fr",
            "source_path": "camels/camels_fr",
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "A550061001", "A369011001", "A284020001", "A330010001",
        ],  # Add this line with the actual object IDs
        "t_range_train": [
            "2017-10-01",
            "2018-10-01",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_test": [
            "2018-10-01",
            "2019-10-01"
        ],  # Add this line with the actual start and end dates for validation.
        "relevant_cols": [
            # List the relevant column names here.
            "tsd_prec",
            "tsd_pet_ou",
            # ... other relevant columns ...
        ],
        "target_cols": [
            # List the target column names here.
            "streamflow",
            # ... other target columns ...
        ],
        "constant_cols": [
            # List the constant column names here.
            # "top_altitude_mean",
            # "top_slo_mean",
            # "sta_area_snap",
            # "top_drainage_density",
            # ... other constant columns ...
        ],
        "forecast_history": 10,
        "warmup_length": 10,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        # "scaler": "StandardScaler",  # Add the scaler configuration here
        "scaler": "DapengScaler",
        "scaler_params": {
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "tsd_prec",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "tsd_pet_ou",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        "stat_dict_file": None,  # Added the missing configuration
        "b_nestedness": True,
    }
    is_tra_val_te = "train"
    dataset = NarxDataset(data_cfgs, is_tra_val_te)
    print(dataset)


# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# Backend tkagg is interactive backend. Turning interactive mode on.
# collected 1 item

# test_narxdataset.py Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 548.94it/s]
# <torchhydro.datasets.narxdataset.NarxDataset object at 0x7fd4cc0efcb0>

def test_stl_decomposition():
    temp_test_path = r"D:\torchhydro\tests\results\test_camels\stldataset_camelsus"
    os.makedirs(temp_test_path, exist_ok=True)
    data_cfgs = {
        "source_cfgs": {
            "source_name": "camels_us",
            "source_path": "camels\camels_us",
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            # # "01022500",
            # # "01030500",
            # # "01031500",
            # # "01047000",
            # # "01052500",
            # # "01054200",
            # # "01055000",
            # # "01057000",
            # # "01073000",
            # # "01078000",
            # # "01118300",
            # # "01121000",
            # # "01123000",
            # # "01134500",
            # # "01137500",
            # # "01139000",
            # # "01139800",
            # # "01142500",
            # # "01144000",
            # "02092500",  # 02108000 -> 02092500
            # "02108000",
        ],  # Add this line with the actual object IDs
        "t_range_train": [
            "1985-10-01",
            "1988-10-01",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_test": [
            "1988-10-01",
            "1989-10-01"
        ],  # Add this line with the actual start and end dates for validation.
        "relevant_cols": [
            # List the relevant column names here.
            "prcp",
            "PET",
            # ... other relevant columns ...
        ],
        "target_cols": [
            # List the target column names here.
            "streamflow",
            # ... other target columns ...
        ],
        "constant_cols": [
            # "elev_mean",
            # "slope_mean",
            # "area_gages2",
            # "frac_forest",
            # "lai_max",
            # "lai_diff",
            # "dom_land_cover_frac",
            # "dom_land_cover",
            # "root_depth_50",
            # "soil_depth_statsgo",
            # "soil_porosity",
            # "soil_conductivity",
            # "max_water_content",
            # "geol_1st_class",
            # "geol_2nd_class",
            # "geol_porostiy",
            # "geol_permeability",
        ],
        "forecast_history": 365,
        "warmup_length": 0,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        # "scaler": "StandardScaler",  # Add the scaler configuration here
        "scaler": "DapengScaler",
        "scaler_params": {
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "tsd_prec",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "tsd_pet_ou",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        "stat_dict_file": None,  # Added the missing configuration
        "b_nestedness": True,
    }
    is_tra_val_te = "train"
    dataset = StlDataset(data_cfgs, is_tra_val_te)
    y, trend, season, residuals, post_season, post_residuals = dataset._stl_decomposition()
    decomposition = pd.DataFrame(
        {"streamflow": y, "trend": trend, "season": season, "residuals": residuals, "post_season": post_season,
         "post_residuals": post_residuals})
    decomposition.index.name = "time"
    file_name = r"D:\torchhydro\tests\results\test_camels\narxdataset_camelsfr_stl\series_decomposition.csv"
    decomposition.to_csv(file_name, sep=" ")
    print(decomposition)
