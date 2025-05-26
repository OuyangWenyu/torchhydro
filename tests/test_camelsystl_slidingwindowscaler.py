"""
mainly test the SlidingWindowScaler, 
"""

import os

from torchhydro import SETTING
from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_scalers import SklearnScalers, TorchhydroScalers
from torchhydro.datasets.scalers import DapengScaler, SlidingWindowScaler
from tests.test_camelsystl_stl_mi import Ystl


def test_SlidingWindowScaler():
    temp_test_path = "test_camels\slidingwindowscaler_camelsus"
    os.makedirs(temp_test_path, exist_ok=True)

    data_cfgs =  {
        "source_cfgs": {
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01022500",
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
        ],
        # camels-us time_range: ["1980-01-01", "2014-12-31"]
        "t_range_train": ["2011-10-01", "2012-09-30"],
        "t_range_valid": ["2012-10-01", "2013-09-30"],
        "t_range_test": ["2013-10-01", "2014-09-30"],
        "relevant_cols": [
            # NOTE: prcp must be the first variable
            "prcp",
            "PET"
            # "dayl",
            # "srad",
            # "swe",
            # "tmax",
            # "tmin",
            # "vp",
        ],
        "target_cols": [
            "streamflow",
        ],
        "constant_cols": [
            "elev_mean",
            "slope_mean",
            "area_gages2",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "geol_1st_class",
            "geol_2nd_class",
            "geol_porostiy",
            "geol_permeability",
        ],
        "forecast_history": 0,
        "warmup_length": 0,
        "forecast_length": 30,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "SlidingWindowScaler",  # Add the scaler configuration here
        "stat_dict_file": None,  # Added the missing configuration
        # dataset="StlDataset",
        # "dataset": "StreamflowDataset",
        # scaler="StandardScaler",
        # "scaler": "SlidingWindowScaler",
        # scaler="DapengScaler",
        # scaler="MinMaxScaler",
        # scaler_params={
        #     "prcp_norm_cols": [
        #         "streamflow",
        #     ],
        #     "gamma_norm_cols": [
        #         "prcp",
        #         "pr",
        #         "total_precipitation",
        #         "potential_evaporation",
        #         "ET",
        #         "PET",
        #         "ET_sum",
        #         "ssm",
        #     ],
        #     "pbm_norm": True,
        # },
        # b_decompose=True,
        # which_first_tensor="sequence",
    }
    is_tra_val_te = "train"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    print(dataset)


def test_data():
    x = Ystl().pet
    xx = x[:30]
    print(xx)
# [1.2, 1.3, 0.9, 0.55, 0.85, 1.15, 0.9, 0.85, 0.7, 0.7, 0.8, 0.95, 1.05, 0.75, 0.6, 0.55, 1.1, 1.75, 1.3, 0.8, 1.05, 1.15, 1.25, 1.85, 1.8, 1.0, 0.75, 0.45, 0.35, 0.95]

def test_init():
    sws = SlidingWindowScaler()
    print(sws)
# <torchhydro.datasets.scalers.SlidingWindowScaler object at 0x7f37c1ac9e80>

def test_cal_stat():
    x = Ystl().prcp
    sws = SlidingWindowScaler()
