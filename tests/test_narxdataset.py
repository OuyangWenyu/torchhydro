
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
            "2018-09-30",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_valid": ["2018-10-01", "2019-09-30"],
        "t_range_test": [
            "2019-09-30",
            "2020-10-01"
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
        ],  # Add this line with the actual object IDs
        "t_range_train": ["1980-10-01", "2012-10-01"],  # Add this line with the actual start and end dates for training.
        "t_range_test": ["2012-10-01", "2014-10-01"],  # Add this line with the actual start and end dates for validation.
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
        "forecast_history": 0,
        "warmup_length": 0,
        "forecast_length": 1095,
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
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "PET",
                "tsd_pet_ou",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        "stat_dict_file": None,  # Added the missing configuration
    }
    is_tra_val_te = "train"
    dataset = StlDataset(data_cfgs, is_tra_val_te)
    y, trend, season, residuals, post_season, post_residuals = dataset._stl_decomposition()
    decomposition = pd.DataFrame({"streamflow": y, "trend": trend, "season": season, "residuals": residuals, "post_season": post_season,"post_residuals": post_residuals})
    decomposition.index.name = "time"
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\series_decomposition.csv"
    decomposition.to_csv(file_name, sep=" ")
    print(decomposition)
# PASSED                       [100%]
# time        streamflow       trend  ...  post_season  post_residuals
# 0           224.0  618.576884  ...  -314.775083      -79.801801
# 1           190.0  619.222341  ...  -300.990738     -128.231603
# 2           171.0  619.830675  ...  -278.736115     -170.094560
# 3           316.0  620.401173  ...  -261.521193      -42.879980
# 4           457.0  620.933207  ...  -253.225250       89.292043
# ...           ...         ...  ...          ...             ...
# 11675       110.0  527.724254  ...  -457.090793       39.366539
# 11676       110.0  527.068357  ...  -377.700677      -39.367680
# 11677       106.0  526.381652  ...  -263.285016     -157.096635
# 11678       157.0  525.664081  ...  -168.843955     -199.820126
# 11679       504.0  524.915743  ...  -127.166887      106.251144
#
# [11680 rows x 6 columns]

# PASSED                       [100%]
# time        streamflow  trend   ...  post_season  post_residuals
# 0           509.0  1657.540532  ...  -890.472233     -258.068299
# 1           518.0  1659.645543  ...  -866.767943     -274.877600
# 2           516.0  1661.781477  ...  -843.782457     -301.999020
# 3           620.0  1663.943871  ...  -824.940393     -219.003479
# 4           759.0  1666.128112  ...  -806.134351     -100.993761
# ...           ...          ...  ...          ...             ...
# 11675       146.0  1329.582108  ... -1221.126386       37.544278
# 11676       139.0  1330.872114  ... -1196.141768        4.269654
# 11677       131.0  1332.234708  ... -1153.255612      -47.979096
# 11678       127.0  1333.667480  ... -1101.229787     -105.437692
# 11679       137.0  1335.167812  ... -1044.596524     -153.571288
#
# [11680 rows x 6 columns]


def pick_leap_year(start_date, end_date):
    # start_date = self.t_s_dict["t_final_range"][0]
    # end_date = self.t_s_dict["t_final_range"][1]
    start = start_date.split("-")
    end = end_date.split("-")
    year_start = int(start[0])
    month_start = int(start[1])
    year_end = int(end[0])
    month_end = int(end[1])
    n = year_end - year_start
    if month_start > 2:
        year = list(range(year_start+1, year_end+1))
    else:
        year = list(range(year_start, year_end+1))
    leap_year = []
    month_day = "-02-29"
    for i in range(len(year)):
        remainder = year[i] % 4
        if remainder == 0:
            year_month_day = str(year[i]) + month_day
            leap_year.append(year_month_day)
    return leap_year

def test_pick_leap_year():
    start_date = "1980-10-01"
    end_date = "2014-10-01"
    leap_year = pick_leap_year(start_date, end_date)
    print(leap_year)
# PASSED                          [100%]
# ['1984-02-29', '1988-02-29', '1992-02-29', '1996-02-29', '2000-02-29', '2004-02-29', '2008-02-29', '2012-02-29']
