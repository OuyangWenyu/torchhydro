"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2025-01-10 10:11:30
LastEditors: Wenyu Ouyang
Description: Train a model for 3775 basins
FilePath: /HydroForecastEval/scripts/train_googlefloodhub_camels_671basins_ear5land_less_param_new_rolling_large_horizon.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pandas as pd

import sys
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("D:\stream_data", "camels", "camels_us")
camels = Camels(camels_dir)
gage_id = ["01013500"]
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

length = 7
dim = 128
scaler = "DapengScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels_test", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        train_mode=False,
        stat_dict_file=r"D:\work\torchhydro\experiments\results\camels\simplelstm_DapengScaler_128_0.4_ens_01\dapengscaler_stat.json",
        project_dir=r"D:\work\torchhydro\experiments",
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[1],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 71,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={
            "load_way": "pth",
            "pth_path": f"D:\work\\torchhydro\experiments\\results\camels\simplelstm_DapengScaler_128_0.4_ens_{hru_delete}/best_model.pth",
        },
        gage_id=gage_id,
        batch_size=384,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval=1,
        var_t=["prcp", "dayl", "srad", "tmax", "tmin", "vp", "PET"],
        scaler_params={
            "prcp_norm_cols": [
                # "streamflow_input",
                "streamflow",
            ],
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },
        var_c=[
            "p_mean",
            "pet_mean",
            "p_seasonality",
            "frac_snow",
            "aridity",
            "high_prec_freq",
            "high_prec_dur",
            "low_prec_freq",
            "low_prec_dur",
            "elev_mean",
            "slope_mean",
            "area_gages2",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "gvf_max",
            "gvf_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "soil_depth_pelletier",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "sand_frac",
            "silt_frac",
            "clay_frac",
            "geol_1st_class",
            "glim_1st_class_frac",
            "geol_2nd_class",
            "glim_2nd_class_frac",
            "carbonate_rocks_frac",
            "geol_porostiy",
            "geol_permeability",
        ],
        # scaler="DapengScaler",
        scaler=scaler,
        var_out=["streamflow"],
        dataset="AlphaEarthDataset",
        train_epoch=10,
        save_epoch=1,
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        # train_period=["1980-01-01", "1981-12-31"],
        # valid_period=["2010-01-01", "2013-12-31"],
        # test_period=["2014-01-01", "2015-12-31"],
        loss_func="RMSESum",
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0],
        #     "device": [2],
        #     "item_weight": [1],
        # },
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        # lr_scheduler={
        #     epoch: (
        #         0.5
        #         if 1 <= epoch <= 5
        #         else (
        #             0.2
        #             if 6 <= epoch <= 10
        #             else (
        #                 0.1
        #                 if 11 <= epoch <= 15
        #                 else 0.05 if 16 <= epoch <= 20 else 0.02
        #             )
        #         )
        #     )
        #     for epoch in range(1, 21)
        # },
        which_first_tensor="sequence",
        # calc_metrics=True,
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        rolling=1,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=2,
        model_type="Normal",
        valid_batch_mode="train",
        # valid_batch_mode="test",
        evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            "eval_way": "1pace",
            # "pace_idx": -1,
            "pace_idx": -1,
        },
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
