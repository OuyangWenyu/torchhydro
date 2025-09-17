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
from hydrodataset.camelshourly import CamelsHourly
#from hydrodataset.camels_aef import CamelsAef

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("/Users/cylenlc/data/camels_hourly")
camels = CamelsHourly(camels_dir)
# gage_id = camels.read_site_info()["gauge_id"].values.tolist()
gage_id = ['01022500', '01031500']

gage_id = sorted([x for x in gage_id])

assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

length = 7
dim = 128
scaler = "DapengScaler"
# scaler = "StandardScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        #project_dir="D:\\torchhydro\\text2attr",
        sub=project_name,
        source_cfgs={
            "source_name": "camels_hourly",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[0],
        model_name="MTSLSTM",
        model_hyperparam={
            "input_sizes": 28,  # 只给一个数即可（会扩展成两频：周/日）
            "hidden_sizes": [64, 64],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": False,

            # 关键：自动构造低频 + 切片传递
            "auto_build_lowfreq": True,
            "build_factor": 7,  # 7天一周
            "agg_reduce": "mean",  # 或 "sum"；也可给 per_feature_aggs
            # "per_feature_aggs": ["sum","mean",...],  # 长度==28（可选）
            "truncate_incomplete": True,
            "slice_transfer": True,
            "slice_use_ceil": True,
        },
        model_loader={"load_way": "best"},
        # gage_id=gage_id[5000:5009],
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=384,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval=1,
        var_t=[
                "convective_fraction",
                "longwave_radiation",
                "potential_energy",
                "potential_evaporation",
                "pressure",
                "shortwave_radiation",
                "specific_humidity",
                "temperature",
                "total_precipitation",
                "wind_u",
                "wind_v",
            ],
        scaler_params={
            "prcp_norm_cols": [
                # "streamflow_input",
                "streamflow",
            ],
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },
        var_c=[
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
        # scaler="DapengScaler",
        scaler=scaler,
        var_out=["qobs_mm_per_hour"],
        dataset="CamelsHourlyDataset",
        train_epoch=20,
        save_epoch=1,
        train_period=["1990-01-01", "1991-12-31"],
        valid_period=["1990-01-01", "1991-12-31"],
        test_period=["1990-01-01", "1991-12-31"],
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
        rolling=0,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=2,
        model_type="Normal",
        #valid_batch_mode="train",
        # valid_batch_mode="test",
        #evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            #"eval_way": "1pace",
            # "pace_idx": -1,
            #"pace_idx": -1,
        #},
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
