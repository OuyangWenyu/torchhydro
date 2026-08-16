"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2025-01-14 20:06:11
LastEditors: Wenyu Ouyang
Description: Train a model for 3775 basins
FilePath: \HydroForecastEval\scripts\tl_lstm_camels_to_songliao.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from definitions import DATASET_DIR, PROJECT_DIR, RESULT_DIR
from scripts_consts import (
    VAR_T_CHOSEN_FROM_XU,
    VAR_C_CHOSEN_FROM_XU,
)

data_path = os.path.join(DATASET_DIR, "googleflood")
dataset = SelfMadeHydroDataset(data_path=data_path, time_unit=["1D"])
full_gage_id = dataset.read_site_info()["basin_id"].values.tolist()
# gage_id = sorted([basin for basin in full_gage_id if basin.startswith("songliao")])
gage_id = ["songliao_11601301"]

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
DEVICE = 0
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

trained_weight_path = os.path.join(
    RESULT_DIR,
    "train_google",
    "exp_multi_layer_lstm_hidden_256",
    "best_model.pth",
)


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        "tl_google",
        "songliao_11601301",
    )
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        project_dir=PROJECT_DIR,
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": data_path,
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "googleflood",
            },
        },
        ctx=[DEVICE],
        model_name="LinearMultiLayerLSTMModel",
        model_hyperparam={
            "linear_size": len(VAR_C_CHOSEN_FROM_XU) + len(VAR_T_CHOSEN_FROM_XU),
            "input_size": 46,
            "output_size": 1,
            "hidden_size": 256,
            "num_layers": 2,
            "dr": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=50,
        forecast_history=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval=1,
        var_t=VAR_T_CHOSEN_FROM_XU,
        var_c=VAR_C_CHOSEN_FROM_XU,
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_hourly",
                # "temperature_2m",
                # "surface_net_solar_radiation_hourly",
                "snowfall_hourly",
                # "surface_pressure",
            ],
            "pbm_norm": False,
        },
        continue_train=False,
        weight_path=trained_weight_path,
        # weight_path_add={
        #     "freeze_params": [
        #         "lstm.bias_hh_l0",
        #         "lstm.bias_ih_l0",
        #         "lstm.weight_hh_l0",
        #         "lstm.weight_ih_l0",
        #     ]
        # },
        train_epoch=50,
        save_epoch=1,
        train_period=["2010-01-01", "2020-12-31"],
        # train_period=["1985-10-01", "1986-11-30"],
        valid_period=["2021-01-01", "2023-12-31"],
        test_period=["2021-01-01", "2023-12-31"],
        loss_func="RMSESum",
        opt="Adam",
        opt_param={
            "lr": 0.0005,
        },
        lr_scheduler={
            "lr_factor": 0.95,
        },
        which_first_tensor="sequence",
        calc_metrics=True,
        metrics=["NSE", "RMSE"],
        early_stopping=True,
        patience=5,
        model_type="TransLearn",
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
