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

import sys
from torchhydro import SETTING
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
# from definitions import DATASET_DIR, PROJECT_DIR


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("events_train_3h", "regional_weighted_mse_loss")
    config_data = default_config_file()
    # data_dir = r"C:\data"
    # source_path = os.path.join(data_dir, "songliaorrevent")
    DEVICE = -1

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
            "other_settings": {
                #"time_unit": ["3h"],
                "dataset_name": "songliaorrevent",
                "offset_to_utc": True,  # if you use Chinese dataset with start time 08:00, you need to set it to True
                "trange4cache": ["1950-01-01-02", "2024-12-31-23"],
            },
        },
        ctx=[DEVICE],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 128,
        },
        gage_id=[
            "songliao_20800900",
            "songliao_20810200",
            "songliao_21100150",
            "songliao_21110150",
            # "songliao_21113800",
            "songliao_21401050",
            "songliao_21401550",
        ],
        batch_size=8,
        #warmup_length=90,
        hindcast_length=90,
        forecast_length=24,
        # for flood event dataset, we need to set the forecast length for testing
        frwin=24,
        min_time_unit="h",
        min_time_interval="3",
        var_t=["rain"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        dataset="FloodEventDataset",
        scaler="DapengScaler",
        variable_length_cfgs={
            # whether to use variable length training
            "use_variable_length": True,
            # variable length type:
            # - "fixed": use predefined lengths (replaces old multi_length_training)
            # - "dynamic": automatic padding with mask (replaces old mask_cfgs)
            "variable_length_type": "dynamic",
            # for "fixed" type: specify exact sequence lengths to use
            "fixed_lengths": None,
            # Pad strategy: "Pad" or "multi_table" (multi_table not fully tested yet)
            "pad_strategy": "Pad",
        },
        train_mode=True,  # 设置为False，只测试不训练
        train_epoch=60,
        save_epoch=5,
        model_loader={"load_way": "specified", "test_epoch": 60},
        train_period=["1980-01-01-02", "2018-12-31-23"],
        valid_period=["2019-01-01-02", "2024-12-31-23"],
        test_period=["2019-01-01-02", "2024-12-31-23"],
        #test_period=["2020-01-01-02", "2024-12-30-23"],
        # loss_func="RMSESum",
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 1,
            "non_flood_weight": 0,
            "flood_strategy": "weight",
            "device": [0],
        },
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            # "lr": 0.01,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
