"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2024-11-05 11:40:28
LastEditors: Wenyu Ouyang
Description:
FilePath: \torchhydro\experiments\train_with_era5land.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pathlib

import pandas as pd
import pytest
import hydrodatasource.configs.config as hdscc
import xarray as xr
import torch.multiprocessing as mp

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import train_worker
from torchhydro.trainers.trainer import train_and_evaluate

# from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent.parent, "data/basin_5819.csv"),
    dtype={"id": str},
)
gage_id = show["id"].values.tolist()
# gage_id = ["grdc_1159310", "grdc_1159300"]


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("train_with_google", "ex_test2")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": "/ftproot/google_flood",
            "other_settings": {
                "time_unit": ["1D"],
            },
        },
        ctx=[2],
        model_name="SeqForecastLSTM",
        model_hyperparam={
            "static_input_dim": 15,
            "dynamic_input_dim": 6,
            "embedding_dim": 1,
            "hidden_dim": 256,
            "output_dim": 1,
            "prec_window": 1,
            "use_paper_model": True,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=256,
        forecast_history=180,
        forecast_length=5,
        min_time_unit="D",
        min_time_interval=1,
        var_t=[
            # "precipitationCal",
            "total_precipitation_hourly",
            "temperature_2m",
            "surface_net_solar_radiation_hourly",
            "surface_net_thermal_radiation_hourly",
            "snowfall_hourly",
            "surface_pressure",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow"],
        dataset="SeqForecastDataset",
        # sampler="BasinBatchSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_period=["2019-06-01", "2021-11-01"],
        test_period=["2021-11-01", "2022-12-01"],
        valid_period=["2021-11-01", "2022-12-01"],
        loss_func="RMSESum",
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0],
        #     "device": [2],
        #     "item_weight": [1],
        # },
        opt="Adam",
        # lr_scheduler={
        #     "lr": 0.0001,
        #     "lr_factor": 0.9,
        # },
        lr_scheduler={
            epoch: (
                0.5
                if 1 <= epoch <= 9
                else (
                    0.2
                    if 10 <= epoch <= 29
                    else (
                        0.1
                        if 30 <= epoch <= 69
                        else 0.05 if 70 <= epoch <= 89 else 0.02
                    )
                )
            )
            for epoch in range(1, 101)
        },
        which_first_tensor="batch",
        calc_metrics=False,
        # metrics=["NSE", "RMSE"],
        early_stopping=True,
        rolling=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        # model_type="Normal",
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
