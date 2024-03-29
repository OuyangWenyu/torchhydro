"""
Author: Xinzhuo Wu
Date: 2023-07-25 16:47:19
LastEditTime: 2024-03-29 08:53:02
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_mean_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import hydrodataset as hds
import pandas as pd
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate


@pytest.fixture()
def config():
    project_name = "test_mean_lstm/exp6_03"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_path=os.path.join(hds.ROOT_DIR),
        rainfall_source_path="basins-origin/hour_data/1h/mean_data/mean_data_forcing",
        streamflow_source_path="basins-origin/hour_data/1h/mean_data/mean_data_target",
        attributes_path="basins-origin/attributes.nc",
        ctx=[0],
        model_name="SimpleLSTMForecast",
        model_hyperparam={
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 256,
            "forecast_length": 24,
        },
        gage_id=[
            "21401550",
        ],
        batch_size=256,
        rho=168,
        var_t=["gpm_tp", "gfs_tp"],
        var_c=[
            "area",  # 面积
            # "ele_mt_smn",  # 海拔(空间平均)
            # "slp_dg_sav",  # 地形坡度 (空间平均)
            # "sgr_dk_sav",  # 河流坡度 (平均)
            # "for_pc_sse",  # 森林覆盖率
            # "glc_cl_smj",  # 土地覆盖类型
            # "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            # "inu_pc_slt",  # 淹没范围 (长期最大)
            # "cmi_ix_syr",  # 气候湿度指数
            # "aet_mm_syr",  # 实际蒸散发 (年平均)
            # "snw_pc_syr",  # 雪盖范围 (年平均)
            # "swc_pc_syr",  # 土壤水含量
            # "gwt_cm_sav",  # 地下水位深度
            # "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            # "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow"],
        dataset="MEAN_Dataset",
        sampler="WuSampler",
        scaler="DapengScaler",
        train_epoch=50,
        save_epoch=1,
        te=50,
        train_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
            {"start": "2018-07-01", "end": "2018-09-29"},
            {"start": "2020-07-01", "end": "2020-09-29"},
            {"start": "2021-07-01", "end": "2021-09-29"},
        ],
        test_period=[
            {"start": "2019-07-01", "end": "2019-09-29"},
        ],
        valid_period=[
            {"start": "2019-07-01", "end": "2019-09-29"},
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={1: 1e-3},
        which_first_tensor="sequence",
        lr_factor=0.96,
        # lr_patience=0,
        # weight_decay=1e-5,
        lr_val_loss=True,
        early_stopping=True,
        patience=4,
        ensemble=True,
        ensemble_items={
            "kfold": 5,
            "batch_sizes": [256],
        },
        user="zxw",
    )
    update_cfg(config_data, args)
    return config_data


def test_mean_lstm(config):
    train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
