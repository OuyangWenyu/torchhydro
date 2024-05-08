"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2024-04-17 13:31:16
LastEditors: Xinzhuo Wu
Description: 
FilePath: /torchhydro/tests/test_train_seq2seq.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate

import logging

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


@pytest.fixture()
def config():
    project_name = "test_mean_seq2seq/ex24"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land",
                "target": "basins-origin/hour_data/1h/mean_data/streamflow_basin",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 19,  # dual比single少2
            "output_size": 1,
            "hidden_size": 256,
            "cnn_size": 120,
            "forecast_length": 24,
            "model_mode": "dual",
            "prec_window": 1,  # 将前序径流一起作为输出，选择的时段数，该值需小于等于rho
        },
        model_loader={"load_way": "best"},
        gage_id=[
            # "21401550",碧流河
            "01181000",
            # "01411300",2021年缺失，暂时不用
            "01414500",
            # "02016000",
            # "02018000",
            # "02481510",
            # "03070500",
            # "08324000",
            # "11266500",
            # "11523200",
            # "12020000",
            # "12167000",
            # "14185000",
            # "14306500",
        ],
        batch_size=256,
        rho=336,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
            "sm_rootzone",
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
        dataset="ERA5LandDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=1,
        save_epoch=1,
        train_period=[
            # ("2019-06-01", "2019-10-31"),
            # ("2020-06-01", "2020-10-31"),
            # ("2021-06-01", "2021-10-31"),
            ("2023-06-01", "2023-09-30"),
        ],
        test_period=[
            ("2022-06-01", "2022-09-30"),
        ],
        valid_period=[
            ("2023-06-01", "2023-09-30"),
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        rolling=False,
        static=False,
        early_stopping=True,
        patience=4,
        ensemble=True,
        ensemble_items={
            "kfold": 5,
            "batch_sizes": [256],
        },
    )
    update_cfg(config_data, args)
    return config_data


def test_seq2seq(config):
    train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
