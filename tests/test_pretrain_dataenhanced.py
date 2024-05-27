"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2024-02-14 16:12:50
LastEditors: Wenyu Ouyang
Description: A test function for transfer learning
FilePath: \torchhydro\tests\test_transfer_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import logging
import pandas as pd
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv("data/basin_id(46+1).csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


@pytest.fixture()
def config():
    weight_path = os.path.join(
        os.getcwd(), "results", "test_mean_seq2seq", "ex22", "best_model.pth"
    )
    project_name = "test_pretrain_enhanced/ex1"
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm",
                "target": "basins-origin/hour_data/1h/mean_data/streamflow_basin",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_type="TransLearn",
        model_name="DataEnhanced",
        model_hyperparam={
            "hidden_length": 256,
            "input_size": 19,
            "output_size": 1,
            "hidden_size": 256,
            "cnn_size": 120,
            "forecast_length": 24,
            "model_mode": "dual",
            "prec_window": 1,
        },
        model_loader={"load_way": "best"},
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        loss_func="RMSESum",
        batch_size=256,
        forecast_history=336,
        forecast_length=24,
        train_period=[
            ("2015-06-01", "2015-09-30"),
            ("2016-06-01", "2016-09-30"),
            ("2017-06-01", "2017-09-30"),
            ("2018-06-01", "2018-09-30"),
            ("2019-06-01", "2019-09-30"),
            ("2020-06-01", "2020-09-30"),
            ("2021-06-01", "2021-09-30"),
            ("2022-06-01", "2022-09-30"),
        ],
        test_period=[
            ("2023-06-01", "2023-09-30"),
        ],
        valid_period=[
            ("2023-06-01", "2023-09-30"),  # 目前只支持一个时段
        ],
        dataset="MultiSourceDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": [
                "encoder1.lstm.weight_ih_l0",
                "encoder1.lstm.weight_hh_l0",
                "encoder1.lstm.bias_ih_l0",
                "encoder1.lstm.bias_hh_l0",
                "encoder1.fc.weight",
                "encoder1.fc.bias",
                "encoder1.fc2.weight",
                "encoder1.fc2.bias",
                "encoder2.lstm.weight_ih_l0",
                "encoder2.lstm.weight_hh_l0",
                "encoder2.lstm.bias_ih_l0",
                "encoder2.lstm.bias_hh_l0",
                "encoder2.fc.weight",
                "encoder2.fc.bias",
                "encoder2.fc2.weight",
                "encoder2.fc2.bias",
                "decoder1.lstm.weight_ih_l0",
                "decoder1.lstm.weight_hh_l0",
                "decoder1.lstm.bias_ih_l0",
                "decoder1.lstm.bias_hh_l0",
                "decoder1.fc_out.weight",
                "decoder1.fc_out.bias",
                "decoder1.attention.attn.weight",
                "decoder2.lstm.weight_ih_l0",
                "decoder2.lstm.weight_hh_l0",
                "decoder2.lstm.bias_ih_l0",
                "decoder2.lstm.bias_hh_l0",
                "decoder2.fc_out.weight",
                "decoder2.fc_out.bias",
                "decoder2.attention.attn.weight",
            ]
        },
        continue_train=True,
        train_epoch=50,
        save_epoch=1,
        var_t=[
            "gpm_tp",
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
        gage_id=[
            # "21401550",碧流河
            # "01181000",
            # "01411300",2021年缺失，暂时不用
            # "01414500",
            # "02016000",
            "02018000",
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
        which_first_tensor="batch",
        rolling=False,
        long_seq_pred=False,
        early_stopping=True,
        patience=10,
        ensemble=True,
        ensemble_items={
            "kfold": 9,
            "batch_sizes": [1024],
        },
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_dataenhanced(config):
    # train_and_evaluate(config)
    ensemble_train_and_evaluate(config)
