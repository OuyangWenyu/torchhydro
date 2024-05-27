"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2024-05-27 16:23:45
LastEditors: Wenyu Ouyang
Description: A test function for transfer learning
FilePath: \torchhydro\tests\test_pretrain_datafusion.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.models.seq2seq import DataEnhancedModel, GeneralSeq2Seq
import logging

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


@pytest.fixture()
def config():
    weight_path = os.path.join(
        os.getcwd(), "results", "test_pretrain_enhanced", "ex1", "best_model.pth"
    )
    project_name = "test_pretrain_fusion/exp1"
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/mean_data_merged",
                "target": "basins-origin/hour_data/1h/mean_data/mean_data_merged",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_type="TransLearn",
        model_name="DataFusion",
        model_hyperparam={
            "original_model": DataEnhancedModel(
                original_model=GeneralSeq2Seq(
                    input_size=16,
                    output_size=1,
                    hidden_size=256,
                    forecast_length=24,
                    model_mode="dual",
                ),
                hidden_length=256,
            ),
            "cnn_size": 120,  # todo 有冗余
            "forecast_length": 24,
            "model_mode": "dual",
        },
        model_loader={"load_way": "best"},
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        loss_func="RMSESum",
        batch_size=256,
        forecast_history=168,
        forecast_length=24,
        train_period=[
            ("2022-07-08", "2022-09-29"),
        ],
        test_period=[
            ("2023-07-08", "2023-09-29"),
        ],
        valid_period=[
            ("2023-07-08", "2023-09-29"),
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
                "encoder2.lstm.weight_ih_l0",
                "encoder2.lstm.weight_hh_l0",
                "encoder2.lstm.bias_ih_l0",
                "encoder2.lstm.bias_hh_l0",
                "encoder2.fc.weight",
                "encoder2.fc.bias",
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
                "lstm.weight_ih_l0",
                "lstm.weight_hh_l0",
                "lstm.bias_ih_l0",
                "lstm.bias_hh_l0",
                "fc.weight",
                "fc.bias",
            ]
        },
        continue_train=True,
        train_epoch=20,
        save_epoch=1,
        var_t=["gpm_tp(mm/h)", "smap(m3/m3)", "streamflow(m3/s)"],
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
        var_out=["streamflow(m3/s)"],
        gage_id=[
            "01181000",
            "01414500",
        ],
        which_first_tensor="batch",
        rolling=False,
        long_seq_pred=False,
        early_stopping=True,
        patience=4,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_dataenhanced(config):
    train_and_evaluate(config)
