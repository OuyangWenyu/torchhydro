"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: Test a full training and evaluating process with Spp_Lstm
FilePath: \torchhydro\tests\test_spp_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex9"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        streamflow_source_path="/ftproot/biliuhe/merge_streamflow.nc",
        rainfall_source_path="/ftproot/biliuhe",  # 文件夹(流域编号.nc)
        attributes_path="/ftproot/biliuhe/camelsus_attributes.nc",
        gfs_source_path="/ftproot/biliuhe/gfs_forcing",  # 文件夹(流域编号.nc)
        download=0,
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 60,
            "dropout": 0.25,
            "len_c": 15,  # 需要与len(var_c)相等
            "in_channels": 1,  # 卷积层输入通道数，等于len(var_t)
            "out_channels": 8,  # 输出通道数
        },
        gage_id=["1_02051500", "86_21401550"],  # ["86_21401550"],
        batch_size=256,
        var_t=[
            ["tp"],
            # "dswrf",
            # "pwat",
            # "2r",
            # "2sh",
            # "2t",
            # "tcc",
            # "10u",
            # "10v",
        ],  # tp作为list放在第一个，后面放gfs的气象数据
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
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=50,
        save_epoch=1,
        te=50,
        train_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
            {"start": "2018-07-01", "end": "2018-09-29"},
            {"start": "2019-07-01", "end": "2019-09-29"},
            {"start": "2020-07-01", "end": "2020-09-29"},
        ],
        test_period=[
            {"start": "2021-07-01", "end": "2021-09-29"},
        ],
        valid_period=[
            {"start": "2021-07-01", "end": "2021-09-29"},
        ],
        loss_func="NSELoss",  # RMSESum",#loss可以选择
        opt="Adam",
        lr_scheduler={1: 1e-3},
        lr_factor=0.5,
        lr_patience=0,
        weight_decay=1e-5,  # L2正则化衰减权重
        lr_val_loss=True,  # False则用NSE作为指标，而不是val loss,来更新lr、model、早退，建议选择True
        which_first_tensor="sequence",
        early_stopping=True,
        patience=4,
        rolling=False,  # evaluate 不采用滚动预测
        ensemble=True,
        ensemble_items={
            "kfold": 5,  # exi_0即17年验证,...exi_4即21年验证
            "batch_sizes": [256],
        },
        user="zxw",  # 注意，在本地(/home/xxx)需要有privacy_config.yml
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    # train_and_evaluate(config)
    ensemble_train_and_evaluate(config)
