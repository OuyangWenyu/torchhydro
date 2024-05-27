"""
Author: Xinzhuo Wu
Date: 2023-09-30 1:20:18
LastEditTime: 2024-02-13 13:18:21
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process with Spp_Lstm
FilePath: \torchhydro\tests\test_spp_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import pytest
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex1"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="HydroGrid",
        source_path=[
            {
                "gpm": "basins-origin/hour_data/1h/grid_data/grid_gpm_data",
                "gfs": "basins-origin/hour_data/1h/grid_data/grid_gfs_data",
                "smap": "basins-origin/hour_data/1h/grid_data/grid_smap_data",
                "target": "basins-origin/hour_data/1h/mean_data/mean_data_target",
                "attributes": "basins-origin/attributes.nc",
            }
        ],
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={  # p代表降水+GFS气象网络分支，S代表土壤属性网络分支
            "forecast_history": 168,
            "forecast_length": 24,
            "p_n_output": 1,
            "p_n_hidden_states": 60,
            "p_dropout": 0.25,
            "p_in_channels": 1,  # 1 or 9 卷积层输入通道数，等于len(var_t[0]+len(var_t[1]))，即降水与GFS气象属性的数量
            "p_out_channels": 8,  # 输出通道数，值越大，模型越复杂，建议比in_channels大
            "len_c": 15,  # 需要与len(var_c)相等,即如果不用流域属性，则设置为0
            "s_forecast_history": None,  # 土壤属性所用历史时段，比如只能获取两天半前的数据，则为108,目前只支持108
            "s_n_output": 1,
            "s_n_hidden_states": 60,
            "s_dropout": 0.25,
            "s_in_channels": 8,  # 卷积层输入通道数，等于len(var_t[2])，即土壤属性的数量
            "s_out_channels": 8,
        },
        gage_id=["02051500", "21401550"],
        batch_size=256,
        var_t=[  # var_t[0]为["gpm_tp"], var_t[1]为GFS, var_t[2]为土壤含水量，无则["None"]
            ["gpm_tp"],
            [
                "None",
                # "gfs_tp",
                # "dswrf",
                # "pwat",
                # "2r",
                # "2sh",
                # "2t",
                # "tcc",
                # "10u",
                # "10v",
            ],
            [
                "None",
            ],
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
        dataset="GridDataset",
        sampler="HydroSampler",
        scaler="MutiBasinScaler",
        train_epoch=1,
        save_epoch=1,
        te=1,
        train_period=[
            ("2017-07-08", "2017-09-29"),
            ("2018-07-08", "2018-09-29"),
            ("2019-07-08", "2019-09-29"),
            ("2020-07-08", "2020-09-29"),
        ],
        test_period=[
            ("2021-07-08", "2021-09-29"),
        ],
        valid_period=[
            ("2021-07-08", "2021-09-29"),
        ],
        loss_func="RMSESum",  # NSELoss、RMSESum、MAPELoss、MASELoss、MAELoss 可选
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },  # 初始化学习率(第1轮开始即使用1e-3)，可强制指定某一轮使用的学习率； 学习率更新权重
        which_first_tensor="sequence",
        early_stopping=True,
        patience=4,  # 连续n次valid loss不下降，则停止训练，与early_stopping配合使用
        rolling=False,  # evaluate 不采用滚动预测
        ensemble=True,  # 交叉验证
        ensemble_items={
            "kfold": 5,  # exi_0即17年验证,...exi_4即21年验证
            "batch_sizes": [256],
        },
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
