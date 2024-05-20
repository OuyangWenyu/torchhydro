"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-20 10:40:46
LastEditors: Xinzhuo Wu
Description: 
FilePath: /torchhydro/tests/train_with_gpm_streamflow.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import pandas as pd
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv("data/basin_id(46+1).csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


def main():
    # 创建测试配置
    config_data = create_config()

    # 运行测试函数
    test_seq2seq(config_data)


def create_config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = "train_with_gpm_streamflow/ex1"
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 168,
            "prec_window": 1,  # 将前序径流一起作为输出，选择的时段数，该值需小于等于rho，建议置为1
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        batch_size=1024,
        rho=672,
        var_t=[
            "gpm_tp",
            "sm_surface",
            "streamflow",
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
        var_out=["streamflow", "sm_surface"],
        dataset="GPMSTRDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2016-06-01", "2023-12-31")],
        test_period=[("2015-06-01", "2016-05-31")],
        valid_period=[("2015-06-01", "2016-05-31")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        rolling=False,
        static=False,
        early_stopping=True,
        patience=8,
        model_type="MTL",
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


def test_seq2seq(config_data):
    # 运行测试
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
