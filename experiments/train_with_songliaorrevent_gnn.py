"""
Author: Yang Wang
Date: 2025-01-08 15:00:00
LastEditTime: 2025-07-13 18:12:30
LastEditors: Wenyu Ouyang
Description: GNN训练脚本，使用torchhydro框架
FilePath: \torchhydro\experiments\train_with_songliaorrevent_gnn.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import logging

from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

# 配置日志
logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


def create_simple_gnn_config():
    """创建简单的GNN配置"""
    # 获取默认配置
    config_data = default_config_file()

    # 设置实验目录
    project_name = os.path.join("gnn_experiment", "songliao_3h_test")

    # 使用有有效数据的流域ID（基于3h数据）
    example_gage_ids = [
        "songliao_20800900",
        "songliao_20810200",
        "songliao_21100150",
        "songliao_21110150",
        #"songliao_21113800",
        "songliao_21401050",
        "songliao_21401550"
    ]

    # ==== 智能推断 in_channels 和 output_size ====
    hindcast_length = 20
    forecast_length = 2
    var_t = [
        "total_precipitation_hourly",
        "temperature_2m",
        "u_component_of_wind_10m",
        "surface_pressure"
    ]
    var_c = [
        "area",
        "ele_mt_smn",
        "slp_dg_sav",
        "for_pc_sse",
        "run_mm_syr",
    ]
    # 站点变量（如有 GNN 站点特征）
    station_cols = ["DRP"]  # 如有多个，补全即可
    feature_num = len(var_t) + len(var_c) + len(station_cols)
    in_channels = (hindcast_length + forecast_length) * feature_num
    output_size = hindcast_length+forecast_length  # flood_event 只预测未来1步

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "source_name": "stationhydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
        },
        # station_cfgs={
        #     "station_cols": station_cols,  # 站点只有降雨数据目前,
        #     },
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 模型配置
        model_name="GCN",  # 使用GCN模型
        model_hyperparam={
            "in_channels": in_channels,
            "hidden_channels": 64,
            "num_hidden": 2,
            "param_sharing": True,
            "edge_orientation": "downstream",
            "output_size": output_size,
            "root_gauge_idx": 1,
        },
        dataset="GNNDataset",
        scaler="DapengScaler",
        batch_size=2,
        warmup_length=0,
        hindcast_length=hindcast_length,
        forecast_length=forecast_length,
        # station_cols 不是 cmd 的参数，直接通过 gnn_cfgs 传递
        # gnn_cfgs={
        #     # 站点数据配置 - 使用3h数据中实际存在的变量
        #     "station_cols": station_cols  # 站点只有降雨数据目前
        # },
        min_time_unit="h",
        min_time_interval=3,
        var_t=var_t,
        var_c=var_c,
        var_out=["inflow", "flood_event"],
        train_epoch=10,
        save_epoch=10,
        train_period=["1995-06-01-02", "1995-08-31-02"],  # 使用有完整inflow数据的时段
        test_period=["1995-06-01-02", "1996-06-01-02"],   # 测试时段
        # TODO: 测试rolling=1是否可行，之前加 force_is_new_batch_way 的方式不合理，
        # 会把原来的代码逻辑损坏，所以这里改为rolling=1，但是需要测试下在FloodEvent数据下是否可行
        # 并且注意，下面hrwin和frwin的设置需要与rolling=1保持一致
        rolling=1,  # 使用滑动窗口，与训练模式类似
        model_loader={"load_way": "specified", "test_epoch": 10},
        evaluator={"eval_way": "floodevent"},
        # 设置评估窗口与训练窗口保持一致
        hrwin=hindcast_length,  # 与训练时的hindcast_length保持一致
        frwin=forecast_length,  # 与训练时的forecast_length保持一致
        which_first_tensor="batch",

        # 优化器配置
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },
        # 损失函数
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "device": [0],
        },
        # 其他配置
        # num_workers=4,
        # pin_memory=True,
        early_stopping=True,
        patience=5,
        calc_metrics=True,
        continue_train=False,
    )

    # 更新配置
    update_cfg(config_data, args)

    return config_data


def train_gnn_model():
    """训练GNN模型"""
    try:
        # 创建配置
        config_data = create_simple_gnn_config()

        # 开始训练
        print("开始训练GNN模型...")
        train_and_evaluate(config_data)

        print("训练完成！")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    """
    使用说明：
    1. 根据你的数据路径修改 source_cfgs 中的 source_path
    2. 根据你的站点修改 example_gage_ids 列表
    3. 根据你的数据特征修改 var_t, var_c, var_out 变量
    4. 根据需要调整模型超参数 model_hyperparam
    5. 根据需要调整训练参数（epoch数、学习率等）
    """
    train_gnn_model()
