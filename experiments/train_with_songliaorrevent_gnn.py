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
    project_name = os.path.join("gnn_experiment", "simple_test")

    # 示例站点ID列表（你可以根据你的数据修改这些）
    example_gage_ids = [
        "01013500",  # 示例站点ID
        "01022500",
        "01030500",
        "01031500",
        "01047000",
    ]

    # 配置训练参数
    args = cmd(
        sub=project_name,
        # 数据源配置
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": "/ftproot/basins-interim/",
        },
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 模型配置
        model_name="GCN",  # 使用GCN模型
        model_hyperparam={
            "in_channels": 5,  # 输入特征数（与var_t长度对应）
            "hidden_channels": 64,  # 隐藏层大小
            "num_hidden": 2,  # 隐藏层数
            "param_sharing": True,  # 参数共享
            "edge_orientation": "downstream",  # 边的方向
            "edge_weights": None,  # 边权重
            "output_size": 1,  # 输出维度
            "root_gauge_idx": None,  # 根节点索引
        },
        # 数据配置
        dataset="GNNDataset",  # 使用GNNDataset
        scaler="DapengScaler",  # 数据标准化
        batch_size=len(example_gage_ids),  # 批量大小设为站点数
        hindcast_length=168,  # 7天 * 24小时
        forecast_length=24,  # 预测1天
        # 时间配置
        min_time_unit="h",
        min_time_interval=1,
        # 变量配置
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
        ],
        var_c=[
            "area",
            "ele_mt_smn",
            "slp_dg_sav",
            "for_pc_sse",
            "run_mm_syr",
        ],
        var_out=["streamflow"],
        # 训练配置
        train_epoch=10,
        save_epoch=2,
        train_period=[("2020-01-01-00", "2022-12-31-23")],
        test_period=[("2023-01-01-00", "2023-12-31-23")],
        # 优化器配置
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },
        # 损失函数
        loss_func="RMSE",
        # 其他配置
        num_workers=4,
        pin_memory=True,
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
