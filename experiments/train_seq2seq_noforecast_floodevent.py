"""
Author: Wenyu Ouyang
Date: 2025-12-16
Description: Seq2SeqNoForecast模型训练脚本 - 洪水场次数据，不使用预测期降雨作为输入（消融实验）
FilePath: \\torchhydro\\experiments\\train_seq2seq_noforecast_floodevent.py
Copyright (c) 2021-2025 Wenyu Ouyang. All rights reserved.
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


def create_seq2seq_noforecast_config():
    """
    创建Seq2SeqNoForecast模型配置

    这个配置用于训练不使用预测期降雨作为输入的Seq2Seq模型
    模型只依赖后验期降雨和自回归预测
    """
    # 获取默认配置
    config_data = default_config_file()

    # 设置实验目录
    project_name = os.path.join("flood_event_Seq2SeqNoForecast", "exp20251216")

    # 设备配置（-1表示CPU，0表示GPU 0）
    DEVICE = -1

    # 训练超参数
    hindcast_length = 56  # 后验期长度7天
    forecast_length = 16  # 预测期长度2天

    args = cmd(
        sub=project_name,
        # ========== 数据源配置 ==========
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
                "dataset_name": "songliaorrevent",
                "net_rain_key": "net_rain",
                "obs_flow_key": "inflow",
                "delta_t_hours": 3.0,
            },
        },
        ctx=[DEVICE],

        # ========== 模型配置 ==========
        model_name="Seq2SeqNoForecast",  # 使用不带预测期降雨输入的Seq2Seq模型
        model_hyperparam={
            "en_input_size": 1,           # 编码器输入特征数（降雨）
            "de_input_size": 1,           # 解码器输入维度（将被模型内部忽略）
            "output_size": 1,             # 输出特征数（只预测径流，flood_event仅用于loss加权）
            "hidden_size": 16,            # LSTM隐藏层大小
            "forecast_length": forecast_length,        # 预测步长（必须与training_cfgs.forecast_length一致）
            "hindcast_output_window": hindcast_length, # 后验期输出窗口（输出完整后验期）
            "teacher_forcing_ratio": 0.5, # teacher forcing比例（训练时使用真实值的概率）
        },
        # hindcast_output_window=hindcast_length,  # 数据集配置：后验期输出窗口

        # ========== 站点配置 ==========
        gage_id=["songliao_21401550"],  # 训练站点

        # ========== 数据集配置 ==========
        dataset="FloodEventSeq2SeqDataset",  # 使用洪水场次Seq2Seq数据集
        scaler="DapengScaler",
        batch_size=8,
        warmup_length=0,
        hindcast_length=hindcast_length,
        forecast_length=forecast_length,
        frwin=forecast_length,
        min_time_unit="h",
        min_time_interval="3",

        # ========== 输入输出变量 ==========
        var_t=["rain"],                       # 时间序列变量（降雨）
        t_rm_nan=False,
        var_c=["None"],                       # 常量变量（无）
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],    # 输出变量（径流 + 洪水事件标记）
        target_as_input=False,

        # ========== 变长序列配置 ==========
        variable_length_cfgs={
            "use_variable_length": True,
            "variable_length_type": "dynamic",
            "fixed_lengths": None,
            "pad_strategy": "Pad",
        },

        # ========== 训练配置 ==========
        train_epoch=20,
        save_epoch=1,
        model_loader={"load_way": "best"},

        # ========== 时间段配置 ==========
        train_period=[
            ["1984-01-01-02", "2017-12-31-23"],  # 真实历史数据
            ["2026-07-01-02", "3150-12-31-23"],  # 增强数据
        ],
        valid_period=["2018-01-01-02", "2024-12-31-23"],  # 验证期
        test_period=["2018-01-01-02", "2023-12-31-23"],   # 测试期

        # ========== 损失函数配置 ==========
        loss_func="FloodLoss",              # 洪水区间loss
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            # Score-based weighting parameters (for augmented data with quality scores 1-100)
            # Uncomment below to enable score-based weighting:
            "use_score_weighting": False,    # Enable score-based weighting (flood_mask: -1/0=non-flood, 1-100=scores)
            "score_weight_min": 0.5,        # Weight for lowest quality (score=1)
            "score_weight_max": 3.0,        # Weight for highest quality (score=100)
            "score_alpha": 3.0,             # Control high-quality emphasis (>1: emphasize high scores)
            "device": [DEVICE],
        },
        # loss_func="PeakShapeLoss",        # 洪峰三指标loss
        # loss_param={
        #     "loss_func": "MSELoss",
        #     "peak_window_size": 8,         # 洪峰前后8步（24小时，3h分辨率）
        #     "shape_weight": 0.5,           # 形状相似度权重（皮尔逊相关系数）
        #     "mse_weight": 0.5,             # MSE权重（绝对值匹配）
        #     "flood_weight": 1.0,           # 洪水期权重（暂时保留，未来可能移除）
        #     "non_flood_weight": 0.5,       # 非洪水期权重
        #     "device": [DEVICE],
        # },

        # ========== 优化器配置 ==========
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },

        # ========== 其他配置 ==========
        which_first_tensor="sequence",  # Seq2Seq模型要求序列维度在前
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},  # 只评估洪水场次

        # ========== 洪峰对齐采样配置 ==========
        peak_aligned_sampling={
            "enabled": True,          # 启用洪峰对齐采样
            "slide_step": 1,          # 滑动步长
            "peak_selection": "first" # 选择第一个洪峰
        },
    )

    # 更新配置
    update_cfg(config_data, args)

    return config_data


def train_seq2seq_noforecast():
    """训练Seq2SeqNoForecast模型"""
    try:
        # 创建配置
        config_data = create_seq2seq_noforecast_config()

        # 开始训练
        print("=" * 80)
        print("开始训练 Seq2SeqNoForecast 模型（消融实验：无预测期降雨输入）")
        print("=" * 80)

        train_and_evaluate(config_data)

        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    """
    配置说明：
       - 模型：Seq2SeqNoForecast（编码器-解码器架构，但解码器不使用预测期降雨输入）
       - 数据集：FloodEventSeq2SeqDataset（基于洪水场次组织数据，支持洪峰对齐采样）
       - 采样策略：从洪峰点开始向后滑动窗口
       - 评估方式：只评估洪水场次期间的性能

    """
    train_seq2seq_noforecast()
