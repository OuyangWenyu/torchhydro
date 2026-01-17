"""
Author: Wenyu Ouyang
Date: 2025-12-16
Description: SimpleLSTM模型训练脚本 - 洪水场次数据
FilePath: \\torchhydro\\experiments\\train_simplelstm_floodevent.py
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


def create_simplelstm_config():
    """
    SimpleLSTM是一个简单的单向LSTM模型，适合作为基线模型与Seq2Seq对比
    模型结构：LinearIn -> LSTM -> Dropout -> LinearOut
    """
    # 获取默认配置
    config_data = default_config_file()

    # 设置实验目录
    project_name = os.path.join("flood_event_SimpleLSTM", "exp20251216")

    # 设备配置（-1表示CPU，0表示GPU 0）
    DEVICE = -1

    # 训练超参数
    hindcast_length = 56  # 后验期长度7天 (3h * 56 = 168h = 7天)
    forecast_length = 16  # 预测期长度2天 (3h * 16 = 48h = 2天)

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
        model_name="SimpleLSTM",  # 使用简单LSTM模型
        model_hyperparam={
            "input_size": 1,      # 输入特征数（降雨）
            "output_size": 1,     # 输出特征数（只预测径流，flood_event仅用于loss加权）
            "hidden_size": 16,    # LSTM隐藏层大小
            "dr": 0.3,            # Dropout率
        },

        # ========== 站点配置 ==========
        gage_id=["songliao_21401550"],  # 训练站点

        # ========== 数据集配置 ==========
        dataset="AugmentedFloodEventDataset",  # 使用增强洪水场次数据集
        scaler="DapengScaler",
        batch_size=8,
        warmup_length=240,
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
            ["1984-01-01-02", "2017-12-31-23"],
            ["2026-07-01-02", "3150-12-31-23"],  # 真实历史数据
        ],
        valid_period=["1984-01-01-02", "2023-12-31-23"],  # 验证期
        test_period=["1984-01-01-02", "2023-12-31-23"],   # 测试期

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
        # loss_func="PeakShapeLoss",          # 洪峰三指标loss
        # loss_param={
        #     # 三个指标权重（暂定都是1.0）
        #     "peak_magnitude_weight": 1.0,
        #     "peak_timing_weight": 1.0,
        #     "flood_volume_weight": 1.0,

        #     # 峰现时间的非对称惩罚
        #     "late_penalty_factor": 2.0,   # 晚的惩罚更多
        #     "early_penalty_factor": 0.5,  # 早的惩罚少一点

        #     # 全局权重
        #     "flood_weight": 1.0,
        #     "non_flood_weight": 0.5,
        # },

        # ========== 优化器配置 ==========
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },

        # ========== 其他配置 ==========
        which_first_tensor="sequence",  # SimpleLSTM要求序列维度在前
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},  # 只评估洪水场次

        # ========== 洪峰对齐采样配置 ==========
        peak_aligned_sampling={
            "enabled": False,          # 启用洪峰对齐采样
            "slide_step": 1,          # 滑动步长
            "peak_selection": "first" # 选择第一个洪峰
        },
    )

    # 更新配置
    update_cfg(config_data, args)

    return config_data


def train_simplelstm():
    """训练SimpleLSTM模型"""
    try:
        # 创建配置
        config_data = create_simplelstm_config()

        # 开始训练
        print("=" * 80)
        print("开始训练 SimpleLSTM 模型（基线模型）")
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
    SimpleLSTM 模型说明:

    1. 模型结构:
       - 输入线性层: input_size -> hidden_size
       - LSTM层: hidden_size -> hidden_size
       - Dropout层
       - 输出线性层: hidden_size -> output_size

    2. 与Seq2Seq的区别:
       - SimpleLSTM: 单向处理整个序列，一次性输出所有时间步
       - Seq2Seq: 编码器-解码器结构，分阶段处理

    3. 数据格式:
       - 输入: [seq_len, batch_size, input_size]
         其中 seq_len = hindcast_length + forecast_length
       - 输出: [seq_len, batch_size, output_size]

    4. 适用场景:
       - 基线模型，用于对比Seq2Seq的改进效果
       - 验证是否需要编码器-解码器结构
       - 评估Teacher Forcing机制的价值

    使用方法:
        python experiments/train_simplelstm_floodevent.py
    """
    train_simplelstm()
