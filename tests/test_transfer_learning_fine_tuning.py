"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2025-01-XX XX:XX:XX
LastEditors: Wenyu Ouyang
Description: 迁移学习微调测试 - 在预训练模型基础上增加新输入变量（如rain）
             Transfer learning fine-tuning test - Adding new input variables (like rain) to pretrained model
FilePath: \\torchhydro\\tests\\test_transfer_learning_fine_tuning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

DEVICE = 1

@pytest.mark.requires_data
@pytest.mark.skip(reason="Gages/Mopex/SupData4Camels data sources removed from torchhydro (moved to hydrodatasource)")
def test_transfer_gages_lstm_model():
    """
    ========== 迁移学习微调：输入通道扩展 + 缺失值掩码 ==========
    Transfer Learning Fine-tuning: Input Channel Expansion + Missing Value Masking

    该函数演示如何在预训练模型基础上微调，并增加新的输入变量。
    This function demonstrates how to fine-tune a pretrained model with new input variables.

    核心改动 Key Changes:
    1. input_size: 41 -> 42 (新增rain变量 / Added rain variable)
    2. var_t: 增加 "rain" 到输入变量列表 / Add "rain" to input variable list
    3. weight_path_add: 配置输入扩展参数 / Configure input expansion parameters
    4. missing_value_vars: 配置缺失值掩码 / Configure missing value masking

    预训练模型 Pretrained Model:
    - 数据集 Dataset: CAMELS (美国流域数据)
    - 输入维度 Input size: 41 (6个时序变量 + 35个流域属性)
    - 模型 Model: SimpleLSTM with wconv=True

    微调模型 Fine-tuning Model:
    - 数据集 Dataset: Songliao (松辽流域洪水事件数据)
    - 输入维度 Input size: 42 (7个时序变量[新增rain] + 35个流域属性)
    - 新增变量 New variable: "rain" (实测降雨，存在缺失值)

    ========== 方案一：部分权重加载 + 输入通道扩展 ==========
    Plan 1: Partial Weight Loading + Input Channel Expansion

    工作流程 Workflow:
    1. 从预训练模型加载LSTM层等共享层的权重
       Load LSTM and other shared layer weights from pretrained model

    2. 输入层权重扩展：将input_size=41扩展到42
       Input layer weight expansion: expand input_size from 41 to 42
       - 对于已有的41个通道，直接复制预训练权重
         For existing 41 channels, copy pretrained weights
       - 对于新增的第42个通道（rain），使用初始化策略（默认copy_mean）
         For new 42nd channel (rain), use initialization strategy (default: copy_mean)

    3. 缺失值掩码处理：rain变量存在缺失值
       Missing value masking: rain variable has missing values
       - Dataset返回 (xc, mask) 元组，mask标记有效/缺失位置
         Dataset returns (xc, mask) tuple, mask marks valid/missing positions
       - 损失函数可利用mask排除缺失值的影响
         Loss function can use mask to exclude influence of missing values

    适用场景 Use Cases:
    - 在预训练模型基础上微调到新流域 Fine-tune pretrained model to new basin
    - 新流域有额外的观测数据（如实测降雨） New basin has additional observations (like rain)
    - 新增变量数据不完整，存在缺失值 New variable data is incomplete with missing values
    """

    # ========== 1. 预训练模型路径配置 ==========
    # Pretrained Model Path Configuration
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "simple_lstm_DapengScaler",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)

    # ========== 2. 项目和数据源配置 ==========
    # Project and Data Source Configuration
    project_name = "test_camels/exp_fine_tuning"
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
            ),
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "songliaorrevent",
                "offset_to_utc": True,
                "obs_flow_key": "inflow",
            },
        },
        ctx=[DEVICE],
        gage_id=[
            'songliao_20800900',
            'songliao_20810200',
            'songliao_21100150',
            'songliao_21110150',
            'songliao_21113800',
            'songliao_21401050',
            'songliao_21401550',
        ],

        # ========== 3. 模型配置 - 关键改动：input_size扩展 ==========
        # Model Configuration - Key Change: input_size expansion
        model_type="TransLearn",
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 42,  # 改动：41 -> 42，新增rain变量 
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
            # "wconv": True,
        },

        # ========== 4. 训练参数配置 ==========
        # Training Parameters Configuration
        batch_size=8,
        hindcast_length=0,
        forecast_length=20,
        frwin=20,
        min_time_unit="D",
        min_time_interval="1",

        # ========== 5. 输入变量配置 - 关键改动：新增rain变量 ==========
        # Input Variables Configuration - Key Change: Added rain variable
        var_t=[
            "total_precipitation_hourly",  # ERA5Land网格降水 / ERA5Land gridded precipitation
            "rain",  # 新增：实测降雨（有缺失值）/ NEW: Observed rainfall (with missing values)
            "temperature_2m",
            "surface_net_solar_radiation_hourly",
            "surface_net_thermal_radiation_hourly",
            "snowfall_hourly",
            "surface_pressure"
        ],
        t_rm_nan=False,  # 重要：不删除NaN，由缺失值掩码处理 / Important: Don't remove NaN, handled by missing value mask

        # 流域属性变量（与预训练模型相同）
        # Basin attribute variables (same as pretrained model)
        var_c=[
            "area",
            "pet_mm_syr",
            "aet_mm_syr",
            "ele_mt_sav",
            "lka_pc_sse",
            "for_pc_sse",
            "slp_dg_sav",
            "lkv_mc_usu",
            "ero_kh_sav",
            "tmp_dc_syr",
            "riv_tc_usu",
            "gdp_ud_sav",
            "soc_th_sav",
            "kar_pc_sse",
            "ppd_pk_sav",
            "nli_ix_sav",
            "pst_pc_sse",
            "pac_pc_sse",
            "snd_pc_sav",
            "pop_ct_usu",
            "swc_pc_syr",
            "ria_ha_usu",
            "slt_pc_sav",
            "cly_pc_sav",
            "crp_pc_sse",
            "inu_pc_slt",
            "cmi_ix_syr",
            "snw_pc_syr",
            "ari_ix_sav",
            "ire_pc_sse",
            "rev_mc_usu",
            "inu_pc_smn",
            "urb_pc_sse",
            "prm_pc_sse",
            "gla_pc_sse"
        ],
        c_rm_nan=False,

        # 输出变量配置（保持不变）
        # Output variables configuration (unchanged)
        var_out=[
            "inflow",  # 入库流量 / Inflow
            "flood_event"  # 洪水事件标记 / Flood event indicator
        ],

        # ========== 6. 数据集配置 - 关键改动：缺失值掩码 ==========
        # Dataset Configuration - Key Change: Missing value masking
        dataset="FloodEventDataset",
        rs=1234,
        train_period=["2000-01-01","2014-12-31"],
        test_period=["2020-01-01","2023-12-31"],
        scaler="DapengScaler",
        sampler="KuaiSampler",


        # ========== 7. 权重加载配置 - 关键改动：输入扩展 ==========
        # Weight Loading Configuration - Key Change: Input expansion
        weight_path=weight_path,
        # ⚠️ 重要：不设置stat_dict_file，让训练阶段从训练数据重新计算统计量
        # Important: Do NOT set stat_dict_file, let training stage recalculate statistics from training data
        # 因为新增了rain变量，旧的stat文件不包含rain的统计量
        # Since we added rain variable, old stat file doesn't contain rain statistics
        # stat_dict_file=None,  # 默认值，可以不写

        # 权重加载附加配置：输入通道扩展
        # Weight loading additional configuration: Input channel expansion
        weight_path_add={
            "input_expansion": {
                "old_input_size": 41,  # 预训练模型的输入维度 / Pretrained model input size
                # new_input_size会自动从model_hyperparam["input_size"]获取
                # new_input_size is automatically obtained from model_hyperparam["input_size"]
                "init_strategy": "copy_mean",  # 新通道权重初始化策略 / New channel weight initialization strategy
                # 可选策略 Available strategies:
                # - "copy_mean": 复制已有通道权重的均值（推荐，稳定）
                #                Copy mean of existing channel weights (recommended, stable)
                # - "copy_first": 复制第一个通道的权重
                #                 Copy first channel weights
                # - "random": 随机初始化（与预训练权重分布相同）
                #             Random initialization (same distribution as pretrained weights)
                # - "zero": 零初始化（不推荐，可能导致梯度消失）
                #           Zero initialization (not recommended, may cause gradient vanishing)
            }
            # 可选：冻结部分参数
            # Optional: Freeze some parameters
            # "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },

        # ========== 8. 训练和优化配置 ==========
        # Training and Optimization Configuration
        continue_train=False,
        train_epoch=20,
        save_epoch=1,
        train_mode=False,  # 改为True以启动训练 / Set to True to start training
        opt="Adam",

        # 损失函数配置：洪水加权损失
        # Loss function configuration: Flood weighted loss
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,  # 洪水期权重 / Flood period weight
            "flood_strategy": "weight",  # 加权策略 / Weighting strategy
            "device": [DEVICE],
            # 注意：FloodLoss会自动处理缺失值掩码（如果Dataset返回）
            # Note: FloodLoss will automatically handle missing value mask (if returned by Dataset)
        },

        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },

        # ========== 9. 其他配置 ==========
        # Other Configuration
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=1,
        calc_metrics=True,
        evaluator={"eval_way": "floodevent"},
    )

    # ========== 10. 执行训练和评估 ==========
    # Execute Training and Evaluation
    cfg = default_config_file()
    update_cfg(cfg, args)
    # ========== 11. 手动添加缺失值掩码配置（cmd()函数不支持此参数）==========
    # Manually add missing value masking configuration (not supported by cmd() function)
    # 新增：缺失值掩码配置（通用设置，适用于任何变量名）
    # NEW: Missing value masking configuration (generic setting for any variable name)
    cfg["data_cfgs"]["missing_value_vars"] = ["rain"] 
    train_and_evaluate(cfg)
    print("All processes are finished!")

    
