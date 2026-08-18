"""
Author: Yang Wang
Date: 2025-09-03 12:00:00
LastEditTime: 2025-09-03 12:00:00
LastEditors: Wenyu Ouyang
Description: 时空洪水预测模型对比测试脚本 - 对比LSTM、GNN、LSTM+GNN和Transformer架构 (简化版 main)
FilePath: \\torchhydro\\experiments\\train_with_gnn_lstm.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import logging
import warnings
import sys

from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 配置日志和警告 - 最大限度减少详细输出
logging.basicConfig(level=logging.WARNING)  # 只显示WARNING及以上级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)  # 减少详细输出

# 抑制所有警告，包括RuntimeWarning
warnings.filterwarnings("ignore")


def create_spatiotemporal_config(test_mode="normal", model_name="SpatioTemporalFloodNet3D_LSTM_only"):
    """创建时空洪水预测模型配置
    
    Args:
        test_mode: 测试模式 -> normal | sim1 | sim2 | sim3
        model_name: 模型名称 -> 见下方 model_configs
    """
    # 获取默认配置
    config_data = default_config_file()

    # 4个核心模型配置
    model_configs = {
        "SpatioTemporalFloodNet3D_LSTM_only": {
            "model_name": "SpatioTemporalFloodNet3D",  # 使用3D时空洪水网络
            "experiment_dir": "3d_lstm_only",
            "description": "LSTM-only基线 - 无图网络，仅LSTM时序建模+平均池化",
            "use_gnn": False,  # 关键参数：禁用GNN
            "use_lstm": True,  # 关键参数：启用LSTM
            "hidden_channels": 16,  # 减小隐藏维度，防止过拟合
            "dropout": 0.2,  # 增加dropout，防止过拟合
        },
        "SpatioTemporalFloodNet3D_GNN_only": {
            "model_name": "SpatioTemporalFloodNet3D",  # 使用3D时空洪水网络
            "experiment_dir": "3d_gnn_only",
            "description": "GNN-only基线 - 无时序网络，仅GNN空间建模+平均池化",
            "use_gnn": True,   # 关键参数：启用GNN
            "use_lstm": False, # 关键参数：禁用LSTM
            "hidden_channels": 16,  # 减小隐藏维度，防止过拟合
            "dropout": 0.2,  # 增加dropout，防止过拟合
        },
        "SpatioTemporalFloodNet3D_LSTM_GNN": {
            "model_name": "SpatioTemporalFloodNet3D",  # 使用3D时空洪水网络
            "experiment_dir": "3d_lstm_gnn",
            "description": "LSTM+GNN完整版 - LSTM时序建模+GNN空间交互+多种聚合",
            "use_gnn": True,   # 关键参数：启用GNN
            "use_lstm": True,  # 关键参数：启用LSTM
            "hidden_channels": 16,  # 减小隐藏维度，防止过拟合
            "dropout": 0.2,  # 增加dropout，防止过拟合
        },
        # === 🚀 新增：先空间后时间+Transformer架构 ===
        "SpatialFirstTemporalFloodNet3D": {
            "model_name": "SpatialFirstTemporalFloodNet3D",
            "experiment_dir": "3d_gnn_experiment_spatial_first_transformer",
            "description": "先空间后时间+Transformer - 空间优先+Transformer时序建模，数值稳定优化版",
            # 数值稳定的关键超参数
            "hidden_channels": 16,      # 减小隐藏维度，防止过拟合
            "num_hidden": 2,            # 进一步减少GNN层数，提高稳定性
            "n_heads": 2,               # 进一步减少Transformer头数，防止梯度爆炸
            "transformer_layers": 1,    # 进一步减少Transformer层数，提高稳定性
            "dropout": 0.2,             # 增加dropout比例，防止过拟合和数值不稳定
        }
    }
    
    # 验证模型名称
    if model_name not in model_configs:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(model_configs.keys())}")
    
    # 获取当前模型配置
    current_config = model_configs[model_name]
    
    # 设置实验目录
    base_dir = current_config["experiment_dir"]
    project_name = os.path.join(base_dir, f"songliao_3h_test_{test_mode}")

    print(f"📁 实验目录: {project_name}")
    print(f"🤖 模型: {model_name} ({current_config['description']})  | 模式: {test_mode}")

    # 使用有有效数据的流域ID（基于3h数据）
    example_gage_ids = [
        "songliao_20800900",
        "songliao_20810200", 
        "songliao_21100150",
        "songliao_21110150",
        "songliao_21401050",
        "songliao_21401550"
    ]

    # ==== 4维时空GNN参数配置 ====
    # 时间窗口设置
    warmup_length, hindcast_length, forecast_length = 90, 0, 12  # 90*3h历史, 1*3h预测

    # 特征设置
    var_t = ["rain"]  # 气象变量
    var_c = ["None"]  # 暂时不用流域特征
    station_cols = ["DRP"]  # 站点变量
    use_basin_features = False  # 先不用流域特征，简化模型
    
    # ==== 计算4维时空参数 ====
    seq_len = warmup_length + hindcast_length + forecast_length  # 时间序列长度: 91
    output_time = hindcast_length + forecast_length  # 输出时间长度: 1
    num_features = len([col for col in station_cols if col != "None"])  # 每个时间步的特征数: 1
    
    # 注意：4维时空GNN不需要展平，in_channels现在表示初始特征维度
    in_channels = num_features  # 每个时间步的特征维度: 1
    output_size = 1  # 输出径流量

    print(f"📊 seq_len={seq_len}, num_features={num_features}, output_time={output_time}")

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "dataset_id": "stationhydrodataset",
            "time_unit": ["3h"],
        },
        station_cfgs={
            # 站点数据配置 - 使用3h数据中实际存在的变量
            "station_cols": ["DRP"],  # TM=温度, 从站点数据中选择变量
            "station_rm_nan": True,
            "station_time_units": ["3h"],
            "station_scaler_type": "DapengScaler",
            "use_basin_features": use_basin_features,   # （决定是否使用流域平均的一些属性）或者不设置（默认为True），如果是false，则只使用站点特征
            # 邻接矩阵配置
            "edge_orientation": "bidirectional",  # 邻接矩阵的方向{upstream, downstream,bidirectional}默认是upstream
            "use_adjacency": True,
            "adjacency_src_col": "ID",
            "adjacency_dst_col": "NEXTDOWNID", 
            "adjacency_edge_attr_cols": ["dist_hdn", "elev_diff", "strm_slope"],
            "adjacency_weight_col": None,  # 不使用权重，返回边属性
            "return_edge_weight": False,
            },  
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 4维时空GNN模型配置
        model_name=current_config["model_name"],  # 使用4维时空GNN模型
        model_hyperparam={
            # ==== SpatioTemporalFloodNet参数 (根据具体模型调整) ====
            "num_features": num_features,     # 输入特征数 (SpatioTemporalFloodNet使用)
            "in_channels": in_channels,       # 每个时间步的特征维度 (3D/4D模型使用)
            "output_size": output_size,       # 输出特征数
            "seq_len": seq_len,               # 输入序列长度
            "output_time": output_time,       # 输出序列长度
            
            # ==== 根据模型类型调整超参数 ====
            "hidden_channels": current_config.get("hidden_channels", 32),  # 使用配置文件中的值或默认值
            "st_blocks": 1,                   # 减少时空块数量，简化模型
            "temporal_kind": "gru",           # 时间编码器类型 ("tcn" | "gru" | "transformer")
            "temporal_kernel_size": 3,        # 时间卷积核大小
            "temporal_layers": 1,             # 减少时间层数
            "spatial_kind": "gcn",            # 空间编码器类型 ("gcn" | "gat" | "gcn2")
            "aggregation": "mean",            # 聚合方式 ("mean" | "attention" | "multiscale")
            "dropout": current_config.get("dropout", 0.2),  # 增加默认dropout到0.2，防止过拟合
            
            # ==== 核心GNN参数 ====
            "num_hidden": current_config.get("num_hidden", 2),  # 使用配置文件中的值或默认值
            "param_sharing": False,          # 共享参数，减少模型容量
            "gnn_type": "gcn2",              # GNN类型 (3D模型使用)
            "aggregate_to_graph": True,     # True=图级输出(流域级径流), False=节点级输出
            
            # ==== LSTM-only vs GNN-only vs LSTM+GNN 对比参数 ====
            "use_gnn": current_config.get("use_gnn", True),  # 是否使用GNN，默认True
            "use_lstm": current_config.get("use_lstm", True), # 是否使用LSTM，默认True
            
            # # ==== 🔧 时间映射对比参数 ====  
            # "use_time_mlp": current_config.get("use_time_mlp", True),  # 是否使用MLP时间映射，默认True
            
            # ==== 🚀 新架构：Transformer专用参数 ====
            "n_heads": current_config.get("n_heads", 8),            # Transformer头数
            "transformer_layers": current_config.get("transformer_layers", 4),  # Transformer层数
                        
        },
        dataset="GNNDataset", scaler="DapengScaler", batch_size=4,  # 小batch适合时序专注模型
        warmup_length=warmup_length, hindcast_length=hindcast_length, forecast_length=forecast_length,
        min_time_unit="h", min_time_interval=3,
        var_t=var_t, var_c=var_c, var_out=["inflow", "flood_event"],
        train_mode=False,  # 训练模式 - 修复：改为True以启用训练
        
        train_epoch=40, save_epoch=1,   # 减少epoch，防止过拟合
        # train_period=["1980-01-01-02", "2016-12-31-23"],
        # #valid_period=["1980-01-01-02", "2019-12-31-23"],
        # valid_period=["2017-01-01-02", "2019-12-31-23"],
        # #test_period=["2020-01-01-02", "2024-12-31-23"],
        # test_period=["2020-01-01-02", "2024-12-31-23"],
        train_period=["1980-01-01-02", "2018-12-31-23"],
        #valid_period=["1980-01-01-02", "2019-12-31-23"],
        valid_period=["2019-01-01-02", "2024-12-31-23"],
        #test_period=["2020-01-01-02", "2024-12-31-23"],
        test_period=["2019-01-01-02", "2024-12-31-23"],        
        
        rolling=-1,  # 不使用滚动窗口
        model_loader={"load_way": "best"},  # 对应训练epoch
        evaluator={"eval_way": "floodevent"}, which_first_tensor="batch",
        # 优化器配置 - 数值稳定的设置
        opt="Adam",
        lr_scheduler={
            "lr": 0.0005,     # 降低学习率，防止训练不稳定
            "lr_factor": 0.7, # 更强的学习率衰减
            "lr_patience": 5,  # 增加patience
        },
        # 损失函数
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2,
            "flood_strategy": "weight",
            "device": [0],
        },
        # 其他配置
        num_workers=4,
        # pin_memory=True,
        # early_stopping=True,
        # patience=5,
        # calc_metrics=True,
        # continue_train=False,
    )

    # 更新配置
    update_cfg(config_data, args)
    
    # 手动添加空间测试参数到training_cfgs（确保参数正确传递）
    if "training_cfgs" not in config_data:
        config_data["training_cfgs"] = {}
    config_data["training_cfgs"]["spatial_test_mode"] = test_mode
    
    return config_data


def train_spatiotemporal_model(test_mode="normal", model_name="SpatioTemporalFloodNet3D_LSTM_only"):
    """训练时空洪水预测模型
    
    Args:
        test_mode: 测试模式，可选"normal", "sim1", "sim2", "sim3"
        model_name: 模型名称，可选"SpatioTemporalFloodNet3D_LSTM_only", "SpatioTemporalFloodNet3D_GNN_only", 
                   "SpatioTemporalFloodNet3D_LSTM_GNN", "SpatialFirstTemporalFloodNet3D"
    """
    try:
        # 创建配置
        cfg = create_spatiotemporal_config(test_mode=test_mode, model_name=model_name)

        # 开始训练/测试
        print(f"开始训练: {model_name} | 模式: {test_mode}")
        train_and_evaluate(cfg)

        print(f"完成: {model_name} | 模式: {test_mode}")

    except Exception as e:
        print(f"❌ 失败: {model_name} | 模式: {test_mode} | 错误: {e}")
        raise


# ===================== 核心运行入口（简化版） =====================
VALID_MODELS = [
    "SpatioTemporalFloodNet3D_LSTM_only",
    "SpatioTemporalFloodNet3D_GNN_only",
    "SpatioTemporalFloodNet3D_LSTM_GNN",
    "SpatialFirstTemporalFloodNet3D",
]
#VALID_SCENARIOS = ["normal", "sim1", "sim2", "sim3"]
VALID_SCENARIOS = ["sim1", "sim2", "sim3"]


def run_session(model_choice: str, scenario: str):
    """根据用户选择运行: 单模型 / 全部模型 × 单情景
    Args:
        model_choice: 模型名称或 'all'
        scenario: normal | sim1 | sim2 | sim3
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario 必须为 {VALID_SCENARIOS}")
    

    models = VALID_MODELS if model_choice == "all" else [model_choice]
    #scenario= VALID_SCENARIOS if scenario == "all" else [scenario]
    for m in models:
        if m not in VALID_MODELS:
            raise ValueError(f"未知模型: {m} (可选: {VALID_MODELS})")
        train_spatiotemporal_model(test_mode=scenario, model_name=m)

    print("\n✅ 运行完成: 模型=", model_choice, "| 情景=", scenario)


def main(model_choice: str | None = None, scenario: str | None = None):
    """主入口（可直接在脚本顶部编辑，也可命令行传参）
    优先级: 函数参数 > 命令行参数 > 默认块
    """
    # ---- 用户可编辑默认块（Notebook 或 直接运行脚本时修改这里） ----
    #DEFAULT_MODEL = "all"          # 或指定: SpatioTemporalFloodNet3D_LSTM_only
    DEFAULT_MODEL = "SpatioTemporalFloodNet3D_LSTM_GNN"          # 或指定: SpatioTemporalFloodNet3D_LSTM_only
    
    
    #DEFAULT_SCENARIO = "normal"     # normal | sim1 | sim2 | sim3
    DEFAULT_SCENARIO = "sim3"     # all | normal | sim1 | sim2 | sim3

    # ---- 命令行简单解析（保持兼容，不再提供复杂子命令） ----
    if model_choice is None or scenario is None:
        if len(sys.argv) >= 2:
            model_choice = sys.argv[1]
        if len(sys.argv) >= 3:
            scenario = sys.argv[2]

    model_choice = model_choice or DEFAULT_MODEL
    scenario = scenario or DEFAULT_SCENARIO

    print("=== 时空洪水训练会话 ===")
    print(f"模型选择: {model_choice}")
    print(f"情景: {scenario}")
    print("======================")

    run_session(model_choice, scenario)


if __name__ == "__main__":
    """使用说明:
    1) 直接修改 main() 中 DEFAULT_MODEL / DEFAULT_SCENARIO 后运行:
       python train_with_gnn_lstm_sim1.py
    2) 命令行指定:
       python train_with_gnn_lstm_sim1.py SpatioTemporalFloodNet3D_LSTM_only normal
       python train_with_gnn_lstm_sim1.py all sim2
    3) Notebook 中:
       from experiments.train_with_gnn_lstm_sim1 import main
       main("SpatioTemporalFloodNet3D_LSTM_GNN", "sim1")
    """
    main()
