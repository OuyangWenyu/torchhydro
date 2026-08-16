"""
Author: Yang Wang
Date: 2025-09-03 12:00:00
LastEditTime: 2025-09-03 12:00:00
Description: WDNE自适应图网络模型测试脚本 - 测试WDNEInspiredFloodNet / WDNEAdaptiveRoutingNet
FilePath: \\torchhydro\\experiments\\train_with_wdne.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import logging
import warnings
import sys

from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

logging.basicConfig(level=logging.WARNING)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore")


def _build_model_hyperparam(current_config, in_channels, output_size, seq_len, output_time,
                            num_static_features, use_basin_features=False):
    """根据模型类型构建对应的超参数字典"""
    model_type = current_config.get("model_type", "")

    common = {
        "output_size": output_size,
        "seq_len": seq_len,
        "output_time": output_time,
        "hidden_channels": current_config.get("hidden_channels", 32),
        "dropout": current_config.get("dropout", 0.3),
    }

    if model_type in ("wdne", "wdne_adaptive"):
        common.update({
            "in_channels": in_channels,  # 只含动态特征
            "model_type": model_type,
            "num_static_features": num_static_features,
            "structure_embedding_dim": current_config.get("structure_embedding_dim", 16),
            "structure_fusion": "gated",
            "topology_encoder_type": "mlp",
            "use_structure_embedding": current_config.get("use_structure_embedding", True),
            "local_lstm_layers": 1,
            "route_steps": current_config.get("route_steps", 4),
            "route_decay": current_config.get("route_decay", 0.5),
            "graph_output_mode": current_config.get("graph_output_mode", "residual"),
            "root_output_mode": current_config.get("root_output_mode", "root"),
        })
    else:
        # Baseline SpatioTemporalFloodNet3D 参数 — 需要全量输入维度
        total_in = in_channels + num_static_features if use_basin_features else in_channels
        common.update({
            "in_channels": total_in,
            "num_features": total_in,
            "st_blocks": 1,
            "temporal_kind": "gru",
            "temporal_kernel_size": 3,
            "temporal_layers": 1,
            "spatial_kind": "gcn",
            "aggregation": current_config.get("aggregation", "mean"),
            "num_hidden": current_config.get("num_hidden", 2),
            "param_sharing": False,
            "gnn_type": current_config.get("gnn_type", "gcn2"),
            "aggregate_to_graph": True,
            "use_gnn": current_config.get("use_gnn", True),
            "use_lstm": current_config.get("use_lstm", True),
            "n_heads": 8,
            "transformer_layers": 4,
        })

    return common


def create_wdne_config(test_mode="normal", model_name="WDNEAdaptiveRoutingNet", basins=None, use_static=False):
    """创建WDNE自适应图网络模型配置

    Args:
        test_mode: 测试模式 ("normal", "sim1", "sim2", "sim3")
        model_name: 模型名称
        basins: 流域列表, None=全部6个, 可指定单个如["songliao_21401550"]
        use_static: 是否加入流域静态属性
    """
    config_data = default_config_file()

    model_configs = {
        "Baseline_LSTM_only": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "baseline_lstm_only",
            "description": "Baseline LSTM-only - 纯时序建模, 无GNN",
            "model_type": "baseline",
            "hidden_channels": 32,
            "use_gnn": False,
            "use_lstm": True,
            "aggregation": "mean",
            "gnn_type": "gcn2",
            "num_hidden": 2,
            "dropout": 0.3,
        },
        "Baseline_LSTM_GNN": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "baseline_lstm_gnn",
            "description": "Baseline LSTM+GNN - 普通时空图卷积",
            "model_type": "baseline",
            "hidden_channels": 32,
            "use_gnn": True,
            "use_lstm": True,
            "aggregation": "mean",
            "gnn_type": "gcn2",
            "num_hidden": 2,
            "dropout": 0.3,
        },
        "WDNEInspiredFloodNet": {
            "model_name": "WDNEInspiredFloodNet",
            "experiment_dir": "wdne_inspired",
            "description": "WDNE结构感知 - 拓扑嵌入+局部LSTM+GCN",
            "model_type": "wdne",
            "hidden_channels": 32,
            "structure_embedding_dim": 16,
            "route_steps": 4,
            "route_decay": 0.5,
            "graph_output_mode": "residual",
            "root_output_mode": "root",
            "dropout": 0.3,
        },
        "WDNEAdaptiveRoutingNet": {
            "model_name": "WDNEAdaptiveRoutingNet",
            "experiment_dir": "wdne_adaptive",
            "description": "WDNE自适应路由",
            "model_type": "wdne_adaptive",
            "hidden_channels": 32,
            "structure_embedding_dim": 16,
            "route_steps": 4,
            "route_decay": 0.5,
            "graph_output_mode": "residual",
            "root_output_mode": "root",
            "dropout": 0.3,
        },
        # === 实验变体: 关结构嵌入 + mean pool ===
        "WDNE_NoStruct_Mean": {
            "model_name": "WDNEAdaptiveRoutingNet",
            "experiment_dir": "wdne_nostruct_mean",
            "description": "WDNE无结构嵌入+mean pool",
            "model_type": "wdne_adaptive",
            "hidden_channels": 32,
            "structure_embedding_dim": 4,
            "use_structure_embedding": False,
            "route_steps": 4,
            "route_decay": 0.5,
            "graph_output_mode": "residual",
            "root_output_mode": "mean",
            "dropout": 0.3,
        },
        "WDNE_Mean": {
            "model_name": "WDNEAdaptiveRoutingNet",
            "experiment_dir": "wdne_mean",
            "description": "WDNE自适应路由+mean pool输出",
            "model_type": "wdne_adaptive",
            "hidden_channels": 32,
            "structure_embedding_dim": 16,
            "route_steps": 4,
            "route_decay": 0.5,
            "graph_output_mode": "residual",
            "root_output_mode": "mean",
            "dropout": 0.3,
        },
        # === 实验2: 加静态属性(5流域, 去掉21100150) ===
        "LSTM_only_static": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "static_5basins_lstm_only",
            "description": "LSTM-only + 静态属性(5流域)",
            "model_type": "baseline",
            "hidden_channels": 32, "use_gnn": False, "use_lstm": True,
            "aggregation": "mean", "gnn_type": "gcn2", "num_hidden": 2, "dropout": 0.3,
        },
        "LSTM_GNN_static": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "static_5basins_lstm_gnn",
            "description": "LSTM+GNN + 静态属性(5流域)",
            "model_type": "baseline",
            "hidden_channels": 32, "use_gnn": True, "use_lstm": True,
            "aggregation": "mean", "gnn_type": "gcn2", "num_hidden": 2, "dropout": 0.3,
        },
        "WDNE_static": {
            "model_name": "WDNEAdaptiveRoutingNet",
            "experiment_dir": "static_5basins_wdne",
            "description": "WDNE + 静态属性(5流域)",
            "model_type": "wdne_adaptive",
            "hidden_channels": 32, "structure_embedding_dim": 16,
            "route_steps": 4, "route_decay": 0.5,
            "graph_output_mode": "residual", "root_output_mode": "root", "dropout": 0.3,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(model_configs.keys())}")

    # 静态属性实验: 自动设置 basins + use_static
    if model_name.endswith("_static") and basins is None:
        basins = [b for b in ["songliao_20800900","songliao_20810200","songliao_21110150","songliao_21401050","songliao_21401550"]]
        use_static = True

    current_config = model_configs[model_name]

    base_dir = current_config["experiment_dir"]
    project_name = os.path.join(base_dir, f"songliao_3h_{test_mode}")

    print(f"📁 实验结果将保存到: {project_name}")
    print(f"🤖 使用模型: {model_name} ({current_config['description']})")
    print(f"🎯 当前测试模式: {test_mode}")

    # 流域配置
    if basins is None:
        example_gage_ids = [
            "songliao_20800900",
            "songliao_20810200",
            "songliao_21100150",
            "songliao_21110150",
            "songliao_21401050",
            "songliao_21401550"
        ]
    else:
        example_gage_ids = basins

    # 时间窗口: 72*3h历史 → 预测12*3h (仅降雨特征，不宜过长)
    warmup_length, hindcast_length, forecast_length = 72, 0, 12
    seq_len = warmup_length + hindcast_length + forecast_length
    output_time = hindcast_length + forecast_length

    # 特征配置
    var_t = ["rain"]
    station_cols = ["DRP"]
    output_size = 1

    if use_static:
        # 5流域 + 静态属性(排除area极端值)
        var_c = ["ele_mt_sav", "slp_dg_sav", "for_pc_sse", "pre_mm_syr", "run_mm_syr"]
        use_basin_features = True
        num_features = len([c for c in station_cols if c != "None"])
        in_channels = num_features + len(var_t)  # = 1 + 1 = 2
        num_static_features = len(var_c)  # = 5
    else:
        var_c = ["None"]
        use_basin_features = False
        num_features = len([c for c in station_cols if c != "None"])
        in_channels = num_features  # = 1
        num_static_features = 0

    print(f"📊 参数: seq_len={seq_len}, in_channels={in_channels}, "
          f"num_static_features={num_static_features}, output_time={output_time}")

    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "stationhydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
        },
        station_cfgs={
            "station_cols": ["DRP"],
            "station_rm_nan": True,
            "station_time_units": ["3h"],
            "station_scaler_type": "DapengScaler",
            "use_basin_features": use_basin_features,
            "use_adjacency": True,
            "edge_orientation": "downstream",  # WDNE推荐downstream (河流方向)
            "adjacency_src_col": "ID",
            "adjacency_dst_col": "NEXTDOWNID",
            "adjacency_edge_attr_cols": ["dist_hdn", "elev_diff", "strm_slope"],
            "adjacency_weight_col": None,  # 不指定权重列，所有边权重默认为1
            "return_edge_weight": True,
        },
        gage_id=example_gage_ids,
        model_name=current_config["model_name"],
        model_hyperparam=_build_model_hyperparam(current_config, in_channels, output_size,
                                                  seq_len, output_time, num_static_features,
                                                  use_basin_features),
        dataset="GNNDataset", scaler="DapengScaler", batch_size=2,
        warmup_length=warmup_length, hindcast_length=hindcast_length, forecast_length=forecast_length,
        min_time_unit="h", min_time_interval=3,
        var_t=var_t, var_c=var_c, var_out=["inflow", "flood_event"],
        train_mode=True,
        train_epoch=60, save_epoch=1,
        train_period=["1980-01-01-02", "2014-12-31-23"],
        valid_period=["2015-01-01-02", "2018-12-31-23"],
        test_period=["2019-01-01-02", "2024-12-31-23"],
        rolling=-1,
        model_loader={"load_way": "best"},
        evaluator={"eval_way": "floodevent"}, which_first_tensor="batch",
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.5,
            "lr_patience": 8,
        },
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2,
            "flood_strategy": "weight",
            "device": [0],
        },
        early_stopping=True,
        patience=10,
        calc_metrics=True,
        continue_train=False,
    )

    update_cfg(config_data, args)

    if "training_cfgs" not in config_data:
        config_data["training_cfgs"] = {}
    config_data["training_cfgs"]["spatial_test_mode"] = test_mode

    return config_data


def train_wdne_model(test_mode="normal", model_name="WDNEAdaptiveRoutingNet"):
    """训练WDNE自适应图网络模型"""
    try:
        config_data = create_wdne_config(test_mode=test_mode, model_name=model_name)
        print(f"开始 {model_name} 训练...")
        train_and_evaluate(config_data)
        print(f"{model_name} 训练完成!")
    except Exception as e:
        print(f"{model_name} 训练出错: {e}")
        raise


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        test_mode = sys.argv[2] if len(sys.argv) > 2 else "normal"

        valid_models = ["Baseline_LSTM_only", "Baseline_LSTM_GNN", "WDNEInspiredFloodNet", "WDNEAdaptiveRoutingNet",
                       "WDNE_NoStruct_Mean", "WDNE_Mean", "LSTM_only_static", "LSTM_GNN_static", "WDNE_static"]

        if model_name in valid_models:
            train_wdne_model(test_mode=test_mode, model_name=model_name)
        elif model_name.lower() == "all":
            for model in valid_models:
                try:
                    print(f"\n{'='*50}")
                    print(f"开始测试模型: {model}")
                    train_wdne_model(test_mode=test_mode, model_name=model)
                except Exception as e:
                    print(f"❌ {model} 测试失败: {e}")
                    continue
        else:
            print(f"未知模型: {model_name}, 可选: {valid_models}")
    else:
        # 默认运行 WDNEAdaptiveRoutingNet
        print("📋 未指定参数，默认运行 WDNEAdaptiveRoutingNet...")
        train_wdne_model(test_mode="normal", model_name="WDNEAdaptiveRoutingNet")


if __name__ == "__main__":
    """
    WDNE自适应图网络模型测试脚本使用说明:

    1. 单个模型测试:
       python train_with_wdne.py WDNEAdaptiveRoutingNet normal
       python train_with_wdne.py WDNEInspiredFloodNet normal

    2. 全部模型对比:
       python train_with_wdne.py all normal

    模型说明:
    - WDNEInspiredFloodNet: 结构感知洪水预测网络
      拓扑嵌入(河流角色编码) + 局部LSTM(节点级时序) + GCN图传播(空间交互)
      特点: 显式建模河流网络拓扑结构，用拓扑特征+静态属性融合编码每个节点的结构角色

    - WDNEAdaptiveRoutingNet: 自适应路由洪水预测网络
      在WDNE基础上，将静态GCN替换为动态自适应边权重:
      1. 根据当前气象条件(driver_context)和局部预测(local_prior)动态计算边权重
      2. 使用目标节点softmax归一化，确保每个节点的入边权重和为1
      3. 门控更新机制，控制新信息与旧状态的比例
      特点: 能够学习事件条件化的下游传播模式 (如洪水vs干旱时传播路径不同)

    关键超参数:
    - structure_embedding_dim: 结构嵌入维度 (河流拓扑角色的编码维度)
    - route_steps: 图传播步数 (信息沿河流传递的跳数)
    - route_decay: 传播衰减系数 (远距离上游影响随距离衰减)
    - graph_output_mode: "residual"(残差修正) / "blend_residual"(混合残差) / "main"(纯图输出)
    """
    main()
