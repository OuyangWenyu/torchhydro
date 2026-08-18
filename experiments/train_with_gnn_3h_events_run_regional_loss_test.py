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

def create_4d_gnn_config(test_mode="normal", model_name="BasicGCN"):
    """创建4维时空GNN配置
    
    Args:
        test_mode: 测试模式
            - "normal": 正常测试
            - "sim1": 上游降水敏感性测试（上游降水x2）
            - "sim2": 中游降水敏感性测试（中游降水x2）
            - "sim3": 下游降水敏感性测试（下游降水x2）
        model_name: 模型名称
            - "BasicGCN": 基础GCN模型（4维时空输入，基础图卷积）
            - "STGCN": 时空图卷积网络（时序卷积+图卷积）
            - "ASTGCN": 注意力时空图卷积网络（时序注意力+空间注意力）
            - "GraphWaveNet": 图波网络（扩张卷积+自适应图学习）
    """
    # 获取默认配置
    config_data = default_config_file()

    # 4个核心模型配置
    model_configs = {
        "BasicGCN": {
            "model_name": "GCN",  # 使用别名映射到BasicGCNModel
            "experiment_dir": "4d_gnn_experiment_basic_gcn",
            "description": "基础GCN - 4维时空输入，基础图卷积"
        },
        "STGCN": {
            "model_name": "STGCN",  # 使用别名映射到STGCNModel
            "experiment_dir": "4d_gnn_experiment_stgcn", 
            "description": "时空图卷积网络 - 时序卷积+图卷积"
        },
        "CrossModalGNN": {
            "model_name": "CrossModalGNN",  # 使用别名映射到ASTGCNModel
            "experiment_dir": "4d_gnn_experiment_CrossModalGNN",
            "description": "注意力时空图卷积网络 - 时序注意力+空间注意力"
        },
        "GraphWaveNet": {
            "model_name": "GraphWaveNet",  # 使用别名映射到GraphWaveNetModel
            "experiment_dir": "4d_gnn_experiment_graphwavenet",
            "description": "图波网络 - 扩张卷积+自适应图学习"
        },
        "SpatioTemporalFloodNet": {
            "model_name": "SpatioTemporalFloodNet",  # 使用新的SpatioTemporalFloodNet模型
            "experiment_dir": "4d_gnn_experiment_spatiotemporal_flood_net",
            "description": "时空洪水预测网络 - 可配置时空骨干和聚合"
        },
        "FloodGCN4D": {
            "model_name": "FloodGCN4D",  # 使用新的4D batch GCN模型
            "experiment_dir": "4d_gnn_experiment_flood_gcn_4d",
            "description": "4D洪水预测GCN - 基于gnn copy 4.py的成功架构"
        },
        "FloodGCN3D": {
            "model_name": "FloodGCN3D",  # 使用新的3D batch GCN模型
            "experiment_dir": "3d_gnn_experiment_flood_gcn_3d",
            "description": "3D洪水预测GCN - 基于3D batch处理和多种聚合方式"
        },
        "MeanAggregationFloodNet": {
            "model_name": "MeanAggregationFloodNet",  # 平均聚合版本
            "experiment_dir": "3d_gnn_experiment_mean_aggregation",
            "description": "3D洪水预测GCN - 平均聚合版本"
        },
        "AttentionAggregationFloodNet": {
            "model_name": "AttentionAggregationFloodNet",  # 注意力聚合版本
            "experiment_dir": "3d_gnn_experiment_attention_aggregation",
            "description": "3D洪水预测GCN - 注意力聚合版本"
        },
        "MultiScaleAggregationFloodNet": {
            "model_name": "MultiScaleAggregationFloodNet",  # 多尺度聚合版本
            "experiment_dir": "3d_gnn_experiment_multiscale_aggregation",
            "description": "3D洪水预测GCN - 多尺度聚合版本"
        },
        "SpatioTemporalFloodNet3D_LSTM_only": {
            "model_name": "SpatioTemporalFloodNet3D",  # 使用3D时空洪水网络
            "experiment_dir": "3d_gnn_experiment_lstm_only",
            "description": "LSTM-only基线 - 无图网络，仅LSTM时序建模+平均池化",
            "use_gnn": False  # 关键参数：禁用GNN
        },
        "SpatioTemporalFloodNet3D_LSTM_GNN": {
            "model_name": "SpatioTemporalFloodNet3D",  # 使用3D时空洪水网络
            "experiment_dir": "3d_gnn_experiment_lstm_gnn",
            "description": "LSTM+GNN完整版 - LSTM时序建模+GNN空间交互+多种聚合",
            "use_gnn": True   # 关键参数：启用GNN
        },
        # === 🔧 新增：时间映射对比实验 ===
        "SpatioTemporalFloodNet3D_MLP_Time": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "3d_gnn_experiment_mlp_time",
            "description": "LSTM+GNN+MLP时间映射 - 使用神经网络进行时间维度映射",
            "use_gnn": True,
            "use_time_mlp": True  # 使用MLP进行时间映射
        },
        "SpatioTemporalFloodNet3D_Truncate_Time": {
            "model_name": "SpatioTemporalFloodNet3D",
            "experiment_dir": "3d_gnn_experiment_truncate_time", 
            "description": "LSTM+GNN+简单截断 - 直接截断时间序列进行映射",
            "use_gnn": True,
            "use_time_mlp": False  # 简单截断时间序列
        }
    }
    
    # 验证模型名称
    if model_name not in model_configs:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(model_configs.keys())}")
    
    # 获取当前模型配置
    current_config = model_configs[model_name]
    
    # 设置实验目录
    base_dir = current_config["experiment_dir"]
    if test_mode == "normal":
        project_name = os.path.join(base_dir, "songliao_3h_test_normal")
    else:
        project_name = os.path.join(base_dir, f"songliao_3h_test_{test_mode}")

    print(f"📁 实验结果将保存到: {project_name}")
    print(f"🤖 使用模型: {model_name} ({current_config['description']})")
    print(f"🎯 当前测试模式: {test_mode}")

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
    warmup_length, hindcast_length, forecast_length = 160, 0, 12  # 160*3h历史, 1*3h预测

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

    print(f"📊 4维时空参数: seq_len={seq_len}, num_features={num_features}, output_time={output_time}")

    # 🚀 训练时间优化配置（统一超参数以便公平对比）
    # 说明：为进行跨模型公平对比，将核心可调节容量超参数统一。
    # 如果后续需要恢复模型特定配置，可将 unify_hparams 置为 False。
    unify_hparams = True
    if unify_hparams:
        fast_training = {
            "hidden_channels": 32,     # 统一隐藏维度
            "num_hidden": 2,           # 统一MLP/投影等隐藏层数
            "dropout": 0.2,            # 统一通用dropout比例
            "num_layers": 2,           # 统一层数（如适用）
            "time_mapping_method": "last",  # 时间映射统一（部分模型可能忽略）
            "tcn_dropout": 0.05,       # 统一时序模块dropout
            "output_head_dropout": 0.0,# 统一输出头dropout
            # 给非 STGCN 模型可选的默认 norm_type，占位；实际仍由下方逻辑区分
            "norm_type": "batch",
        }
        print("✅ 已启用统一超参数模式：hidden=32, num_hidden=2, dropout=0.2, num_layers=2, lr=3e-4")
    else:
        # 原始分支（目前不启用，只作占位说明）
        if model_name == "STGCN":
            fast_training = {
                "hidden_channels": 32,
                "num_hidden": 2,
                "dropout": 0.2,
                "num_layers": 2,
                "time_mapping_method": "last",
                "tcn_dropout": 0.05,
                "output_head_dropout": 0.0,
            }
        else:
            fast_training = {
                "hidden_channels": current_config.get("hidden_channels", 64),
                "num_hidden": current_config.get("num_hidden", 2),
                "dropout": current_config.get("dropout", 0.3),
                "num_layers": 2,
            }

    # ====== 📌 自动复用 normal 基线统计文件 (DapengScaler 需要) ======
    # 在纯测试模式 (train_mode=False) 且使用敏感性场景(sim1/2/3) 时，尝试加载 normal 训练产生的 dapengscaler_stat.json
    # 目录规则: results/<base_dir>/songliao_3h_test_normal/dapengscaler_stat.json
    reuse_stat_file = False
    stat_dict_file_path = None
    # 默认模型加载策略；normal 训练时使用 latest，敏感性评估时改为 pth
    model_loader_cfg = {"load_way": "latest"}
    # 注意：此处的 train_mode 由下面 cmd 参数决定，这里根据我们即将传入的值进行判断
    intended_train_mode = False  # 当前脚本设置为测试模式
    if not intended_train_mode and test_mode != "normal":
        baseline_case_dir = os.path.join(os.getcwd(), "results", base_dir, "songliao_3h_test_normal")
        candidate_stat = os.path.join(baseline_case_dir, "dapengscaler_stat.json")
        if os.path.isfile(candidate_stat):
            stat_dict_file_path = candidate_stat
            reuse_stat_file = True
            print(f"🔁 发现基线统计文件，将复用: {candidate_stat}")
            # === 追加：尝试复用 normal 训练得到的模型权重 ===
            baseline_weight = None
            if os.path.isdir(baseline_case_dir):
                try:
                    # 优先按常见 epoch 列表回溯查找（最高优先级靠前）
                    preferred_epochs = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
                    for ep in preferred_epochs:
                        cand = os.path.join(baseline_case_dir, f"model_Ep{ep}.pth")
                        if os.path.isfile(cand):
                            baseline_weight = cand
                            break
                    # 若上面未找到，按 model_Ep*.pth 解析最大 epoch
                    if baseline_weight is None:
                        from glob import glob
                        import re
                        ep_files = glob(os.path.join(baseline_case_dir, "model_Ep*.pth"))
                        if ep_files:
                            def _ep_num(fp):
                                m = re.search(r"model_Ep(\d+)\.pth", fp)
                                return int(m.group(1)) if m else -1
                            ep_files.sort(key=_ep_num)
                            baseline_weight = ep_files[-1]
                    # 再退一步：时间戳 *_model.pth （如 29_September_xxx_model.pth）
                    if baseline_weight is None:
                        ts_files = [f for f in os.listdir(baseline_case_dir) if f.endswith("_model.pth")]
                        if ts_files:
                            ts_files.sort(key=lambda f: os.path.getctime(os.path.join(baseline_case_dir, f)))
                            baseline_weight = os.path.join(baseline_case_dir, ts_files[-1])
                except Exception as _e:  # noqa: F841
                    baseline_weight = None
            if baseline_weight and os.path.isfile(baseline_weight):
                model_loader_cfg = {"load_way": "pth", "pth_path": baseline_weight}
                print(f"🔗 将复用 normal 训练权重进行敏感性评估: {baseline_weight}")
            else:
                print("⚠️ 未找到可复用的 normal 权重文件，将继续尝试 'latest'，若 sim 目录无权重会失败。")
        else:
            print("⚠️ 未找到基线 normal 统计文件: "
                  f"{candidate_stat}\n   -> 请先运行 normal 训练 (train_mode=True, test_mode=normal) 以生成统计文件，"
                  "否则 DapengScaler 会因缺少 dapengscaler_stat.json 报错。")

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "dataset_id": "stationhydrodataset",
            "source_path": "E:/data/ClassC/songliaorrevent",
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
            
            # ==== 根据模型类型调整超参数 - 标准版本 ====
            "hidden_channels": fast_training["hidden_channels"],
            "num_layers": fast_training["num_layers"],  # STGCN层数控制
            # 显式传递时间映射策略
            "time_mapping_method": fast_training.get("time_mapping_method", "pool"),
            # === 新增：归一化类型（防止BatchNorm在batch=1时导致输出恒定）===
            # 可选：batch | layer | group | none，默认 group 更稳健
            "norm_type": "group" if model_name == "STGCN" else fast_training.get("norm_type", "batch"),
            # 传递新的dropout细化参数到STGCN
            "tcn_dropout": fast_training.get("tcn_dropout", 0.1),
            "output_head_dropout": fast_training.get("output_head_dropout", 0.2),
            # 移除efficient参数，使用模型原始复杂度
            "st_blocks": 1,                   # 减少时空块数量，简化模型
            "temporal_kind": "gru",           # 时间编码器类型 ("tcn" | "gru" | "transformer")
            "temporal_kernel_size": 3,        # 时间卷积核大小
            "temporal_layers": 1,             # 减少时间层数
            "spatial_kind": "gcn",            # 空间编码器类型 ("gcn" | "gat" | "gcn2")
            "aggregation": "mean",            # 聚合方式 ("mean" | "attention" | "multiscale")
            "dropout": fast_training["dropout"],
            
            # ==== 核心GNN参数 ====
            "num_hidden": fast_training["num_hidden"],
            "param_sharing": False,          # 共享参数，减少模型容量
            "gnn_type": "gcn",              # GNN类型 (3D模型使用)
            "aggregate_to_graph": True,     # True=图级输出(流域级径流), False=节点级输出
            
            # ==== LSTM-only vs LSTM+GNN 对比参数 ====
            "use_gnn": current_config.get("use_gnn", True),  # 是否使用GNN，默认True
            
            # ==== 🔧 时间映射对比参数 ====  
            "use_time_mlp": current_config.get("use_time_mlp", True),  # 是否使用MLP时间映射，默认True
            
            # ==== 🚀 新架构：Transformer专用参数 ====
            "n_heads": current_config.get("n_heads", 8),            # Transformer头数
            "transformer_layers": current_config.get("transformer_layers", 4),  # Transformer层数
            
            # ==== 🎯 STGCN图聚合配置（顶层，兼容旧实现）====
            "enable_graph_aggregation": True if model_name == "STGCN" else False,
            # 顶层聚合方式：为防止过度放大或抹平，先与 data_feature 内保持一致（都用 mean）
            "aggregation_method": "mean" if model_name == "STGCN" else "mean",
            "final_output_dim": output_size,
            "truncate_time": output_time,

            # ==== 🐞 STGCN 专用 data_feature：用于新版 STGCNModel 调试与聚合控制 ====
            # 仅在 STGCN 模型时会被使用，其他模型会忽略
            "data_feature": (
                {
                    # 图级聚合与投影
                    "enable_graph_aggregation": True,
                    # 与顶层保持一致，避免前后冲突；后续若测试注意力/求和影响，可单独改此处
                    "aggregation_method": "mean",
                    "final_output_dim": output_size,
                    "truncate_time": output_time,
                    # 幅值快速校准（仿射）
                    "enable_affine_calibration": True,
                    # 一次拼大图 + 过滤跨图边
                    "process_all_in_one": True,
                    "filter_cross_graph_edges": True,
                    # 调试开关：打印形状与统计、并返回节点级输出方便定位
                    # 如需排查“输出恒定”问题，可把 debug_shapes 设为 True
                    "debug_shapes": False,
                    "return_node_outputs": False,
                    # 恢复图级线性投影与原输出头，便于幅度校准
                    "disable_output_projection": False,
                    "disable_output_head_bn_dropout": False,
                }
                if model_name == "STGCN" else None
            ),
                        
        },
        dataset="GNNDataset", scaler="DapengScaler", 
        batch_size=4,  
        warmup_length=warmup_length, hindcast_length=hindcast_length, forecast_length=forecast_length,
        min_time_unit="h", min_time_interval=3,
        var_t=var_t, var_c=var_c, var_out=["inflow", "flood_event"],

        train_mode=True,  # 训练模式
        # train_mode=False,  # 测试模式：仅评估，不重新训练

        stat_dict_file=stat_dict_file_path,
        
        train_epoch=60,  # 延长训练以提高拟合
        save_epoch=5,   # 🔧 每5轮保存一次，监控训练进展
        # train_period=["1980-01-01-02", "2016-12-31-23"],
        # #valid_period=["1980-01-01-02", "2019-12-31-23"],
        # valid_period=["2017-01-01-02", "2019-12-31-23"],
        # #test_period=["2020-01-01-02", "2024-12-31-23"],
        # test_period=["2020-01-01-02", "2024-12-31-23"],
        train_period=["1980-01-01-02", "2013-12-31-23"],
        #valid_period=["1980-01-01-02", "2019-12-31-23"],
        valid_period=["2014-01-01-02", "2019-12-31-23"],
        #test_period=["2020-01-01-02", "2024-12-31-23"],
        test_period=["2020-01-01-02", "2024-12-31-23"],        
        
        rolling=-1,  # 不使用滚动窗口

        model_loader=model_loader_cfg,  # 根据是否是敏感性评估动态决定加载策略
        
        evaluator={"eval_way": "floodevent"}, which_first_tensor="batch",
        # 优化器配置 - 防止过拟合的设置
        opt="Adam",
        lr_scheduler={
            # 统一学习率与调整策略，避免不同 lr 带来混淆
            "lr": 0.0003,
            "lr_factor": 0.5,
            # 统一 patience（若以后重启 early_stopping 可复用）
            "lr_patience": 3,
        },
        # 损失函数
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",  # 保持MSE，对预测精度敏感
            "flood_weight": 2.0,     # 🔧 提高洪水权重，让模型更关注洪峰预测
            "flood_strategy": "weight",
            "device": [0],
        },
        # 其他配置 - 🚀 训练加速优化
        num_workers=8 if model_name == "STGCN" else 4,  # 增加数据加载进程数
        pin_memory=True,  # 启用pin_memory加速GPU数据传输
        early_stopping=False,  # 🚫 取消早停，让模型训练完整个epoch
        # patience=8 if model_name == "STGCN" else 10,    # 不再需要patience参数
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


def train_4d_gnn_model(test_mode="normal", model_name="BasicGCN"):
    """训练4维时空GNN模型
    
    Args:
        test_mode: 测试模式，可选"normal", "sim1", "sim2", "sim3"
        model_name: 模型名称，可选"BasicGCN", "STGCN", "ASTGCN", "GraphWaveNet"
    """
    try:
        # 创建配置
        config_data = create_4d_gnn_config(test_mode=test_mode, model_name=model_name)

        # 开始训练/测试
        print(f"开始{model_name}-{test_mode}模式的4维时空GNN模型测试...")
        train_and_evaluate(config_data)

        print(f"{model_name}-{test_mode}模式完成！")

    except Exception as e:
        print(f"{model_name}-{test_mode}模式过程中出现错误: {e}")
        raise


def main():
    """主函数，支持多种运行模式"""
    
    # 代码内设置参数 - 无需命令行传参
    model_name = "FloodGCN3D"  
    test_mode = "normal"  # 默认正常测试模式
    
    # 检查命令行参数（保留兼容性，但优先使用代码内设置）
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        #model_name='compare_lstm_gnn'
        test_mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
        
        # 验证模型名称
        valid_models = ["BasicGCN", "STGCN", "ASTGCN", "GraphWaveNet", "SpatioTemporalFloodNet", "FloodGCN4D", 
                       "FloodGCN3D", "MeanAggregationFloodNet", "AttentionAggregationFloodNet", "MultiScaleAggregationFloodNet",
                          "SpatioTemporalFloodNet3D_LSTM_only", "SpatioTemporalFloodNet3D_LSTM_GNN",
                            "SpatioTemporalFloodNet3D_MLP_Time",
                       "SpatioTemporalFloodNet3D_Truncate_Time"]
        
        # 特殊处理对比实验命令
        if model_name.lower() == "compare_lstm_gnn":
            print("🔬 开始 LSTM-only vs LSTM+GNN 对比实验")
            print("="*60)
            
            # 首先运行LSTM-only基线
            try:
                print("\n📊 步骤1: 训练LSTM-only基线模型")
                print("-"*40)
                train_4d_gnn_model(test_mode="normal", model_name="SpatioTemporalFloodNet3D_LSTM_only")
                print("✅ LSTM-only基线训练完成")
            except Exception as e:
                print(f"❌ LSTM-only基线训练失败: {e}")
            
            # 然后运行LSTM+GNN完整版
            try:
                print("\n📊 步骤2: 训练LSTM+GNN完整模型")
                print("-"*40)
                train_4d_gnn_model(test_mode="normal", model_name="SpatioTemporalFloodNet3D_LSTM_GNN")
                print("✅ LSTM+GNN完整版训练完成")
            except Exception as e:
                print(f"❌ LSTM+GNN完整版训练失败: {e}")
            
            print("\n🎯 LSTM-only vs LSTM+GNN 对比实验完成！")
            print("📂 结果保存在以下目录：")
            print("   - LSTM-only基线: results/3d_gnn_experiment_lstm_only/")
            print("   - LSTM+GNN完整版: results/3d_gnn_experiment_lstm_gnn/")
            print("💡 请对比两个模型的训练/验证损失，分析图网络空间特性提取的有效性")
            return
        
        # 🚀 新增：三种架构全面对比实验
        elif model_name.lower() == "compare_three_architectures":
            print("🚀 开始三种时空架构全面对比实验")
            print("="*70)
            print("📋 对比架构:")
            print("   1. LSTM-only基线: 仅LSTM时序建模+平均池化")
            print("   2. Time→Space: LSTM时序建模→GNN空间交互")  
            print("   3. Space→Time: GNN空间交互→Transformer时序建模")
            print("="*70)
            
            architectures = [
                ("SpatioTemporalFloodNet3D_LSTM_only", "🔷 LSTM-only基线"),
                ("SpatioTemporalFloodNet3D_LSTM_GNN", "🔶 Time→Space (LSTM→GNN)"),
                ("SpatialFirstTemporalFloodNet3D", "🔺 Space→Time (GNN→Transformer)"),
            ]
            
            results_summary = []
            
            for model_key, description in architectures:
                try:
                    print(f"\n📊 正在训练: {description}")
                    print("-"*50)
                    train_4d_gnn_model(test_mode="normal", model_name=model_key)
                    print(f"✅ {description} 训练完成")
                    results_summary.append((description, "✅ 成功"))
                except Exception as e:
                    print(f"❌ {description} 训练失败: {e}")
                    results_summary.append((description, f"❌ 失败: {e}"))
                    continue
            
            print(f"\n🎉 三种时空架构对比实验完成！")
            print("="*70)
            print("📊 实验结果总结:")
            for desc, status in results_summary:
                print(f"   {desc}: {status}")
            
            print("\n📂 结果保存目录:")
            print("   1. LSTM-only基线: results/3d_gnn_experiment_lstm_only/")
            print("   2. Time→Space: results/3d_gnn_experiment_lstm_gnn/")
            print("   3. Space→Time: results/3d_gnn_experiment_spatial_first_transformer/")
            
            print("\n💡 分析建议:")
            print("   - 对比训练/验证损失收敛情况")
            print("   - 分析测试集泛化能力")
            print("   - 关注Space→Time架构是否有更好的空间学习能力")
            print("   - 评估Transformer vs LSTM的时序建模效果")
            return
        
        if model_name in valid_models:
            # 验证测试模式
            if test_mode in ["normal", "sim1", "sim2", "sim3"]:
                # 运行指定的模型和测试模式
                train_4d_gnn_model(test_mode=test_mode, model_name=model_name)
            else:
                print(f"未知测试模式: {test_mode}")
                print("可用测试模式: normal, sim1, sim2, sim3")
        elif model_name.lower() == "all":
            # 运行所有模型的对比测试
            test_mode = test_mode if test_mode in ["normal", "sim1", "sim2", "sim3"] else "normal"
            print(f"🚀 开始所有模型的对比测试 (测试模式: {test_mode})")
            
            for model in valid_models:
                try:
                    print(f"\n{'='*50}")
                    print(f"开始测试模型: {model}")
                    print(f"{'='*50}")
                    train_4d_gnn_model(test_mode=test_mode, model_name=model)
                    print(f"✅ {model} 模型测试完成")
                except Exception as e:
                    print(f"❌ {model} 模型测试失败: {e}")
                    continue
            
            print(f"\n🎉 所有模型对比测试完成！")
        else:
            print(f"未知模型: {model_name}")
            print(f"可用模型: {', '.join(valid_models)}")
            print("或使用 'all' 来测试所有模型")
            print("或使用 'compare_lstm_gnn' 来进行LSTM-only vs LSTM+GNN对比实验")
    else:
        # 使用代码内设置的默认参数
        print(f"📋 使用代码内设置参数: {model_name} - {test_mode}")
        print(f"🚀 开始训练LSTM+GNN完整架构")
        
        # 直接运行指定的模型
        train_4d_gnn_model(test_mode=test_mode, model_name=model_name)


if __name__ == "__main__":
    """
    4维时空GNN模型测试脚本使用说明：
    
    1. 单个模型测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py BasicGCN normal      # 基础GCN模型
       python train_with_gnn_3h_events_run_regional_loss_test.py STGCN normal        # 时空图卷积网络
       python train_with_gnn_3h_events_run_regional_loss_test.py ASTGCN normal       # 注意力时空图卷积网络
       python train_with_gnn_3h_events_run_regional_loss_test.py GraphWaveNet normal # 图波网络
       python train_with_gnn_3h_events_run_regional_loss_test.py SpatioTemporalFloodNet normal # 时空洪水预测网络
       python train_with_gnn_3h_events_run_regional_loss_test.py FloodGCN4D normal # 4D洪水预测GCN(基于copy4架构)
       python train_with_gnn_3h_events_run_regional_loss_test.py FloodGCN3D normal # 3D洪水预测GCN(基于3D batch处理)
       python train_with_gnn_3h_events_run_regional_loss_test.py MeanAggregationFloodNet normal # 平均聚合版本
       python train_with_gnn_3h_events_run_regional_loss_test.py AttentionAggregationFloodNet normal # 注意力聚合版本
       python train_with_gnn_3h_events_run_regional_loss_test.py MultiScaleAggregationFloodNet normal # 多尺度聚合版本
       
    2. 🚀 三种时空架构全面对比实验（最新推荐）：
       python train_with_gnn_3h_events_run_regional_loss_test.py compare_three_architectures normal  # 三种架构对比
       python train_with_gnn_3h_events_run_regional_loss_test.py SpatialFirstTemporalFloodNet3D normal  # 新架构：先空间后时间+Transformer
       
    3. LSTM-only vs LSTM+GNN 对比实验：
       python train_with_gnn_3h_events_run_regional_loss_test.py compare_lstm_gnn normal  # 自动运行对比实验
       python train_with_gnn_3h_events_run_regional_loss_test.py SpatioTemporalFloodNet3D_LSTM_only normal  # 仅LSTM基线
       python train_with_gnn_3h_events_run_regional_loss_test.py SpatioTemporalFloodNet3D_LSTM_GNN normal   # LSTM+GNN完整版
       
    4. 所有模型对比测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py all normal          # 运行所有模型的对比测试
       
    5. 空间敏感性测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py STGCN sim1         # 上游降水x2
       python train_with_gnn_3h_events_run_regional_loss_test.py ASTGCN sim2        # 中游降水x2
       python train_with_gnn_3h_events_run_regional_loss_test.py GraphWaveNet sim3  # 下游降水x2
       
    6. 批量空间敏感性测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py all sim1           # 所有模型的上游敏感性测试
    
    模型说明：
    - BasicGCN: 基础GCN模型，支持4维时空输入但只使用基本图卷积
    - STGCN: 时空图卷积网络，结合时序卷积和图卷积进行时空建模
    - ASTGCN: 注意力时空图卷积网络，结合时序注意力和空间注意力
    - GraphWaveNet: 图波网络，结合扩张卷积和自适应图学习
    - SpatioTemporalFloodNet: 时空洪水预测网络，支持可配置的时空骨干和聚合方式
    - FloodGCN4D: 4D洪水预测GCN，基于gnn copy 4.py的成功架构，强化时序建模+轻量空间建模
    - FloodGCN3D: 3D洪水预测GCN，基于3D batch处理，学习gnn_st_flood_simple的多种聚合方式
    - MeanAggregationFloodNet: 平均聚合版本的3D洪水预测GCN，简单有效
    - AttentionAggregationFloodNet: 注意力聚合版本的3D洪水预测GCN，智能节点选择
    - MultiScaleAggregationFloodNet: 多尺度聚合版本的3D洪水预测GCN，综合多种统计信息
    - 🚀 SpatialFirstTemporalFloodNet3D: 先空间后时间+Transformer架构，防过拟合优化版

    🔬 三种时空建模架构对比：
    1. LSTM-only: 纯时序建模，无空间交互，作为基线对照
    2. Time→Space: 先LSTM时序建模，再GNN空间交互（传统方法）
    3. Space→Time: 先GNN空间交互，再Transformer时序建模（新方法）

    
    测试模式说明：
    - normal: 正常测试
    - sim1: 上游降水敏感性测试（上游站点降水x2）
    - sim2: 中游降水敏感性测试（中游站点降水x2）
    - sim3: 下游降水敏感性测试（下游站点降水x2）
    
    结果目录：
    - BasicGCN: results/4d_gnn_experiment_basic_gcn/
    - STGCN: results/4d_gnn_experiment_stgcn/
    - ASTGCN: results/4d_gnn_experiment_astgcn/
    - GraphWaveNet: results/4d_gnn_experiment_graphwavenet/
    - SpatioTemporalFloodNet: results/4d_gnn_experiment_spatiotemporal_flood_net/
    - FloodGCN4D: results/4d_gnn_experiment_flood_gcn_4d/
    - FloodGCN3D: results/3d_gnn_experiment_flood_gcn_3d/
    - MeanAggregationFloodNet: results/3d_gnn_experiment_mean_aggregation/
    - AttentionAggregationFloodNet: results/3d_gnn_experiment_attention_aggregation/
    - MultiScaleAggregationFloodNet: results/3d_gnn_experiment_multiscale_aggregation/
    
    � 三种架构对比实验结果目录（最新推荐）：
    - LSTM-only基线: results/3d_gnn_experiment_lstm_only/
    - Time→Space: results/3d_gnn_experiment_lstm_gnn/
    - Space→Time+Transformer: results/3d_gnn_experiment_spatial_first_transformer/
    
    🔬 对比实验结果目录：
    - LSTM-only基线: results/3d_gnn_experiment_lstm_only/
    - LSTM+GNN完整版: results/3d_gnn_experiment_lstm_gnn/
    
    核心特性：
    - 真正的4维时空建模：[batch, nodes, seq_len, features]
    - 保持时空结构完整性，不进行展平操作
    - 每个模型都实现了专门的4D前向传播方法
    - 支持图级和节点级输出
    - 统一的接口和参数设置
    
    配置说明：
    - 根据你的数据路径修改 source_cfgs 中的 source_path
    - 根据你的站点修改 example_gage_ids 列表  
    - 根据你的数据特征修改 var_t, var_c, var_out 变量
    - 根据需要调整模型超参数 model_hyperparam
    - 根据需要调整训练参数（epoch数、学习率等）
    """
    main()
