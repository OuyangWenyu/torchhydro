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

# ================================
# 配置参数区域
# ================================

# 路径配置
PATH_CONFIG = {
    "data_dir": "/home/mahaoran/GNN/data_jialingjiang_29",
    "result_dir": "/home/mahaoran/GNN/result/test_jialingjiang_HF_autoregressive_29",
    "log_dir": "/home/mahaoran/GNN/logs/test_jialingjiang_HF_autoregressive_29",
}

# 训练参数配置
TRAIN_CONFIG = {
    "num_epochs": 30,  # 训练轮数
    "learning_rate": 0.005,  # 学习率（自回归模型通常需要较小的学习率）
    "lr_decay_factor": 0.95,  # 学习率衰减系数
    "max_grad_norm": 1.0,  # 梯度裁剪阈值
    "num_workers": 0,  # 数据加载器工作进程数
    "random_seed": 1234,  # 随机种子
}


# 数据配置
DATA_CONFIG = {
    "train_start": "2021-01-01",
    "train_end": "2024-12-31",
    "test_start": "2021-01-01",
    "test_end": "2024-12-31",
    "target_station": "sanxia_60703800",
    "basin_id": "sanxia_60703800",
    "normalization_scale": 4,
    "selected_attrs": ["area", "p_mean", "sgr_dk_sav"],
}


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
        # "songliao_21113800",
        "songliao_21401050",
        "songliao_21401550",
    ]

    # ==== 智能推断 in_channels 和 output_size ====
    hindcast_length = 20
    forecast_length = 2
    var_t = [
        "total_precipitation_hourly",
        "temperature_2m",
        "u_component_of_wind_10m",
        "surface_pressure",
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
    output_size = hindcast_length + forecast_length  # flood_event 只预测未来1步

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "source_name": "tgdatasource",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["1D"],
            "sim_flow_path": os.path.join(
                SETTING["local_data_path"]["datasets-interim"],
                "songliao_3h",
                "songliao_3h.csv",
            ),
        },
        # station_cfgs={
        #     "station_cols": station_cols,  # 站点只有降雨数据目前,
        #     },
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 模型配置
        model_name="MultiHopDirectedGConv",  # 使用GCN模型
        model_hyperparam={
            "hindcast_length": 30,  # 历史数据长度M（建议7-30天范围）
            "forecast_length": 3,  # 未来预报数据长度n
            "gcgru_out_channels": 32,  # GConvGRU输出通道数
            "gcgru_K": 15,  # Chebyshev滤波器大小
            "mlp_hidden_dim": 128,  # MLP隐藏层维度
            "mlp_dropout": 0.1,  # MLP Dropout率
            "node_attr_dim": 3,  # 节点静态属性维度
            "node_attr_hidden_dim": 32,  # 节点属性处理隐藏层维度
            # 感受野配置
            # 'receptive_field': {
            #     'type': 'multihop',
            #     'config': {'num_hops': 2}
            # }
            "receptive_field": {
                "type": "attention",
                "config": {"heads": 4, "dropout": 0.1},
            },
        },
        dataset="GNNDataset",
        scaler="StandardScaler",
        batch_size=16,
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
        test_period=["1995-06-01-02", "1996-06-01-02"],  # 测试时段
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
