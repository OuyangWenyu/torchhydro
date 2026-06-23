import os
import torch
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    """
    针对 HydroFNO 适配的 CAMELS-US 训练脚本。
    输入维度约定为 (sequence, batch, features)。
    """
    source_path = SETTING["local_data_path"]["datasets-origin"]

    # 1. 确定输入特征数
    # 基础变量 6 个 + 提前注入的位置编码（假设你注入了 1 维时间编码） = 7
    # 注意：请根据你实际 DataLoader 出来的维度调整此数值
    n_input_features = 24
    n_output_features = 1
    forecast_length = 365

    args = cmd(
        sub=os.path.join("results", "fno_camels"),
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        ctx=[-1],  # 指定 GPU ID
        model_name="FNO",  # 确保此名称已在 torchhydro 的模型注册表中
        model_hyperparam={
            # 这里的参数名必须与 HydroFNO.__init__ 保持一致
            "n_modes": (16,),  # FNO 保持的频率模式数
            "in_channels": n_input_features,
            "out_channels": n_output_features,
            "hidden_channels": 64,  # 原 width
            "n_layers": 4,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "fno_skip": "linear",  # 频谱层跳跃连接
            "channel_mlp_skip": "soft-gating",  # MLP层跳跃连接
            "norm": "batch_norm",  # 使用适配后的 HydroBatchNorm
            "use_channel_mlp": True,
            "channel_mlp_expansion": 0.5,
            "preactivation": False,  # 默认后激活模式
        },
        gage_id=[
            "01022500",
            "01030500",
            "01047000",
            "01054200",
            "01057000",
            "01073000",
            "01118300",
            "01123000",
            "01134500",
            "01137500",
        ],
        batch_size=64,  # FNO 较占显存，若显存不足可调小
        train_epoch=50,  # FNO 通常需要更多 epoch 来收敛频率权重
        save_epoch=10,
        hindcast_length=0,
        forecast_length=forecast_length,
        min_time_unit="D",
        min_time_interval="1",
        # 气象驱动变量
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.DAYLIGHT_DURATION,
            StandardVariable.SOLAR_RADIATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.VAPOR_PRESSURE,
        ],
        # 注意：如果你在 Dataset 层注入了位置编码，请确保对应逻辑已在 Dataset 中实现
        var_out=[StandardVariable.STREAMFLOW],
        dataset="FNODataset",
        scaler="DapengScaler",
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        loss_func="NSELoss",  # 水文任务常用 NSE 或其变体
        opt="Adam",
        opt_param={"lr": 0.001},
        lr_scheduler={"lr_factor": 0.5, "patience": 5},  # 减小学习率以精细调整频率权重
        which_first_tensor="sequence",  # 核心：必须指定 sequence 第一维
        evaluator={"eval_way": "1pace", "pace_idx": -1},
        early_stopping=True,
        patience=10,
    )

    config_data = default_config_file()
    update_cfg(config_data, args)

    # 执行训练与评估
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
