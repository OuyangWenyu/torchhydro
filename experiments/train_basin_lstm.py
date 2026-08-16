"""
Basin-level LSTM 基线: 直接用流域平均降雨 → LSTM → 流域出口径流
不走站点数据，不走图网络 — 纯时序模型
参考: experiments/3h_events_run_regional_weighted_mse_loss.py
"""

import os, sys, logging, warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING


def create_config():
    config_data = default_config_file()

    example_gage_ids = [
        "songliao_20800900", "songliao_20810200", "songliao_21100150",
        "songliao_21110150", "songliao_21401050", "songliao_21401550"
    ]

    args = cmd(
        sub="basin_lstm/test",
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
        },
        ctx=[-1],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 64,
        },
        gage_id=example_gage_ids,
        batch_size=8,
        hindcast_length=72,
        forecast_length=12,
        frwin=12,
        min_time_unit="h",
        min_time_interval="3",
        var_t=["rain"],
        var_c=["None"],
        var_out=["inflow", "flood_event"],
        dataset="FloodEventDataset",
        scaler="DapengScaler",
        train_mode=True,
        train_epoch=60, save_epoch=1,
        train_period=["1980-01-01-02", "2014-12-31-23"],
        valid_period=["2015-01-01-02", "2018-12-31-23"],
        test_period=["2019-01-01-02", "2024-12-31-23"],
        which_first_tensor="sequence",
        opt="Adam",
        lr_scheduler={"lr": 0.001, "lr_factor": 0.5, "lr_patience": 8},
        loss_func="FloodLoss",
        loss_param={"loss_func": "MSELoss", "flood_weight": 2, "flood_strategy": "weight", "device": [0]},
        model_loader={"load_way": "best"},
        evaluator={"eval_way": "floodevent"},
        early_stopping=True,
        patience=10,
        calc_metrics=True,
    )

    update_cfg(config_data, args)
    return config_data


if __name__ == "__main__":
    print("训练 Basin-level LSTM (流域平均, 无站点, 无图网络)...")
    config = create_config()
    train_and_evaluate(config)
    print("完成!")
