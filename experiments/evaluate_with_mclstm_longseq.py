import os
import logging
from pathlib import Path

import pandas as pd
import torch

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

torch.backends.cudnn.enabled = False

sanxia_dir = "/home/litianli/Dataset/"
csv_path = Path("data/223_floodevents.csv")
gage_df = pd.read_csv(csv_path)
gage_id = sorted(gage_df["GAGE_ID"].dropna().tolist())

var_t = [
    "precipitation_obs",
    "temperature_2m",
    "surface_net_solar_radiation_hourly",
    "snowfall_hourly",
]
var_c = [
    "area",
    "pet_mm_syr",
    "aet_mm_syr",
    "ele_mt_sav",
]
input_size = len(var_t) + len(var_c)

train_run_dir = os.path.join("results", "mclstm_longseq", "223")
stat_dict_file = os.path.join(train_run_dir, "dapengscaler_stat.json")
weight_file = os.path.join(train_run_dir, "best_model.pth")
if not os.path.exists(weight_file):
    weight_file = os.path.join(train_run_dir, "model_Ep20.pth")
if not os.path.exists(weight_file):
    raise FileNotFoundError(
        f"找不到模型权重文件，请检查: {os.path.join(train_run_dir, 'best_model.pth')} 或 {os.path.join(train_run_dir, 'model_Ep20.pth')}"
    )
if not os.path.exists(stat_dict_file):
    raise FileNotFoundError(f"找不到统计文件，请检查: {stat_dict_file}")


def get_config():
    cfg = default_config_file()
    args = cmd(
        train_mode=False,
        sub="mclstm_eval/223",
        stat_dict_file=stat_dict_file,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            "other_settings": {
                "time_unit": ["1h"],
                "dataset_name": "223_FloodEvents",
                "delta_t_hours": 1.0,
            },
        },
        ctx=[1],
        model_name="MCLSTM",
        model_hyperparam={
            "input_size": input_size,
            "output_size": 1,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
            "initial_forget_bias": 0.0,
        },
        model_loader={"load_way": "pth", "pth_path": weight_file},
        gage_id=gage_id,
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        batch_size=256,
        warmup_length=0,
        hindcast_length=0,
        forecast_length=24,
        min_time_unit="h",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=["2014-01-01", "2023-07-31"],
        valid_period=["2023-08-01", "2024-12-31"],
        test_period=["2014-01-01", "2023-07-31"],
        rolling=0,
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "FHV", "FLV"],
        fill_nan="no",
        evaluator={"eval_way": "once", "stride": 0},
    )
    update_cfg(cfg, args)
    os.makedirs(cfg["data_cfgs"]["case_dir"], exist_ok=True)
    return cfg


if __name__ == "__main__":
    cfg = get_config()
    train_and_evaluate(cfg)
