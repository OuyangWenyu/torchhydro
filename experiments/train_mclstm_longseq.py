import os
import logging
import torch
import pandas as pd
from pathlib import Path
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
torch.backends.cudnn.enabled = False


def create_mclstm_config():
    cfg = default_config_file()

    sanxia_dir = "/home/litianli/Dataset/"
    csv_path = Path("data/223_floodevents.csv")
    try:
        gage_df = pd.read_csv(csv_path)
        gage_id = sorted(gage_df["GAGE_ID"].dropna().tolist())
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件未找到，请检查路径：{csv_path.absolute()}")

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

    args = cmd(
        sub="mclstm_longseq/223",
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            "other_settings": {
                "time_unit": ["1h"],
                "dataset_name": "223_FloodEvents_for_shap",
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
        gage_id=gage_id,
        batch_size=64,
        train_epoch=20,
        save_epoch=1,
        hindcast_length=0,
        forecast_length=24,
        min_time_unit="h",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        model_loader={"load_way": "best"},
        metrics=["NSE", "RMSE", "KGE", "FHV", "FLV"],
        fill_nan="no",
        evaluator={"eval_way": "once", "stride": 0},
        early_stopping=True,
        patience=2,
        train_period=["2014-01-01", "2023-07-31"],
        valid_period=["2023-08-01", "2024-12-31"],
        test_period=["2023-08-01", "2024-12-31"],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={0: 1, 3: 0.5, 4: 0.2},
        which_first_tensor="sequence",
    )

    update_cfg(cfg, args)
    return cfg


def train_mclstm():
    cfg = create_mclstm_config()
    os.makedirs(cfg["data_cfgs"]["case_dir"], exist_ok=True)
    train_and_evaluate(cfg)


if __name__ == "__main__":
    train_mclstm()
