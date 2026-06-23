import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

gage_id_file = os.path.join(
    SETTING["local_data_path"]["datasets-interim"],
    "attributes",
    "basin_list.csv",
)
basins = pd.read_csv(gage_id_file).squeeze().tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "grdc_attributes.csv",
    )
)
df.drop(columns=["basin_id"], inplace=True)
var_c = df.columns.tolist()
var_t = [
    "tp",
    "d2m",
    "pev",
    "ro",
    "slhf",
    "sp",
    "sro",
    "swvl",
    "u10",
    "t2m",
    "v10",
    "sd",
    "sshf",
]

def create_config_LongTerm():
    project_name = os.path.join("train_with_LongTerm_lstm", "test")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[2],
        model_name="VanillaLSTM",
        model_hyperparam={
            "input_size_sta": len(var_c),
            "input_size_dyn": len(var_t),
            "output_size": 1,
            "hidden_size": 32,
            "num_layers": 1,
            "drop_prob": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=basins,
        batch_size=256,
        forecast_history=72,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="VanillaLSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=20,
        save_epoch=1,
        train_period=["1951-01-01", "1998-12-31"],
        test_period=["1982-01-01", "1992-12-31"],
        valid_period=["1982-01-01", "1992-12-31"],
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.001, "lr_factor": 0.1, "lr_patience": 1},
        which_first_tensor="batch",
        rolling=False,
        calc_metrics=True,
        metrics=["NSE", "RMSE", "R2"],
        early_stopping=True,
        patience=2,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


config = create_config_LongTerm()
train_and_evaluate(config)
