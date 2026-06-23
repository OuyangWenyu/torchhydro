import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

gage_id_file = os.path.join(
    SETTING["local_data_path"]["datasets-interim"],
    "attributes",
    "grdc_basin_id_205.csv",
)
# gage_id_file='/home/yichengsun/My_Code/torchhydro/data/grdc_basins.csv'
basins = pd.read_csv(gage_id_file)['basin_id'].tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "grdc_attr_205.csv",
    )
)
df.drop(columns=["basin_id"], inplace=True)
var_c = df.columns.tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "timeseries",
        "1MS",
        "GRDC_1112200.csv",
    )
)
var_t = df.columns.tolist()
var_t.remove("streamflow")
var_t.remove("time")
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "global_data.csv",
    )
)
var_g = df.columns.tolist()
var_g.remove("time")
var_g.remove("A1")
def create_config_LongTerm():
    project_name = os.path.join("train_with_LongTerm780", "205")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[2],
        model_name="SimpleBALSTM",
        model_hyperparam={
            "output_size": 1,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.4,
            "input_size_dyn": len(var_t),
            "input_size_glo": len(var_g),
            "output_size": 1,
            "input_size_sta": len(var_c),  # len(var_c) max 195
            "hindcast_output_window": 0,
        },
        model_loader={"load_way": "latest"},
        gage_id=basins,
        batch_size=512,
        forecast_history=6,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_g=var_g,
        var_out=["streamflow"],
        dataset="BALSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=10,
        train_period=["1951-01-01", "2020-12-31"],
        test_period=["1982-01-01", "1992-12-31"],
        valid_period=["1982-01-01", "1992-12-31"],
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.0001},
        which_first_tensor="batch",
        rolling=False,
        calc_metrics=True,
        metrics=["NSE", "RMSE", "R2"],
        early_stopping=False,
        patience=2,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


config = create_config_LongTerm()
train_and_evaluate(config)
