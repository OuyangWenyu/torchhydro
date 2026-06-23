import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.resulter import Resulter
from torchhydro.trainers.trainer import set_random_seed

gage_id_file='/home/yichengsun/My_Code/torchhydro/data/grdc_basins.csv'
basins = pd.read_csv(gage_id_file)['basin_id'].tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "filtered_attr.csv",
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
    project_name = os.path.join("train_with_LongTerm", "100_period")
    train_path = '/home/yichengsun/My_Code/torchhydro/results/train_with_LongTerm/100_period'
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[2],
        model_name="BALSTM",
        model_hyperparam={
            "output_size": 1,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.4,
            "input_size_dyn": len(var_t),
            "input_size_glo": 106,
            "output_size": 1,
            "input_size_sta": len(var_c),  # len(var_c) max 195
            "hindcast_output_window": 0,
        },
        model_loader={"load_way": "latest"},
        gage_id=basins,
        batch_size=256,
        forecast_history=72,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="BALSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        test_period=(["1982-01-01", "1992-12-31"]),
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
        train_mode=False,
        weight_path=os.path.join(train_path, "model_Ep100.pth"),
        stat_dict_file=os.path.join(train_path, "dapengscaler_stat.json"),
    )
    update_cfg(config_data, args)
    return config_data


def main():
    config_data = create_config_LongTerm()
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    resulter = Resulter(config_data)
    model = DeepHydro(config_data)
    results = model.model_evaluate()
    resulter.save_result(
        results[0],
        results[1],
    )


if __name__ == "__main__":
    main()

