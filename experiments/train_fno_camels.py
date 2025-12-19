import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import torch


def main():
    """
    Standard training script for FNO on CAMELS-US dataset.
    """
    source_path = SETTING["local_data_path"]["datasets-origin"]

    # Define common model parameters
    n_input_features = 23
    n_output_features = 1
    forecast_length = 365

    args = cmd(
        sub=os.path.join("results", "fno_camels"),
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        ctx=[-1],  # Use first GPU
        model_name="FNO",
        model_hyperparam={
            "n_input_features": n_input_features,
            "n_output_features": n_output_features,
            "modes": 16,
            "width": 64,
            "n_layers": 4,
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
        batch_size=256,
        train_epoch=20,
        save_epoch=5,
        hindcast_length=0,
        forecast_length=forecast_length,
        min_time_unit="D",
        min_time_interval="1",
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.DAYLIGHT_DURATION,
            StandardVariable.SOLAR_RADIATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.VAPOR_PRESSURE,
        ],
        var_out=[StandardVariable.STREAMFLOW],
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.001},
        lr_scheduler={"lr_factor": 0.9},
        which_first_tensor="sequence",
        evaluator={"eval_way": "1pace", "pace_idx": -1},
        early_stopping=True,
        patience=5,
    )

    config_data = default_config_file()
    update_cfg(config_data, args)
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
