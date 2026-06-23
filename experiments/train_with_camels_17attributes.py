"""
Train an LSTM + MoE model (LSTMMoE) on CAMELS-US.

Backbone: LSTM
MoE: token-level, SimpleAnn experts + SimpleAnn gate

Before running:
- Set up `hydro_setting.yml` with your data path.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    # 1. Define parameters for the experiment
    source_path = SETTING["local_data_path"]["datasets-origin"]

    # Common model parameters
    n_input_features = 23
    n_output_features = 1
    n_hidden_states = 128
    forecast_length = 365

    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "SimpleLSTM_rolling=1"),
        # Data source configuration
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        # GPU: [0] 表示单卡; [0,1,2] 多卡
        ctx=[-1],
        # --- Model selection and hyperparameters ---
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 23,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        # Basin IDs for training and evaluation
        gage_id=[
            "01022500",
            "01030500",
            "01047000",
            "01054200",
            "01057000",
            "01073000",
        ],
        # Training settings
        batch_size=256,
        train_epoch=20,
        save_epoch=1,
        # Sequence lengths
        hindcast_length=0,
        forecast_length=forecast_length,
        # Time settings
        min_time_unit="D",
        min_time_interval="1",
        # Input and output variables
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.DAYLIGHT_DURATION,
            StandardVariable.SOLAR_RADIATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.VAPOR_PRESSURE,
        ],
        var_out=[StandardVariable.STREAMFLOW],
        # Data components
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        # Model loading configuration
        model_loader={"load_way": "specified", "test_epoch": 2},
        # Date ranges
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        # Tensor layout
        which_first_tensor="sequence",
        evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            "eval_way": "1pace",
            # "pace_idx": -1,
            "pace_idx": -1,
        },
        early_stopping=True,
        rolling=0,
        patience=2,
        model_type="Normal",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
