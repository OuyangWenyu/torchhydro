"""
This is a simple example of training a standard LSTM model on the CAMELS-US dataset.

To run this script, you need to configure your data path in `~/hydro_setting.yml`.
Create this file in your home directory with the following format:

    storage:
      local:
        root: D:/data/hydrodatasets

The path to individual datasets (e.g., CAMELS-US) is resolved automatically
from the registry — no need to set individual dataset paths.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    """
    This function defines the parameters for the experiment, updates the default configuration,
    and then runs the training and evaluation pipeline.
    """
    # 1. Define parameters for the experiment
    # You can refer to the `cmd` function in `torchhydro.configs.config` for more details.
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "lstm_camels"),
        # Data source — dataset_id auto-resolves path from ~/hydro_setting.yml
        source_cfgs={"dataset_id": "camels_us"},
        # Use CPU for this example. To use GPU, set it to [0], [0, 1], etc.
        ctx=[-1],
        # Model selection and hyperparameters
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        # Basin IDs for training and evaluation
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        # Training settings
        batch_size=8,
        train_epoch=2,  # Set a small number of epochs for quick testing
        save_epoch=1,
        # Sequence lengths
        hindcast_length=0,
        forecast_length=20,
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
        sampler="KuaiSampler",
        scaler="DapengScaler",
        # Model loading configuration for evaluation
        model_loader={"load_way": "specified", "test_epoch": 2},
        # Date ranges for training, validation, and testing
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        # Tensor layout
        which_first_tensor="sequence",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
