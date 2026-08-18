"""
This is an example of training a Diffusion-LSTM model on the CAMELS-US dataset.

This script is adapted from 'examples/lstm_camels_example.py'.

To run this script, you need to:
Set up your data path in your `hydro_setting.yml` file.
This file should be in your user directory. If not, create one.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    """
    This function defines the parameters for the experiment, updates the default configuration,
    and then runs the training and evaluation pipeline for the Diffusion Model.
    """
    # 1. Define parameters for the experiment
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "diffusion_camels"),
        # Data source configuration
        source_cfgs={"dataset_id": "camels_us"},
        # Use CPU for this example. To use GPU, set it to [0], [0, 1], etc.
        ctx=[-1],
        # --- Model selection and hyperparameters for Diffusion Model ---
        model_name="ConditionalDiffusionModel",
        model_hyperparam={
            # Diffusion-specific parameters
            "diffusion": {
                "num_timesteps": 1000,
                "data_dim": 1,  # 1 output feature: streamflow
                # condition_dim = 6 dynamic features (var_t) + 17 static attributes
                "condition_dim": 23,
                "time_emb_dim": 32,
                "beta_schedule": "linear",
            },
            # Configuration for the LSTM-based denoising network
            "denoise_net": {
                "hidden_size": 256,
                "num_layers": 4,
            },
        },
        # Basin IDs for training and evaluation
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
        ],
        # Training settings
        batch_size=4,  # Diffusion models can be memory-intensive, starting with a smaller batch size
        train_epoch=5,  # Set a small number of epochs for quick testing
        save_epoch=1,
        # Sequence lengths
        # For diffusion models, we typically predict the entire sequence at once
        hindcast_length=0,
        forecast_length=365,  # A sequence length of 1 year
        # Time settings
        min_time_unit="D",
        min_time_interval="1",
        # Input and output variables
        # Using 6 standard meteorological variables as input conditions
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
        model_loader={"load_way": "specified", "test_epoch": 5},
        # Date ranges for training, validation, and testing
        train_period=["2000-10-01", "2002-10-01"],
        valid_period=["2002-10-01", "2003-10-01"],
        test_period=["2003-10-01", "2005-10-01"],
        # Loss function and optimizer
        # GenerativeLoss is a placeholder - actual loss is computed inside the diffusion model
        loss_func="GenerativeLoss",
        loss_param={
            "eval_loss_func": "RMSESum",
        },
        opt="Adam",
        lr_scheduler={0: 0.001, 2: 0.0005, 4: 0.0001},
        # Tensor layout
        which_first_tensor="batch",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
