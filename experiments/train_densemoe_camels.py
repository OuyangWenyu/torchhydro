"""
This is an example of training a Mixture-of-Experts (MoE) model on the CAMELS-US dataset.

Each expert in this example is an LSTM model. The gating network is a simple MLP (ANN).

This script is adapted from 'examples/lstm_camels_example.py'.

To run this script, you need to:
Set up your data path in your `hydro_setting.yml` file.
This file should be in your user directory. If not, create one.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

import torch


# show tensor shape in vscode debugger
def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main():
    """
    This function defines the parameters for the experiment, updates the default configuration,
    and then runs the training and evaluation pipeline for the MoE Model.
    """
    # 1. Define parameters for the experiment
    source_path = SETTING["local_data_path"]["datasets-origin"]

    # Define common model parameters
    n_input_features = 23
    n_output_features = 1
    n_hidden_states = 256
    forecast_length = 30
    num_experts = 8

    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "moe_camels"),
        # Data source configuration
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        # Use CPU for this example. To use GPU, set it to [0], [0, 1], etc.
        ctx=[0, 1],
        # --- Model selection and hyperparameters for MoE Model ---
        model_name="DenseMoE",
        model_hyperparam={
            "n_experts": num_experts,
            "static_features": 17,
            # Configuration for the two LSTM experts
            "experts_cfgs": [
                {
                    "model_name": "KuaiLSTM",
                    "model_hyperparam": {
                        "n_input_features": n_input_features,
                        "n_output_features": n_output_features,
                        "n_hidden_states": n_hidden_states,
                    },
                }
                for _ in range(num_experts)
            ],
            # Configuration for the gating network (a simple MLP)
            "gating_cfgs": {
                "model_name": "SimpleAnn",
                "model_hyperparam": {
                    # Input to the gate will now be from the last time step
                    "nx": 17,
                    "hidden_size": [64, 16],  # Hidden layers of the gating network
                    # 'ny' will be automatically set to n_experts by the MoE model
                },
            },
        },
        # Basin IDs for training and evaluation
        gage_id=[
            "01022500",
            "01030500",
            "01047000",
            "01054200",
        ],
        # Training settings
        batch_size=20,
        train_epoch=20,
        save_epoch=1,
        # Sequence lengths
        hindcast_length=0,
        forecast_length=forecast_length,
        # Time settings
        min_time_unit="D",
        min_time_interval="1",
        # Input and output variables (using the same 23 as the LSTM example)
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
        dataset="FNODataset",
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
        rolling=1,
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
