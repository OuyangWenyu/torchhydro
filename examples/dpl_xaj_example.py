"""
This is an advanced example of training a differentiable model.
An LSTM is coupled with the Xinanjiang (XAJ) hydrological model, where the neural network
learns to output the parameters of the XAJ model.

To run this script, you need to:
Set up your data path in your `hydro_setting.yml` file.
This file should be in your user directory. If not, create one.
Refer to the __init__.py file in the torchhydro package to ensure `local_data_path` is set correctly.
For example:
   local_data_path:
       root: 'D:/data'
       datasets-origin: 'D:/data'
"""

import os
from hydrodataset.hydro_dataset import StandardVariable

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    """
    This function defines the parameters for the DPL experiment, updates the default configuration,
    and then runs the training and evaluation pipeline.
    """
    # 1. Define parameters for the DPL experiment
    source_path = SETTING["local_data_path"]["datasets-origin"]
    dpl_args = cmd(
        sub=os.path.join("results", "dpl_xaj_camels"),
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        ctx=[0],  # Use GPU 0 if available, otherwise it will fall back to CPU
        # DPL model selection
        model_name="DplAttrXaj",
        model_hyperparam={
            "n_input_features": 17,
            "n_output_features": 15,  # Number of XAJ model parameters
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 30,  # Warm-up period for the hydrological model
            "param_limit_func": "clamp",
        },
        # DPL models often require a specialized dataset
        dataset="DplDataset",
        # Use a hybrid loss function for multiple outputs
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],  # Weight for streamflow vs. other outputs
            "limit_part": [1],
        },
        # Special scaler settings for models
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                StandardVariable.PRECIPITATION,
                StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
        ],
        train_period=["1985-10-01", "1986-04-01"],
        test_period=["2000-10-01", "2001-10-01"],
        valid_period=None,
        batch_size=50,
        forecast_length=60,
        warmup_length=30,
        # Input variables for the neural network part
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        ],
        # Output variables (streamflow from XAJ, plus a dummy variable)
        var_out=[StandardVariable.STREAMFLOW, StandardVariable.EVAPOTRANSPIRATION],
        n_output=2,
        train_epoch=2,  # Set a small number of epochs for quick testing
        model_loader={"load_way": "specified", "test_epoch": 2},
        opt="Adadelta",
        which_first_tensor="sequence",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, dpl_args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
