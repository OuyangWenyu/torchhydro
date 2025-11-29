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
    n_hidden_states = 256
    forecast_length = 365

    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "lstm_moe_camels"),
        # Data source configuration
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        # GPU: [0] 表示单卡; [0,1,2] 多卡
        ctx=[0],
        # --- Model selection and hyperparameters ---
        model_name="LSTMMoE",
        model_hyperparam={
            "n_input_features": n_input_features,
            "n_output_features": n_output_features,
            "n_hidden_states": n_hidden_states,
            "activation": "gelu",  # relu / gelu
            "normalization": "rmsnorm",  # layernorm / rmsnorm / None
            "dr": 0.4,
            "moe_cfgs": {
                "n_experts": 16,
                "top_k": 2,
                "gating_mode": "full",
                "capacity_factor": 1.0,
                "load_balance_loss_coef": 0.01,
                "noisy_gating": True,
                # 多卡时可以指定专家放到不同 GPU；单卡就用 None
                "expert_devices": ["cpu"],
                # ------- Experts: SimpleAnn MLPs -------
                "experts_cfgs": [
                    {
                        "model_name": "SimpleAnn",
                        "model_hyperparam": {
                            "nx": n_hidden_states,
                            "ny": n_hidden_states,
                            "hidden_size": [n_hidden_states * 4],
                            "dr": 0.0,
                            "activation": "gelu",
                            "normalization": None,
                        },
                    }
                    for _ in range(16)
                ],
                # ------- Gating network: SimpleAnn -------
                "gating_cfgs": {
                    "model_name": "SimpleAnn",
                    "model_hyperparam": {
                        "nx": n_hidden_states,
                        "ny": None,  # 会在 MoE __init__ 里自动改成 n_experts
                        "hidden_size": [n_hidden_states],
                        "dr": 0.0,
                        "activation": "gelu",
                        "normalization": "RMSNorm",
                    },
                },
            },
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
        train_epoch=2,
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
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 1e-3},
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
