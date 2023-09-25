"""
Author: Wenyu Ouyang
Date: 2023-09-20 20:05:10
LastEditTime: 2023-09-25 19:20:51
LastEditors: Wenyu Ouyang
Description: 
FilePath: /torchhydro/experiments/run_camelsdplxaj_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
from configs.config import cmd, default_config_file, update_cfg
import hydrodataset as hds

from trainers.trainer import train_and_evaluate


def run_dplxaj(train_period=None, valid_period=None, test_period=None):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    if train_period is None:
        train_period = ["1985-10-01", "1995-10-01"]
    if valid_period is None:
        valid_period = ["1995-10-01", "2000-10-01"]
    if test_period is None:
        test_period = ["2000-10-01", "2010-10-01"]
    config = default_config_file()
    args = cmd(
        sub="test_camels/expdplxaj",
        source="CAMELS",
        source_region="US",
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        download=0,
        ctx=[0],
        model_name="DplLstmXaj",
        model_hyperparam={
            "n_input_features": 25,
            "n_output_features": 15,
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 10,
            "param_limit_func": "clamp",
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "PET",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
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
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        rho=30,
        var_t=[
            "prcp",
            "PET",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        train_epoch=2,
        te=2,
        warmup_length=10,
        opt="Adadelta",
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)


run_dplxaj(
    train_period=["1985-10-01", "1986-10-01"],
    valid_period=["1986-10-01", "1987-10-01"],
    test_period=["1987-10-01", "1988-10-01"],
)
