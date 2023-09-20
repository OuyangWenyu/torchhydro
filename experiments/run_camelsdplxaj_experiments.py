import os
from configs.config import cmd, default_config_file, update_cfg
import hydrodataset as hds

from trainers.trainer import train_and_evaluate


def run_dplxaj():
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    config = default_config_file()
    args = cmd(
        sub="test_camels/expdplxaj",
        source="CAMELS",
        source_region="US",
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        download=0,
        ctx=[0],
        model_name="DplLstmXaj",
        model_param={
            "n_input_features": 25,
            "n_output_features": 15,
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 10,
            "param_limit_func": "clamp",
        },
        loss_func="RMSESum",
        data_loader="DplDataset",
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
        batch_size=5,
        rho=30,  # batch_size=100, rho=365,
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
        train_epoch=20,
        te=20,
        warmup_length=10,
        opt="Adadelta",
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)


run_dplxaj()
