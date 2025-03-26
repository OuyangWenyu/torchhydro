import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_GB = [
    "elev_mean",
    "slope_fdc",
    "area",
    "shrub_perc",
    "dwood_perc",
    "organic_perc",
    "dom_land_cover",
    "grass_perc",
    "root_depth_50",
    "root_depth",
    "porosity_cosby",
    "conductivity_cosby_50",
    "soil_depth_pelletier",
    "urban_perc",
    "inwater_perc",
    "inter_mod_perc",
    "frac_mod_perc",
]
VAR_T_CHOSEN_FROM_GB = [
    "precipitation",
    "pet",
    "temperature",
    "peti",
    "humidity",
    "shortwave_rad",
    "longwave_rad",
    "windspeed",
]

def run_camelsgbdplsac(
    train_period=None,
    valid_period=None,
    test_period=None
):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    if train_period is None:  # camels-gb time_range: ["1970-10-01", "2015-09-30"]
        train_period = ["2012-10-01", "2013-10-01"]
    if valid_period is None:
        valid_period = ["2013-10-01", "2014-10-01"]
    if test_period is None:
        test_period = ["2014-10-01", "2015-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelsgb"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_gb",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_gb"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_GB)+len(VAR_C_CHOSEN_FROM_CAMELS_GB),
            "n_output_features": 21,
            "n_hidden_states": 256,
            "warmup_length": 10,
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "precipitation",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "pet",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "16001",
            "16003",
            "16004",
            "17001",
            "17003",
            "17004",
            "17005",
            "17015",
            "17018",
            "18001",
            "18002",
            "18003",
            "18008",
            "18010",
            "18011",
            "18014",
            "18017",
            "18018",
            "19001",
            "19006",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=30,
        var_t=VAR_T_CHOSEN_FROM_GB,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_GB,
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        train_epoch=10,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 10,
        },
        warmup_length=10,
        opt="Adadelta",
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")


run_camelsgbdplsac(  # camels-gb time_range: ["1970-10-01", "2015-09-30"]
    train_period=["1990-07-01", "1991-07-01"],
    valid_period=["1991-10-01", "1992-10-01"],
    test_period=["1992-10-01", "1993-10-01"],
)
