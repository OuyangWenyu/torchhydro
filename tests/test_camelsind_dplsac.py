import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_IND = [
    "elev_mean",
    "slope_mean",
    "cwc_area",
    "trees_frac",
    "lai_max",
    "lai_diff",
    "dom_land_cover",
    "dom_land_cover_frac",
    "crops_frac",
    "soil_depth",
    "soil_awc_top",
    "soil_conductivity_top",
    "water_frac",
    "geol_class_1st",
    "geol_class_2nd",
    "geol_porosity",
    "geol_permeability",
]
VAR_T_CHOSEN_FROM_IND = [
    "prcp",
    "pet",
    "tmax",
    "tmin",
    "tavg",
    "srad_lw",
    "srad_sw",
    "wind_u",
    "wind_v",
    "wind",
    "rel_hum",
    "pet_gleam",
    "aet_gleam",
    "evap_canopy",
    "evap_surface",
    "sm_lvl1",
    "sm_lvl2",
    "sm_lvl3",
    "sm_lvl4",
]

def camelsinddplsac_arg(var_c,var_t):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    if train_period is None:  # camels-ind time_range: ["1981-01-01", "2020-12-31"]
        train_period = ["2017-10-01", "2018-10-01"]
    if valid_period is None:
        valid_period = ["2018-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelsind"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_ind",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_ind"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_IND)+len(VAR_C_CHOSEN_FROM_CAMELS_IND),  # 19 + 17 = 36
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
                "prcp",
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
            "03001",
            "03002",
            "03003",
            "03004",
            "03005",
            "03006",
            "03007",
            "03008",
            "03009",
            "03010",
            "03011",
            "03012",
            "03013",
            "03014",
            "03015",
            "03016",
            "03017",
            "03018",
            "03019",
            "03020",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=30,
        var_t=VAR_T_CHOSEN_FROM_IND,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_IND,
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
    return config

def test_camelsinddplsac(camelsinddplsac_arg):
    train_and_evaluate(camelsinddplsac_arg)
    print("All processes are finished!")



