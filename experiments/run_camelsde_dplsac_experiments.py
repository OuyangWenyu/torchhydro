import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_DE = [
    "elev_mean",
    "slope_fdc",
    "area",
    "forests_and_seminatural_areas_perc",
    "artificial_surfaces_perc",
    "agricultural_areas_perc",
    "wetlands_perc",
    "water_bodies_perc",
    "clay_30_100cm_mean",
    "soil_organic_carbon_30_100cm_mean",
    "silt_0_30cm_mean",
    "sand_0_30cm_mean",
    "bulk_density_0_30cm_mean",
    "geochemical_rocktype_silicate_perc",
    "geochemical_rocktype_carbonatic_perc",
    "cavity_pores_perc",
    "aquifer_perc",
]
VAR_T_CHOSEN_FROM_DE = [
    "precipitation_mean",
    "ET",  #  lock of evaporation
    "precipitation_min",
    "precipitation_median",
    "precipitation_max",
    "precipitation_stdev",
    "water_level",
    "humidity_mean",
    "humidity_min",
    "humidity_median",
    "humidity_max",
    "humidity_stdev",
    "radiation_global_mean",
    "radiation_global_min",
    "radiation_global_median",
    "radiation_global_max",
    "radiation_global_stdev",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
]

def run_camelsdedplsac(
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
    if train_period is None:  # camels-de time_range: ["1951-01-01", "2020-12-31"]
        train_period = ["2017-10-01", "2018-10-01"]
    if valid_period is None:
        valid_period = ["2018-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelsde"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_de",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_de"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_DE)+len(VAR_C_CHOSEN_FROM_CAMELS_DE),  # 20 + 17 = 37
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
                "precipitation_mean",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "pet",
                "ET",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "DE110000",
            "DE110010",
            "DE110020",
            "DE110030",
            "DE110040",
            "DE110060",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=30,
        var_t=VAR_T_CHOSEN_FROM_DE,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_DE,
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


run_camelsdedplsac(  # camels-de time_range: ["1951-01-01", "2020-12-31"]
    train_period=["1951-07-01", "1952-07-01"],
    valid_period=["1952-10-01", "1953-10-01"],
    test_period=["1953-10-01", "1954-10-01"],
)

# 99
