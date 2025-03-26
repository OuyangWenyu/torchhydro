import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_SE = [
    "Elevation_mabsl",
    "Slope_mean_degree",
    "Area_km2",
    "Shrubs_and_grassland_percentage",
    "Urban_percentage",
    "Water_percentage",
    "Forest_percentage",
    "Open_land_percentage",
    "Glaciofluvial_sediment_percentage",
    "Bedrock_percentage",
    "Postglacial_sand_and_gravel_percentage",
    "Till_percentage",
    "Wetlands_percentage",
    "Peat_percentage",
    "Silt_percentage",
    "DOR",
    "RegVol_m3",
]
VAR_T_CHOSEN_FROM_SE = [
    "Pobs_mm",
    "ET",  # lock of evaporation
    "Tobs_C",
]

def run_camelssedplsac(
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
    if train_period is None:  # camels-se time_range: ["1961-01-01", "2020-12-31"]
        train_period = ["2017-10-01", "2018-10-01"]
    if valid_period is None:
        valid_period = ["2018-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelsse"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_se",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_se"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_SE)+len(VAR_C_CHOSEN_FROM_CAMELS_SE),  # 3 + 17 = 20
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
                "Pobs_mm",
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
            "5",
            "20",
            "37",
            "97",
            "138",
            "186",
            "200",
            "257",
            "364",
            "591",
            "654",
            "736",
            "740",
            "751",
            "855",
            "1069",
            "1083",
            "1123",
            "1166",
            "1169",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=30,
        var_t=VAR_T_CHOSEN_FROM_SE,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_SE,
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


run_camelssedplsac(  # camels-se time_range: ["1961-01-01", "2020-12-31"]
    train_period=["1961-07-01", "1962-07-01"],
    valid_period=["1962-10-01", "1963-10-01"],
    test_period=["1963-10-01", "1964-10-01"],
)

# 327
