import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_CL = [
    "elev_mean",
    "slope_mean",
    "area",
    "forest_frac",
    "crop_frac",
    "nf_frac",
    "dom_land_cover_frac",
    "dom_land_cover",
    "grass_frac",
    "shrub_frac",
    "wet_frac",
    "imp_frac",
    "fp_frac",
    "geol_class_1st",
    "geol_class_1st_frac",
    "geol_class_2nd",
    "carb_rocks_frac",
]
VAR_T_CHOSEN_FROM_CL = [
    "precip_cr2met",
    "pet_hargreaves",
    "precip_chirps",
    "precip_mswep",
    "precip_tmpa",
    "tmin_cr2met",
    "tmax_cr2met",
    "tmean_cr2met",
    "pet_8d_modis",
    "swe",
]

def run_camelscldplsac(
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
    if train_period is None:  # camels-cl time_range: ["1995-01-01", "2015-01-01"]
        train_period = ["2011-10-01", "2012-10-01"]
    if valid_period is None:
        valid_period = ["2012-10-01", "2013-10-01"]
    if test_period is None:
        test_period = ["2013-10-01", "2014-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelscl"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_cl",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_cl"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_CL)+len(VAR_C_CHOSEN_FROM_CAMELS_CL),  # 10 + 17 = 27
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
                "precip_cr2met",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "pet_hargreaves",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "01001001",
            "01001002",
            "01001003",
            "01020002",
            "01020003",
            "01021001",
            "01021002",
            # "01041002",
            # "01044001",
            # "01050002",
            # "01050004",
            # "01201001",
            # "01201003",
            # "01201005",
            # "01210001",
            # "01211001",
            # "01300009",
            # "01310002",
            # "01410004",
            # "01502002",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=15,
        var_t=VAR_T_CHOSEN_FROM_CL,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_CL,
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        train_epoch=1,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 1,
        },
        warmup_length=10,
        opt="Adadelta",
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")


run_camelscldplsac(  # camels-cl time_range: ["1995-01-01", "2015-01-01"]
    train_period=["1995-01-01", "1996-01-01"],
    valid_period=["1996-10-01", "1997-10-01"],
    test_period=["1997-10-01", "1998-10-01"],
)

# 277
# 286
