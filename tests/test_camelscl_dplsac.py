import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

@pytest.fixture
def var_c():
    return [
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
@pytest.fixture
def var_t():
    return [
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

@pytest.fixture
def camelscldplsac_arg(var_c,var_t):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
  # camels-cl time_range: ["1995-01-01", "2015-01-01"]
    train_period = ["2011-10-01", "2012-10-01"]
    valid_period = ["2012-10-01", "2013-10-01"]
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
            "n_input_features": len(var_c)+len(var_t),  # 10 + 17 = 27
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
        var_t=var_t,
        var_c=var_c,
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
    return config

def test_camelscldplsac(camelscldplsac_arg):
    train_and_evaluate(camelscldplsac_arg)
    print("All processes are finished!")

# 277
# 286
