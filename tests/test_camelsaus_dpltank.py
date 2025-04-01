import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

@pytest.fixture
def var_c():
    return [
        "elev_mean",
        "mean_slope_pct",
        "catchment_area",
        "prop_forested",
        "nvis_grasses_n",
        "lc19_shrbsca",
        "lc01_extracti",
        "lc03_waterbo",
        "nvis_nodata_n",
        "carbnatesed",
        "metamorph",
        "oldrock",
        "lc11_wetlands",
        "claya",
        "sanda",
        "geol_prim",
        "geol_prim_prop",
    ]

@pytest.fixture
def var_t():
    return [
        "precipitation_AWAP",
        "et_morton_actual_SILO",
        "et_morton_point_SILO",
        "et_morton_wet_SILO",
        "et_short_crop_SILO",
        "et_tall_crop_SILO",
        "evap_morton_lake_SILO",
        "evap_pan_SILO",
        "evap_syn_SILO",
        "solarrad_AWAP",
        "tmax_AWAP",
        "tmin_AWAP",
        "vprp_AWAP",
        "mslp_SILO",
        "radiation_SILO",
        "rh_tmax_SILO",
        "rh_tmin_SILO",
        "tmax_SILO",
        "tmin_SILO",
        "vp_deficit_SILO",
        "vp_SILO",
    ]

@pytest.fixture
def camelsausdpltank_arg(
    var_c,
    var_t
):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    # camels-aus time_range: ["1990-01-01", "2010-01-01"]
    train_period = ["2006-10-01", "2007-10-01"]
    valid_period = ["2007-10-01", "2008-10-01"]
    test_period = ["2008-10-01", "2009-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac_camelsaus"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_aus",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_aus"
            ),
        },
        ctx=[-1],
        model_name="DplLstmTank",
        # model_name="DplAnnTank",
        model_hyperparam={
            "n_input_features": len(var_t)+len(var_c),  # 21 + 17 = 38
            "n_output_features": 20,
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
                "precipitation_AWAP",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "et_morton_actual_SILO",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "912101A",
            "912105A",
            "915011A",
            "917107A",
            "919003A",
            "919201A",
            "919309A",
            "922101B",
            "925001A",
            "926002A",
            "G9030124",
            "G9030250",
            "G9070142",
            "A0020101",
            "A0030501",
            "G0010005",
            "G0050115",
            "G0060005",
            "401009",
            "401012",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=30,
        var_t=var_t,
        var_c=var_c,
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

def test_camelsausdpltank(camelsausdpltank_arg):
    train_and_evaluate(camelsausdpltank_arg)
    print("All processes are finished!")

