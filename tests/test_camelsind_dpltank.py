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

@pytest.fixture
def var_t():
    return [
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

@pytest.fixture
def camelsinddpltank_arg(var_c,var_t):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    # camels-ind time_range: ["1981-01-01", "2020-12-31"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    return cmd(
        sub=os.path.join("test_camels", "dpllstmtank_camelsind"),
        # sub=os.path.join("test_camels", "dplanntank_camelsind"),
        source_cfgs={
            "source_name": "camels_ind",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_ind"
            ),
        },
        ctx=[-1],
        model_name="DplLstmTank",
        # model_name="DplAnnTank",
        model_hyperparam={
            "n_input_features": len(var_c)+len(var_t),  # 19 + 17 = 36
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


def test_camelsinddpltank(camelsinddpltank_arg):
    config = default_config_file()
    update_cfg(config, camelsinddpltank_arg)
    train_and_evaluate(config)
    print("All processes are finished!")



