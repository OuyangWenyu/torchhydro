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
        "area_gages2",
        "frac_forest",
        "lai_max",
        "lai_diff",
        "dom_land_cover_frac",
        "dom_land_cover",
        "root_depth_50",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "geol_1st_class",
        "geol_2nd_class",
        "geol_porostiy",
        "geol_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        # NOTE: prcp must be the first variable
        "prcp",
        "PET",
        "dayl",
        "srad",
        # "swe",
        "tmax",
        "tmin",
        "vp",
        "streamflow",
    ]

@pytest.fixture
def camelsdplnarxsac_arg(var_c, var_t):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1985-10-01", "1986-10-01"]
    valid_period = ["1986-10-01", "1987-10-01"]
    test_period = ["1987-10-01", "1988-10-01"]
    return cmd(
        sub=os.path.join("test_camels", "dplnarxsac"),
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        model_name="DplNarxSac",
        model_hyperparam={
            "n_input_features": len(var_t) + len(var_c),  # 8 + 17
            "n_output_features": 21,
            "n_hidden_states": 256,  # 256
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
            "01073000",
            "01078000",
            "01118300",
            "01121000",
            "01123000",
            "01134500",
            "01137500",
            "01139000",
            "01139800",
            "01142500",
            "01144000",
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


def test_camelsdplnarxsac(camelsdplnarxsac_arg):
    config = default_config_file()
    update_cfg(config, camelsdplnarxsac_arg)
    train_and_evaluate(config)
    print("All processes are finished!")
