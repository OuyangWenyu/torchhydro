import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

@pytest.fixture
def var_c():
    return [
        "dem_mean",
        "slope_mean",
        "catch_area",
        "pct_forest_levin_2011",
        "pct_agriculture_levin_2016",
        "pct_urban_levin_2021",
        "pct_naturedry_levin_2018",
        "pct_forest_corine_2000",
        "root_depth",
        "pct_sand",
        "pct_silt",
        "pct_clay",
        "chalk_d",
        "uaquifer_t",
        "pct_aeolain_sand",
        "pct_sandy_till",
        "pct_glam_clay",
    ]

@pytest.fixture
def var_t():
    return [
        "precipitation",
        "pet",
        "temperature",
        "DKM_dtp",
        "DKM_eta",
        "DKM_wcr",
        "DKM_sdr",
        "DKM_sre",
        "DKM_gwh",
        "DKM_irr",
        "Abstraction",
    ]

@pytest.fixture
def camelsdkdplsac_arg(var_c, var_t):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    # camels-dk time_range: ["1989-01-02", "2023-12-31"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "dpllstmsac_camelsdk"),
        # sub=os.path.join("test_camels", "dplannsac_camelsdk"),
        source_cfgs={
            "source_name": "camels_dk",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_dk"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(var_c)+len(var_t),
            "n_output_features": 21,
            "n_hidden_states": 256,
            "warmup_length": 10,
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [  #
                "streamflow",
            ],
            "gamma_norm_cols": [  #
                "precipitation",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "pet",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,  #
        },
        gage_id=[
            "12410011",
            "12430456",
            "12430590",
            "12430591",
            "12430666",
            "12431077",
            "13210113",
            "13210722",
            "13210733",
            "13230003",
            "13230398",
            "13230466",
            "13230469",
            "13230518",
            "13230551",
            "13230688",
            "13230704",
            "13230734",
            "13231260",
            "13231332",
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


def test_camelsdkdplsac(camelsdkdplsac_arg):
    train_and_evaluate(camelsdkdplsac_arg)
    print("All processes are finished!")
