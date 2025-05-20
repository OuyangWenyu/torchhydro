"""
Author: Lili Yu
Date: 2025-03-10 18:00:00
LastEditTime: 2025-03-10 18:00:00
LastEditors: Lili Yu
Description:
"""

import os
import pytest
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture
def var_c():
    return [
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

@pytest.fixture
def var_t():
    return [
        "Pobs_mm",
        "Tobs_C",
    ]

@pytest.fixture
def camelsselsmt_args(var_c,var_t):
    project_name = os.path.join("test_camels", "lstm_camelsse"),
    # camels-se time_range: ["1961-01-01", "2020-12-31"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_se",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_se"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        batch_size=50,
        forecast_history=0,
        forecast_length=366,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=10,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 10,
        },
        # the gage_id.txt file is set by the user, it must be the format like:
        # GAUGE_ID
        # 01013500
        # 01022500
        # ......
        # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_se\\gage_id.txt",
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_camelsselstm(camelsselsmt_args):
    train_and_evaluate(camelsselsmt_args)
    print("All processes are finished!")
