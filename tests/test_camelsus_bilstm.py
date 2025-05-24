"""
Author: Lili Yu
Date: 2025-05-10 18:00:00
LastEditTime: 2025-05-10 18:00:00
LastEditors: Lili Yu
Description: slstm model
"""

import os
import pytest

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture
def var_c():
    return [
        # "elev_mean",
        # "slope_mean",
        # "area_gages2",
        # "frac_forest",
        # "lai_max",
        # "lai_diff",
        # "dom_land_cover_frac",
        # "dom_land_cover",
        # "root_depth_50",
        # "soil_depth_statsgo",
        # "soil_porosity",
        # "soil_conductivity",
        # "max_water_content",
        # "geol_1st_class",
        # "geol_2nd_class",
        # "geol_porostiy",
        # "geol_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        # NOTE: prcp must be the first variable
        "prcp",
        "PET"
        # "dayl",
        # "srad",
        # "swe",
        # "tmax",
        # "tmin",
        # "vp",
    ]

@pytest.fixture
def arg_camelsus_biLstm(
    var_c,
    var_t,
):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    project_name = os.path.join("test_camels", "bilstm_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1981-10-01", "2012-09-30"]
    valid_period = ["2012-10-01", "2013-09-30"]
    test_period = ["2013-10-01", "2014-09-30"]

    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        # model_name="CpuLSTM",
        model_name="biLSTM",
        model_hyperparam={
            "input_size": 2,  # trend, season, residuals
            "output_size": 1,  # trend, season, residuals
            "hidden_size": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        # dataset="StlDataset",
        dataset="StreamflowDataset",
        # scaler="StandardScaler",
        # scaler="DapengScaler",
        scaler="MinMaxScaler",
        # scaler_params={
        #     "prcp_norm_cols": [
        #         "streamflow",
        #     ],
        #     "gamma_norm_cols": [
        #         "prcp",
        #         "pr",
        #         "total_precipitation",
        #         "potential_evaporation",
        #         "ET",
        #         "PET",
        #         "ET_sum",
        #         "ssm",
        #     ],
        #     "pbm_norm": True,
        # },
        batch_size=2,
        forecast_history=0,
        forecast_length=30,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=2,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 2,
        },
        gage_id=[
            "01013500",
            "01022500",
            # # "01030500",
            # # "01031500",
            # # "01047000",
            # # "01052500",
            # # "01054200",
            # # "01055000",
            # # "01057000",
            # # "01073000",
            # # "01078000",
            # # "01118300",
            # # "01121000",
            # # "01123000",
            # # "01134500",
            # # "01137500",
            # # "01139000",
            # # "01139800",
            # # "01142500",
            # # "01144000",
            # "02092500",  # 02108000 -> 02092500
            # "02108000",
        ],
        # b_decompose=True,
        which_first_tensor="sequence",
    )

def test_camelsus_biLstm(arg_camelsus_biLstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_biLstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# scaler="StandardScaler",
#   0%|          | 0/2 [00:00<?, ?it/s]
#  50%|█████     | 1/2 [00:00<00:00,  3.84it/s]
# 100%|██████████| 2/2 [00:00<00:00,  6.92it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 16070.13it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 26630.50it/s]
# Epoch 2 Loss 0.7943 time 24.86 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=2, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.9470 Valid Metric {'NSE of streamflow': [-0.1236947774887085, -0.40001869201660156], 'RMSE of streamflow': [0.6908115744590759, 1.14741051197052], 
# 'R2 of streamflow': [-0.1236947774887085, -0.40001869201660156], 'KGE of streamflow': [-1.4941370555052411, -1.341811298486459], 
# 'FHV of streamflow': [-110.4563217163086, -105.34850311279297], 'FLV of streamflow': [-36.8513069152832, -7.481730937957764]}
# Weights sucessfully loaded
# All processes are finished!

# scaler="DapengScaler",
# Epoch 2 Loss 0.8099 time 23.84 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=2, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.7485 Valid Metric {'NSE of streamflow': [0.14462369680404663, -0.3344918489456177], 'RMSE of streamflow': [0.6793646216392517, 0.8118143081665039], 
# 'R2 of streamflow': [0.14462369680404663, -0.3344918489456177], 'KGE of streamflow': [-1.5522491209569993, 0.11938353292293435], 
# 'FHV of streamflow': [-35.324649810791016, -39.70381164550781], 'FLV of streamflow': [-54.021541595458984, -65.47368621826172]}
# Weights sucessfully loaded
# All processes are finished!

# scaler="MinMaxScaler",
# Epoch 2 Loss 0.0655 time 24.80 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=2, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.0722 Valid Metric {'NSE of streamflow': [-0.022130727767944336, -0.29421377182006836], 'RMSE of streamflow': [0.0523528978228569, 0.08766112476587296], 
# 'R2 of streamflow': [-0.022130727767944336, -0.29421377182006836], 'KGE of streamflow': [-0.34468489189957907, -0.4156171024025459], 
# 'FHV of streamflow': [-79.43367004394531, -88.50468444824219], 'FLV of streamflow': [131.0086212158203, 42.13608169555664]}
# Weights sucessfully loaded
# All processes are finished!