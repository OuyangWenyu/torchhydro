"""test stldataset"""

import os
import sys
import pytest
cur_path = os.path.abspath(os.path.dirname(__file__))

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
        # "PET"
        # "dayl",
        # "srad",
        # "swe",
        # "tmax",
        # "tmin",
        # "vp",
    ]

@pytest.fixture
def arg_camelsus_sltLstm(
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
    project_name = os.path.join("test_camels", "mi_stl_slstm_camelsus")
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
        model_name="sLSTM",
        model_hyperparam={
            "input_size": 3,  # len(var_c) + len(var_t),  # 17 + 7 = 24  trend, season, residuals
            "output_size": 3,  # trend, season, residuals
            "hidden_size": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StlDataset",
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
        batch_size=2,
        forecast_history=0,
        forecast_length=5,
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
        # # the gage_id.txt file is set by the user, it must be the format like:
        # # GAUGE_ID
        # # 01013500
        # # 01022500
        # # ......
        # # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        # # gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt",
        # gage_id_file="/mnt/d/minio/waterism/datasets-origin/camels/camels_us/gage_id.txt",
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
        b_decompose=True,
        which_first_tensor="sequence",
    )

def test_camels_sltLstm(arg_camelsus_sltLstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_sltLstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")

# Epoch 2 Loss 2.3128 time 300.88 lr 1.0

# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 2 Valid Loss 2.7138 Valid Metric {'NSE of trend': [-10.943303108215332, -274.4070739746094], 'RMSE of trend': [0.8217076659202576, 0.6938709616661072], 
# 'R2 of trend': [-10.943303108215332, -274.4070739746094], 'KGE of trend': [-0.8610870836449258, -0.6170529734969572], 'FHV of trend': [-107.43600463867188, -89.41093444824219], 
# 'FLV of trend': [-118.7789077758789, -90.78414154052734], 'NSE of season': [-0.014313340187072754, -0.21583032608032227], 
# 'RMSE of season': [1.3834432363510132, 0.3450661301612854], 'R2 of season': [-0.014313340187072754, -0.21583032608032227], 
# 'KGE of season': [-6.683509917232612, -52.22764135756444], 'FHV of season': [-102.6341323852539, -115.30429077148438], 
# 'FLV of season': [-84.84901428222656, -56.01063919067383], 'NSE of residuals': [-0.015451669692993164, -0.035781025886535645], 
# 'RMSE of residuals': [1.1394627094268799, 0.69868403673172], 'R2 of residuals': [-0.015451669692993164, -0.035781025886535645], 
# 'KGE of residuals': [-1.358872214061789, -1.5301356043648973], 'FHV of residuals': [-101.62789916992188, -102.33538055419922], 
# 'FLV of residuals': [-94.00704956054688, -87.31303405761719]}
# Weights sucessfully loaded
# All processes are finished!