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
        # "PET"
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
            "input_size": 3,  # trend, season, residuals
            "output_size": 3,  # trend, season, residuals
            "hidden_size": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        # dataset="StlDataset",
        dataset="StreamflowDataset",
        scaler="StandardScaler",
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

# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_camelsus_bilstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 33.07it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 13842.59it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 11066.77it/s]
# Torch is using cpu
# I0523 11:42:55.272000 9226 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpn5asw4el
# I0523 11:42:55.277000 9226 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpn5asw4el/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.2441 time 79.54 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=3, bias=True)
# )
# Epoch 1 Valid Loss 0.1524 Valid Metric {'NSE of trend': [0.9664278030395508, 0.7801032662391663], 'RMSE of trend': [0.04356572777032852, 0.01960652321577072], 
# 'R2 of trend': [0.9664278030395508, 0.7801032662391663], 'KGE of trend': [0.9777564199794817, 0.9059797286364728], 'FHV of trend': [7.946622371673584, -2.4419360160827637], 
# 'FLV of trend': [3.407353639602661, -1.047161340713501], 'NSE of season': [0.999338686466217, 0.9931098222732544], 'RMSE of season': [0.03532421588897705, 0.025976551696658134], 
# 'R2 of season': [0.999338686466217, 0.9931098222732544], 'KGE of season': [0.7454516762887267, -5.990536800612962], 'FHV of season': [-2.3274123668670654, 3.058655261993408], 
# 'FLV of season': [1.1397476196289062, 4.437440872192383], 'NSE of residuals': [0.9895573258399963, 0.9957862496376038], 
# 'RMSE of residuals': [0.11555172502994537, 0.04456368088722229], 'R2 of residuals': [0.9895573258399963, 0.9957862496376038], 
# 'KGE of residuals': [-0.08693976493168898, 0.9300683293159135], 'FHV of residuals': [-12.660962104797363, -8.334555625915527], 
# 'FLV of residuals': [6.555215835571289, -1.0391641855239868]}
# Epoch 10 Valid Loss 0.0753 Valid Metric {'NSE of trend': [0.9845370054244995, 0.8134725689888], 'RMSE of trend': [0.02956661395728588, 0.018057703971862793], 
# 'R2 of trend': [0.9845370054244995, 0.8134725689888], 'KGE of trend': [0.9603059095306751, 0.9741126775789218], 'FHV of trend': [-2.309187889099121, -3.230898141860962], 
# 'FLV of trend': [0.4126966893672943, -2.022977828979492], 'NSE of season': [0.9998903274536133, 0.9976521730422974], 
# 'RMSE of season': [0.014385975897312164, 0.015163508243858814], 'R2 of season': [0.9998903274536133, 0.9976521730422974], 
# 'KGE of season': [0.8192451487872607, -1.4752938712668495], 'FHV of season': [-0.2077896147966385, 3.5982978343963623], 
# 'FLV of season': [-0.7067270278930664, 4.180027008056641], 'NSE of residuals': [0.9983240962028503, 0.9990443587303162], 
# 'RMSE of residuals': [0.04629068076610565, 0.021222153678536415], 'R2 of residuals': [0.9983240962028503, 0.9990443587303162], 
# 'KGE of residuals': [0.8446810219718078, 0.8510074151316396], 'FHV of residuals': [-0.35470762848854065, 1.2043641805648804], 
# 'FLV of residuals': [5.226898670196533, 5.44759464263916]}
# Weights sucessfully loaded
# All processes are finished!