"""
Author: Lili Yu
Date: 2025-05-10 18:00:00
LastEditTime: 2025-05-10 18:00:00
LastEditors: Lili Yu
Description: test stldataset
"""

import os
import sys
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
        # scaler="DapengScaler",
        scaler="SlidingWindowScaler",
        scaler_params={
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
            "pbm_norm": False,
            "sw_width": 30,
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

def test_camelsus_sltLstm(arg_camelsus_sltLstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_sltLstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")

# Epoch 10    Loss    2.2842 time 264.16 lr 1.0              this=1/2
# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 10 Valid Loss 2.7280 Valid Metric {'NSE of trend': [-11.231891632080078, -266.3021545410156], 'RMSE of trend': [0.8315759301185608, 0.6835848093032837],
# 'R2 of trend': [-11.231891632080078, -266.3021545410156], 'KGE of trend': [-0.8744244809528539, -0.6064935157740088],
# 'FHV of trend': [-108.46617126464844, -87.94396209716797], 'FLV of trend': [-121.32003021240234, -89.53706359863281],
# 'NSE of season': [-0.019694805145263672, -0.30830419063568115], 'RMSE of season': [1.3871082067489624, 0.3579482436180115],
# 'R2 of season': [-0.019694805145263672, -0.30830419063568115], 'KGE of season': [-7.969893955458206, -62.604723275995354],
# 'FHV of season': [-103.15812683105469, -118.34867858886719], 'FLV of season': [-81.8406753540039, -47.276123046875],
# 'NSE of residuals': [-0.028864741325378418, -0.06969702243804932], 'RMSE of residuals': [1.1469635963439941, 0.7100309133529663],
# 'R2 of residuals': [-0.028864741325378418, -0.06969702243804932], 'KGE of residuals': [-1.93554807113162, -2.161503202809358],
# 'FHV of residuals': [-102.89604949951172, -104.15462493896484], 'FLV of residuals': [-89.35114288330078, -77.45653533935547]}
# Weights sucessfully loaded
# All processes are finished!


# scaler="SlidingWindowScaler",
# test_camelsus_stllstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 27.34it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 7660.83it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 11966.63it/s]
# Torch is using cpu
# Epoch 1 Loss 0.9330 time 282.26 lr 1.0
# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 1 Valid Loss 0.9690 Valid Metric {'NSE of trend': [0.9971779584884644, 0.9980688691139221], 
#                                         'RMSE of trend': [15.76749324798584, 4.608269214630127], 
#                                         'R2 of trend': [0.9971779584884644, 0.9980688691139221], 
#                                         'KGE of trend': [0.9956840718017818, 0.9973178506520254], 
#                                         'FHV of trend': [-0.7215089797973633, -0.7383750081062317], 
#                                         'FLV of trend': [0.1100272387266159, 0.33922556042671204], ''
#                                         'NSE of season': [0.7387727499008179, 0.8455219268798828], 
#                                         'RMSE of season': [726.537841796875, 128.34654235839844], 
#                                         'R2 of season': [0.7387727499008179, 0.8455219268798828], 
#                                         'KGE of season': [-0.7561547605197105, -9.348115482568184], 
#                                         'FHV of season': [-46.93497848510742, -32.48231887817383], 
#                                         'FLV of season': [-2.1360559463500977, -2.6906585693359375], 
#                                         'NSE of residuals': [0.6499893665313721, -0.07610034942626953], 
#                                         'RMSE of residuals': [525.0760498046875, 555.9549560546875], 
#                                         'R2 of residuals': [0.6499893665313721, -0.07610034942626953], 
#                                         'KGE of residuals': [0.37888108434826495, -2.3062537418810063], 
#                                         'FHV of residuals': [-49.3607177734375, -51.38527297973633], 
#                                         'FLV of residuals': [-16.784875869750977, -49.84189987182617]}
# Epoch 2 Loss 0.9321 time 296.14 lr 1.0
# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 2 Valid Loss 0.9714 Valid Metric {'NSE of trend': [0.9971907734870911, 0.998073399066925], 
#                                         'RMSE of trend': [15.731646537780762, 4.60287618637085], 
#                                         'R2 of trend': [0.9971907734870911, 0.998073399066925], 
#                                         'KGE of trend': [0.9961132727932956, 0.9975233441802825], 
#                                         'FHV of trend': [-0.7029908895492554, -0.7321071624755859], 
#                                         'FLV of trend': [0.13022956252098083, 0.3721878230571747], 
#                                         'NSE of season': [0.7368358969688416, 0.8445119857788086], 
#                                         'RMSE of season': [729.226318359375, 128.76544189453125], 
#                                         'R2 of season': [0.7368358969688416, 0.8445119857788086], 
#                                         'KGE of season': [-1.8040211282229421, -19.66817723510768], 
#                                         'FHV of season': [-48.25461959838867, -33.38445281982422], 
#                                         'FLV of season': [-1.5924453735351562, -1.812642216682434], 
#                                         'NSE of residuals': [0.6454375386238098, -0.1802462339401245], 
#                                         'RMSE of residuals': [528.4793090820312, 582.2366333007812], 
#                                         'R2 of residuals': [0.6454375386238098, -0.1802462339401245], 
#                                         'KGE of residuals': [0.6784404938347608, -3.1817412102597515], 
#                                         'FHV of residuals': [-46.2506103515625, -47.301639556884766], 
#                                         'FLV of residuals': [-22.99304962158203, -52.980796813964844]}
# Weights sucessfully loaded
# All processes are finished!
# metric_streamflow.csv
# basin_id,NSE,RMSE,R2,KGE,FHV,FLV
# 01013500,0.7062768936157227,999.5692138671875,0.7062768936157227,0.7492197330946797,-32.404945373535156,-20.46297264099121
# 01022500,0.5360612869262695,465.6806945800781,0.5360612869262695,0.7133659576391984,-30.841632843017578,-46.73394775390625