"""test slstm model"""

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
        "dayl",
        "srad",
        "swe",
        "tmax",
        "tmin",
        "vp",
    ]

@pytest.fixture
def arg_camelsus_slstm(
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
    project_name = os.path.join("test_camels", "slstm_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1985-10-01", "1995-10-01"]
    valid_period = ["1995-10-01", "2000-10-01"]
    test_period = ["2000-10-01", "2010-10-01"]

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
            "input_size": len(var_c) + len(var_t),  # 17 + 7 = 24
            "output_size": 1,
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
                "et_morton_actual_SILO",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        batch_size=512,
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
        # gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt",
        gage_id_file="/mnt/d/minio/waterism/datasets-origin/camels/camels_us/gage_id.txt",
        which_first_tensor="sequence",
    )

def test_camels_slstm(arg_camelsus_slstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_slstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")


#   0%|          | 0/1 [00:00<?, ?it/s]
# 100%|██████████| 1/1 [00:08<00:00,  8.20s/it]
# 100%|██████████| 1/1 [00:08<00:00,  8.20s/it]
# Epoch 10 Loss 0.9405 time 8.20 lr 1.0
# sLSTM(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=2)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 10 Valid Loss 1.0060 Valid Metric {'NSE of streamflow': [0.07402777671813965, 0.1263355016708374, 0.11079108715057373, 0.10311484336853027, 0.03889620304107666, 0.038914501667022705, 0.044403791427612305, 0.06203585863113403, 0.09444379806518555, 0.19090330600738525], 'RMSE of streamflow': [0.8716039657592773, 0.9061514139175415, 1.029007077217102, 1.114901065826416, 1.0154814720153809, 0.938567042350769, 0.9940662384033203, 1.0335123538970947, 1.0223698616027832, 1.1063952445983887], 'R2 of streamflow': [0.07402777671813965, 0.1263355016708374, 0.11079108715057373, 0.10311484336853027, 0.03889620304107666, 0.038914501667022705, 0.044403791427612305, 0.06203585863113403, 0.09444379806518555, 0.19090330600738525], 'KGE of streamflow': [-0.4832414944018626, -0.2376370494556348, -0.3934577790686893, -1.3003097650355921, -0.37078001956519735, -0.2418014119723999, -0.28205195314132214, -0.3978283554578177, -18.08721203372654, 0.06895197920580176], 'FHV of streamflow': [-91.04109191894531, -88.24144744873047, -90.98316955566406, -92.9381103515625, -92.77046966552734, -89.93744659423828, -92.46479034423828, -93.42033386230469, -97.32059478759766, -95.19478607177734], 'FLV of streamflow': [-80.96578979492188, -79.16915893554688, -84.00558471679688, -83.20077514648438, -80.06905364990234, -93.80562591552734, -87.9083251953125, -83.9924545288086, -67.64541625976562, -74.35592651367188]}
# Weights sucessfully loaded
# All processes are finished!