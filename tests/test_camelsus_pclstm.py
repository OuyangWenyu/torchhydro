"""
Author: Lili Yu
Date: 2025-04-10 18:00:00
LastEditTime: 2025-04-10 18:00:00
LastEditors: Lili Yu
Description: test pclstm model
"""

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
def arg_camelsus_pclstm(
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
    project_name = os.path.join("test_camels", "pclstm_camelsus")
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
        # model_name="pcLSTM",
        model_name="sGRU",
        model_hyperparam={
            "input_size": len(var_c) + len(var_t),  # 17 + 7 = 24
            "output_size": 1,
            "hidden_size": 256,
            "num_layers": 10,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
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
        batch_size=10,
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
        # gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt",
        # # gage_id_file="/mnt/d/minio/waterism/datasets-origin/camels/camels_us/gage_id.txt",
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
        which_first_tensor="sequence",
    )

def test_camels_pclstm(arg_camelsus_pclstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_pclstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# ============================= test session starts =============================
# collecting ... collected 1 item
#
# test_camelsus_pclstm.py::test_camels_pclstm
#
# ================= 1 passed, 919 warnings in 350.80s (0:05:50) =================
# PASSED                       [100%]update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization
#
# 100%|██████████| 10/10 [00:00<00:00, 92.59it/s]
# Finish Normalization
#
# 100%|██████████| 10/10 [00:00<00:00, 3332.52it/s]
# Finish Normalization
#
# 100%|██████████| 10/10 [00:00<00:00, 2500.48it/s]
# pclstmcell model list
# pcLSTMCell()
# pcLSTMCell()
# pcLSTMCell()
# Torch is using cpu
# using 0 workers
# Epoch 1 Loss 0.9427 time 187.36 lr 1.0
# pcLSTM(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.9555 Valid Metric {'NSE of streamflow': [0.17299288511276245, 0.2565346360206604, 0.1368417739868164, 0.1658627986907959, 0.12995827198028564, 0.16247081756591797, 0.1656321883201599, 0.1706492304801941, 0.19129526615142822, 0.25567370653152466], 'RMSE of streamflow': [0.8237109184265137, 0.8359085917472839, 1.0138218402862549, 1.0751936435699463, 0.9661774635314941, 0.8761616349220276, 0.9288741946220398, 0.9718331098556519, 0.9661517143249512, 1.061186671257019], 'R2 of streamflow': [0.17299288511276245, 0.2565346360206604, 0.1368417739868164, 0.1658627986907959, 0.12995827198028564, 0.16247081756591797, 0.1656321883201599, 0.1706492304801941, 0.19129526615142822, 0.25567370653152466], 'KGE of streamflow': [0.01800776747338051, -0.015203156634360449, -1.8363335613888374, -3.6089537764499173, -0.26673286954127895, -0.036567028815405855, -0.06926628379298316, -0.18674574629063212, -19.911889047521203, 0.06409145017993945], 'FHV of streamflow': [-77.96661853790283, -77.5823175907135, -88.74953985214233, -87.35554218292236, -84.17518138885498, -81.24860525131226, -84.4189703464508, -85.05383133888245, -89.85176682472229, -86.50046586990356], 'FLV of streamflow': [-86.60645484924316, -71.07475399971008, -66.9136643409729, -75.7506012916565, -70.61716914176941, -89.22324776649475, -81.98222517967224, -80.12534379959106, -54.8542857170105, -70.3723132610321]}
# Epoch 2 Loss 0.8286 time 117.23 lr 1.0
# pcLSTM(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.8512 Valid Metric {'NSE of streamflow': [0.33196747303009033, 0.4387210011482239, 0.303067147731781, 0.3122301697731018, 0.30906856060028076, 0.36298054456710815, 0.32997119426727295, 0.3327034115791321, 0.35790306329727173, 0.4266780614852905], 'RMSE of streamflow': [0.7403191924095154, 0.726302981376648, 0.9109864234924316, 0.9763138890266418, 0.8610023856163025, 0.7641183733940125, 0.8323861956596375, 0.8717300891876221, 0.8608958721160889, 0.9313424229621887], 'R2 of streamflow': [0.33196747303009033, 0.4387210011482239, 0.303067147731781, 0.3122301697731018, 0.30906856060028076, 0.36298054456710815, 0.32997119426727295, 0.3327034115791321, 0.35790306329727173, 0.4266780614852905], 'KGE of streamflow': [0.20747468904196786, 0.41732346120701214, -1.0762395267239575, -4.3232230106372835, 0.038020376055611815, 0.3028261322009115, 0.05048072055806585, -0.18939532351820998, -14.579511669268685, 0.3860716463692674], 'FHV of streamflow': [-48.343610763549805, -48.41633439064026, -57.42325186729431, -63.90618085861206, -60.47911047935486, -57.10389018058777, -66.3650393486023, -65.43780565261841, -63.649725914001465, -67.40142107009888], 'FLV of streamflow': [-50.16298294067383, -50.2427875995636, -51.8878698348999, -52.23051905632019, -41.24124050140381, -53.67841124534607, -40.17685353755951, -42.38273203372955, -34.90665256977081, -44.91439163684845]}
# pclstmcell model list
# pcLSTMCell()
# pcLSTMCell()
# pcLSTMCell()
# Weights sucessfully loaded
# All processes are finished!
# 100%|██████████| 44/44 [02:27<00:00,  3.34s/it]
# 100%|██████████| 44/44 [01:51<00:00,  2.53s/it]


# model_name="sGRU",
