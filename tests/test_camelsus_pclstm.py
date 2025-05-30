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
    # project_name = os.path.join("test_camels", "pclstm_camelsus")
    # project_name = os.path.join("test_camels", "sgru_camelsus")
    # project_name = os.path.join("test_camels", "stackedgru_camelsus")
    project_name = os.path.join("test_camels", "cpugrumodel_camelsus")
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
        # model_name="stackedGRU",
        model_name="CpuGruModel",
        model_hyperparam={
            "input_size": len(var_c) + len(var_t),  # 17 + 7 = 24
            "output_size": 1,
            "hidden_size": 256,
            # "num_layers": 10,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        # scaler="DapengScaler",
        scaler="SlidingWindowScaler",
        # scaler="StandardScaler",
        scaler_params={
            # "prcp_norm_cols": [
            #     "streamflow",
            # ],
            # "gamma_norm_cols": [
            #     "prcp",
            #     "pr",
            #     "total_precipitation",
            #     "potential_evaporation",
            #     "ET",
            #     "et_morton_actual_SILO",
            #     "ET_sum",
            #     "ssm",
            # ],
            "pbm_norm": False,
            "sw_width": 30,
        },
        batch_size=2,
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
# scaler="DapengScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_camelsus_pclstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 111.25it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 5233.07it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 392.49it/s]
# Torch is using cpu
# I0528 21:25:21.795000 21460 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmp2vbs6a5j
# I0528 21:25:21.799000 21460 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmp2vbs6a5j/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.9871 time 67.49 lr 1.0
# sGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GRU(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 1.0696 Valid Metric {'NSE of streamflow': [0.0005115270614624023, 0.009464740753173828],
#                                         'RMSE of streamflow': [1.0351850986480713, 1.1029919385910034],
#                                         'R2 of streamflow': [0.0005115270614624023, 0.009464740753173828],
#                                         'KGE of streamflow': [-227.18479590920316, -0.11256500820848903],
#                                         'FHV of streamflow': [-95.07581329345703, -95.12589263916016],
#                                         'FLV of streamflow': [-107.95575714111328, -107.86727905273438]}
# Epoch 2 Loss 0.9050 time 69.56 lr 1.0
# sGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GRU(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.8365 Valid Metric {'NSE of streamflow': [0.41745156049728394, 0.369215190410614],
#                                         'RMSE of streamflow': [0.7903057336807251, 0.8801931142807007],
#                                         'R2 of streamflow': [0.41745156049728394, 0.369215190410614],
#                                         'KGE of streamflow': [-317.4779793613144, -0.7310460083047847],
#                                         'FHV of streamflow': [-63.30582046508789, -61.96416473388672],
#                                         'FLV of streamflow': [-54.28339767456055, -65.2499008178711]}
# Weights sucessfully loaded
# All processes are finished!


# model_name="stackedGRU",
# scaler="DapengScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item
# test_camelsus_pclstm.py Backend tkagg is interactive backend. Turning interactive mode on.
# update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 116.62it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3934.62it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 329.06it/s]
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Torch is using cpu
# I0529 23:18:45.367000 137786 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpgh75b3ox
# I0529 23:18:45.372000 137786 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpgh75b3ox/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.9776 time 206.47 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 1.0778 Valid Metric {'NSE of streamflow': [-0.0005340576171875, -0.01809382438659668], 
#                                         'RMSE of streamflow': [1.0357264280319214, 1.1182303428649902], 
#                                         'R2 of streamflow': [-0.0005340576171875, -0.01809382438659668], 
#                                         'KGE of streamflow': [-55.419291906907894, -0.4478935889319984], 
#                                         'FHV of streamflow': [-99.01950073242188, -98.95209503173828], 
#                                         'FLV of streamflow': [-102.26830291748047, -102.12815856933594]}
# Epoch 2 Loss 0.9949 time 220.09 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 1.0811 Valid Metric {'NSE of streamflow': [-0.00018012523651123047, -0.030116558074951172], 
#                                         'RMSE of streamflow': [1.0355432033538818, 1.1248135566711426], 
#                                         'R2 of streamflow': [-0.00018012523651123047, -0.030116558074951172], 
#                                         'KGE of streamflow': [-41.75488336250661, -0.5926675862663526], 
#                                         'FHV of streamflow': [-100.72682189941406, -100.63758087158203], 
#                                         'FLV of streamflow': [-98.16048431396484, -98.39163208007812]}
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Weights sucessfully loaded
# All processes are finished!

# model_name="stackedGRU",
# scaler="SlidingWindowScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item
# test_camelsus_pclstm.py Backend tkagg is interactive backend. Turning interactive mode on.
# update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 128.55it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3658.35it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 371.69it/s]
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Torch is using cpu
# I0529 23:38:50.442000 141912 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpx_z7s1x0
# I0529 23:38:50.447000 141912 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpx_z7s1x0/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.3396 time 203.28 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.3162 Valid Metric {'NSE of streamflow': [0.7971249222755432, 0.46267199516296387], 
#                                         'RMSE of streamflow': [0.9275136590003967, 1.761291742324829], 
#                                         'R2 of streamflow': [0.7971249222755432, 0.46267199516296387], 
#                                         'KGE of streamflow': [0.796379936818651, 0.6581788831874534], 
#                                         'FHV of streamflow': [-26.511463165283203, -10.200581550598145], 
#                                         'FLV of streamflow': [7.633522987365723, 48.50153732299805]}
# Epoch 2 Loss 0.3138 time 225.50 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.3161 Valid Metric {'NSE of streamflow': [0.7966038584709167, 0.4684339165687561], 
#                                         'RMSE of streamflow': [0.928704023361206, 1.751822829246521], 
#                                         'R2 of streamflow': [0.7966038584709167, 0.4684339165687561], 
#                                         'KGE of streamflow': [0.7918174743913804, 0.659588431651462], 
#                                         'FHV of streamflow': [-26.80518341064453, -11.133393287658691], 
#                                         'FLV of streamflow': [7.207294464111328, 47.46197509765625]}
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Weights sucessfully loaded
# All processes are finished!


# model_name="stackedGRU",
# scaler="StandardScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item
# test_camelsus_pclstm.py Backend tkagg is interactive backend. Turning interactive mode on.
# update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 122.61it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 1091.56it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 611.28it/s]
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Torch is using cpu
# I0530 08:58:29.637000 145961 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpr6x8xmod
# I0530 08:58:29.644000 145961 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpr6x8xmod/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.9789 time 225.49 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 1.1162 Valid Metric {'NSE of streamflow': [-7.796287536621094e-05, -0.05213916301727295], 
#                                         'RMSE of streamflow': [1.0174580812454224, 1.2069675922393799], 
#                                         'R2 of streamflow': [-7.796287536621094e-05, -0.05213916301727295], 
#                                         'KGE of streamflow': [-0.4763361089747553, -0.7242034155900368], 
#                                         'FHV of streamflow': [-100.43421936035156, -100.35065460205078], 
#                                         'FLV of streamflow': [-97.15518951416016, -97.22679138183594]}
# Epoch 2 Loss 0.9851 time 187.18 lr 1.0
# stackedGRU(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 1.1182 Valid Metric {'NSE of streamflow': [-0.00015652179718017578, -0.0584484338760376], 
#                                         'RMSE of streamflow': [1.0174980163574219, 1.2105810642242432], 
#                                         'R2 of streamflow': [-0.00015652179718017578, -0.0584484338760376], 
#                                         'KGE of streamflow': [-0.514832270485684, -0.7643284410754336], 
#                                         'FHV of streamflow': [-100.8194808959961, -100.66157531738281], 
#                                         'FLV of streamflow': [-94.7048568725586, -94.77054595947266]}
# grucell model list
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# GRUCell()
# Weights sucessfully loaded
# All processes are finished!


# model_name="CpuGruModel",
# scaler="DapengScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item
# test_camelsus_pclstm.py Backend tkagg is interactive backend. Turning interactive mode on.
# update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 92.47it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3498.17it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 247.68it/s]
# Torch is using cpu
# I0530 11:06:20.150000 149387 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpd8nt8hpg
# I0530 11:06:20.156000 149387 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpd8nt8hpg/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.9295 time 42.62 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.8554 Valid Metric {'NSE of streamflow': [0.15796279907226562, 0.07913219928741455], 
#                                         'RMSE of streamflow': [1.9054793119430542, 2.3046061992645264], 
#                                         'R2 of streamflow': [0.15796279907226562, 0.07913219928741455], 
#                                         'KGE of streamflow': [0.06963023243556743, 0.043479186814463944], 
#                                         'FHV of streamflow': [-75.2620849609375, -74.47550201416016], 
#                                         'FLV of streamflow': [23.258079528808594, 78.22360229492188]}
# Epoch 2 Loss 0.7968 time 38.73 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.7365 Valid Metric {'NSE of streamflow': [0.6401435136795044, 0.20411616563796997], 
#                                         'RMSE of streamflow': [1.2456706762313843, 2.142510175704956], 
#                                         'R2 of streamflow': [0.6401435136795044, 0.20411616563796997], 
#                                         'KGE of streamflow': [0.6056791038632781, 0.1997546979124316], 
#                                         'FHV of streamflow': [-37.717864990234375, -61.471683502197266], 
#                                         'FLV of streamflow': [84.76811218261719, 52.190486907958984]}
# Weights sucessfully loaded
# All processes are finished!


# model_name="CpuGruModel",
# scaler="SlidingWindowScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item
# test_camelsus_pclstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 81.81it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 686.13it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 220.38it/s]
# Torch is using cpu
# I0530 11:16:00.125000 149825 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmptmf62eec
# I0530 11:16:00.134000 149825 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmptmf62eec/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.3441 time 41.86 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.3221 Valid Metric {'NSE of streamflow': [0.786907970905304, 0.3810006380081177], 
#                                         'RMSE of streamflow': [0.950581967830658, 1.8904129266738892], 
#                                         'R2 of streamflow': [0.786907970905304, 0.3810006380081177], 
#                                         'KGE of streamflow': [0.849998318479231, 0.5859767017508248], 
#                                         'FHV of streamflow': [-20.081438064575195, 11.14083480834961], 
#                                         'FLV of streamflow': [15.533163070678711, 65.62947845458984]}
# Epoch 2 Loss 0.3231 time 43.66 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.3072 Valid Metric {'NSE of streamflow': [0.7950375080108643, 0.5054892897605896], 
#                                         'RMSE of streamflow': [0.9322731494903564, 1.6896603107452393], 
#                                         'R2 of streamflow': [0.7950375080108643, 0.5054892897605896], 
#                                         'KGE of streamflow': [0.8464543613553817, 0.6751529084332614], 
#                                         'FHV of streamflow': [-21.587797164916992, 2.3329532146453857], 
#                                         'FLV of streamflow': [12.41610336303711, 54.518795013427734]}
# Weights sucessfully loaded
# All processes are finished!


# model_name="CpuGruModel",
# scaler="SlidingWindowScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_camelsus_pclstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 121.23it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3731.59it/s]
# Finish Normalization
#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 310.32it/s]
# Torch is using cpu
# I0530 21:06:50.125000 161348 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmp7xf35x6s
# I0530 21:06:50.130000 161348 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmp7xf35x6s/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.3441 time 35.66 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.3221 Valid Metric {'NSE of streamflow': [0.786907970905304, 0.3810006380081177], 
#                                         'RMSE of streamflow': [0.950581967830658, 1.8904129266738892], 
#                                         'R2 of streamflow': [0.786907970905304, 0.3810006380081177], 
#                                         'KGE of streamflow': [0.849998318479231, 0.5859767017508248], 
#                                         'FHV of streamflow': [-20.081438064575195, 11.14083480834961], 
#                                         'FLV of streamflow': [15.533163070678711, 65.62947845458984]}
# Epoch 2 Loss 0.3231 time 35.57 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.3072 Valid Metric {'NSE of streamflow': [0.7950375080108643, 0.5054892897605896], 
#                                         'RMSE of streamflow': [0.9322731494903564, 1.6896603107452393], 
#                                         'R2 of streamflow': [0.7950375080108643, 0.5054892897605896], 
#                                         'KGE of streamflow': [0.8464543613553817, 0.6751529084332614], 
#                                         'FHV of streamflow': [-21.587797164916992, 2.3329532146453857], 
#                                         'FLV of streamflow': [12.41610336303711, 54.518795013427734]}
# Epoch 3 Loss 0.3024 time 30.53 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 3 Valid Loss 0.3056 Valid Metric {'NSE of streamflow': [0.7918796539306641, 0.6395274996757507], 
#                                         'RMSE of streamflow': [0.9394274950027466, 1.44260573387146], 
#                                         'R2 of streamflow': [0.7918796539306641, 0.6395274996757507], 
#                                         'KGE of streamflow': [0.7467224372156538, 0.6720749785835083], 
#                                         'FHV of streamflow': [-27.19687843322754, -24.55838394165039], 
#                                         'FLV of streamflow': [1.6373004913330078, 20.981609344482422]}
# Epoch 4 Loss 0.3060 time 25.88 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 4 Valid Loss 0.2935 Valid Metric {'NSE of streamflow': [0.8016318678855896, 0.6046372056007385],
#                                         'RMSE of streamflow': [0.9171532392501831, 1.5108085870742798], 
#                                         'R2 of streamflow': [0.8016318678855896, 0.6046372056007385], 
#                                         'KGE of streamflow': [0.8417367449399062, 0.7359522785350702], 
#                                         'FHV of streamflow': [-21.423213958740234, -8.12037467956543], 
#                                         'FLV of streamflow': [9.931952476501465, 41.05870819091797]}
# Epoch 5 Loss 0.2963 time 28.95 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 5 Valid Loss 0.3048 Valid Metric {'NSE of streamflow': [0.7919682264328003, 0.6300483345985413], 
#                                         'RMSE of streamflow': [0.9392275214195251, 1.4614503383636475], 
#                                         'R2 of streamflow': [0.7919682264328003, 0.6300483345985413], 
#                                         'KGE of streamflow': [0.7435538181603916, 0.6454439527814885], 
#                                         'FHV of streamflow': [-27.411291122436523, -28.137060165405273], 
#                                         'FLV of streamflow': [1.2296472787857056, 18.104223251342773]}
# Epoch 6 Loss 0.2897 time 30.89 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 6 Valid Loss 0.2962 Valid Metric {'NSE of streamflow': [0.7969009280204773, 0.6629705429077148], 
#                                         'RMSE of streamflow': [0.9280256032943726, 1.394907832145691], 
#                                         'R2 of streamflow': [0.7969009280204773, 0.6629705429077148], 
#                                         'KGE of streamflow': [0.7843600115521219, 0.70362914812037], 
#                                         'FHV of streamflow': [-24.591012954711914, -22.664411544799805], 
#                                         'FLV of streamflow': [3.4491817951202393, 19.180097579956055]}
# Epoch 7 Loss 0.2827 time 32.72 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 7 Valid Loss 0.2952 Valid Metric {'NSE of streamflow': [0.7912325263023376, 0.5821899175643921],
#                                         'RMSE of streamflow': [0.940886914730072, 1.5531058311462402], 
#                                         'R2 of streamflow': [0.7912325263023376, 0.5821899175643921], 
#                                         'KGE of streamflow': [0.8566637592242913, 0.722795275959214], 
#                                         'FHV of streamflow': [-16.8321590423584, -2.1718318462371826], 
#                                         'FLV of streamflow': [13.518131256103516, 44.430973052978516]}
# Epoch 8 Loss 0.2872 time 33.77 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 8 Valid Loss 0.3096 Valid Metric {'NSE of streamflow': [0.7864210605621338, 0.6285961866378784], 
#                                         'RMSE of streamflow': [0.9516674280166626, 1.4643157720565796], 
#                                         'R2 of streamflow': [0.7864210605621338, 0.6285961866378784], 
#                                         'KGE of streamflow': [0.7374768078574557, 0.59089360284844], 
#                                         'FHV of streamflow': [-26.54060173034668, -34.252262115478516], 
#                                         'FLV of streamflow': [-0.6028153300285339, 2.562837839126587]}
# Epoch 9 Loss 0.2827 time 31.61 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 9 Valid Loss 0.2950 Valid Metric {'NSE of streamflow': [0.7862521409988403, 0.5925910472869873], 
#                                         'RMSE of streamflow': [0.9520436525344849, 1.5336520671844482], 
#                                         'R2 of streamflow': [0.7862521409988403, 0.5925910472869873], 
#                                         'KGE of streamflow': [0.8544465335077207, 0.7243832957819654], 
#                                         'FHV of streamflow': [-15.577754020690918, -1.497160792350769], 
#                                         'FLV of streamflow': [14.12065601348877, 44.870079040527344]}
# Epoch 10 Loss 0.2767 time 31.75 lr 1.0
# CpuGruModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (gru): GruCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 10 Valid Loss 0.3159 Valid Metric {'NSE of streamflow': [0.7458036541938782, 0.5475801825523376], 
#                                          'RMSE of streamflow': [1.0382230281829834, 1.6161526441574097], 
#                                          'R2 of streamflow': [0.7458036541938782, 0.5475801825523376], 
#                                          'KGE of streamflow': [0.8094152410183653, 0.6640587682448318], 
#                                          'FHV of streamflow': [-8.024052619934082, 6.916253089904785], 
#                                          'FLV of streamflow': [18.013246536254883, 49.63261795043945]}
# Weights sucessfully loaded
# All processes are finished!