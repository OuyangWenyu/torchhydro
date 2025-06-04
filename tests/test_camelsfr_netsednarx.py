"""
Author: Lili Yu
Date: 2025-04-10 18:00:00
LastEditTime: 2025-04-10 18:00:00
LastEditors: Lili Yu
Description: test for nestednarx model
"""

import os
import pytest
import torch

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture
def var_c():
    return [
        # "top_altitude_mean",
        # "top_slo_mean",
        # "sta_area_snap",
        # "top_drainage_density",
        # "clc_2018_lvl1_1",
        # "clc_2018_lvl2_11",
        # "clc_2018_lvl3_111",
        # "clc_1990_lvl1_1",
        # "clc_2018_lvl1_2",
        # "top_slo_ori_n",
        # "top_slo_ori_ne",
        # "top_slo_ori_e",
        # "top_slo_flat",
        # "top_slo_gentle",
        # "top_slo_moderate",
        # "top_slo_ori_se",
        # "geo_py",
        # "geo_pa",
    ]

@pytest.fixture
def var_t():
    return [
        "tsd_prec",
        # "tsd_pet_ou",
        # "tsd_prec_solid_frac",
        # "tsd_temp",
        # "tsd_pet_pe",
        # "tsd_pet_pm",
        # "tsd_wind",
        # "tsd_humid",
        # "tsd_rad_dli",
        # "tsd_rad_ssi",
        # "tsd_swi_gr",
        # "tsd_swi_isba",
        # "tsd_swe_isba",
        # "tsd_temp_min",
        # "tsd_temp_max",
        # "streamflow",
    ]

@pytest.fixture
def camelsfr_narx_arg(var_c, var_t):
    project_name = os.path.join("test_camels", "NestedNarx_camelsfr")
    # camels-fr time_range: ["1970-01-01", "2022-01-01"]
    train_period = ["2018-10-01", "2019-10-01"]
    valid_period = ["2019-10-01", "2020-10-01"]
    test_period = ["2020-10-01", "2021-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_fr",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_fr"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="NestedNarx",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 64,
            "input_delay": 2,
            "feedback_delay": 1,
            "num_layers": 10,
            "close_loop": False,
            "nested_model": None,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="NarxDataset",
        # scaler="StandardScaler",
        # scaler="DapengScaler",
        scaler="SlidingWindowScaler",
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
            #     "PET",
            #     "ET_sum",
            #     "ssm",
            # ],
            "pbm_norm": False,
            "sw_width": 30,
        },
        gage_id = [
            "A550061001",
            "A369011001",
            "A284020001",
            "A330010001",
        ],
        batch_size=4,
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
        # the gage_id.txt file is set by the user, it must be the format like:
        # GAUGE_ID
        # 01013500
        # 01022500
        # ......
        # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        # gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_fr\\gage_id.txt",
        which_first_tensor="sequence",
        b_nestedness=True,
        close_loop=True,
    )


def test_camelsfr_nestednarx(camelsfr_narx_arg):
    config_data = default_config_file()
    update_cfg(config_data, camelsfr_narx_arg)
    # with torch.autograd.set_detect_anomaly(True):
    train_and_evaluate(config_data)
    print("All processes are finished!")



# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# Backend tkagg is interactive backend. Turning interactive mode on.
# collected 1 item

# test_camelsfr_netsednarx.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 896.61it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 1823.96it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 29071.03it/s]
# Torch is using cpu
# I0423 20:24:41.245000 24589 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmp5oqniugw
# I0423 20:24:41.249000 24589 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmp5oqniugw/_remote_module_non_scriptable.py
# using 0 work

#   0%|          | 0/12 [00:00<?, ?it/s]
#   8%|▊         | 1/12 [00:02<00:27,  2.46s/it]
#  17%|█▋        | 2/12 [00:04<00:21,  2.13s/it]
#  25%|██▌       | 3/12 [00:05<00:15,  1.68s/it]
#  33%|███▎      | 4/12 [00:06<00:12,  1.51s/it]
#  42%|████▏     | 5/12 [00:07<00:09,  1.37s/it]
#  50%|█████     | 6/12 [00:09<00:08,  1.35s/it]
#  58%|█████▊    | 7/12 [00:10<00:06,  1.32s/it]
#  67%|██████▋   | 8/12 [00:11<00:05,  1.33s/it]
#  75%|███████▌  | 9/12 [00:12<00:03,  1.26s/it]
#  83%|████████▎ | 10/12 [00:14<00:02,  1.23s/it]
#  92%|█████████▏| 11/12 [00:15<00:01,  1.21s/it]
# 100%|██████████| 12/12 [00:17<00:00,  1.60s/it]
# 100%|██████████| 12/12 [00:17<00:00,  1.48s/it]
# photographer   
# Epoch 1   Loss   1.0545 time 19.59 lr 1.0                            this=2/3   first
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=5, out_features=64, bias=True)
#     (narx): RNNCell(64, 64)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# F


#         if n_basin != len(self.basin_list):
# >           raise ValueError("The dimension of input data x dismatch with basintree, please check both." \
#             "\nself.n_call_froward = " + format(self.n_call_froward, 'd') + \
#             "\nn_basin = " + format(n_basin, 'd') + \
#             "\nlen(self.basin_list) = " + format(len(self.basin_list), 'd'))
# E           ValueError: The dimension of input data x dismatch with basintree, please check both.
# E           self.n_call_froward = 13
# E           n_basin = 4
# E           len(self.basin_list) = 18
# help = [1/2, 2/3]


# scaler="SlidingWindowScaler",
# Finish Normalization
#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 1239.39it/s]
# Finish Normalization
#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 16916.31it/s]
# Finish Normalization
#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 26668.13it/s]
# Torch is using cpu
# I0527 21:51:03.814000 13055 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpay1smev1
# I0527 21:51:03.818000 13055 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpay1smev1/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.3909 time 27.41 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# Epoch 1 Valid Loss 0.3886 Valid Metric {'NSE of streamflow': [0.4500880241394043, nan, 0.24559003114700317, 0.9967767000198364, nan, nan, 0.9693315029144287, 0.7414484024047852, nan, nan, 0.98923259973526, 0.9316816329956055, nan, 0.4399816393852234, nan, nan, 0.9858621954917908, 0.9798012971878052],
#                                         'RMSE of streamflow': [0.06945981830358505, 0.0, 0.022734977304935455, 0.01656528376042843, 0.0, 0.0, 0.044577401131391525, 0.025807619094848633, 0.0, 0.0, 0.04439866170287132, 0.03567321226000786, 0.0, 0.4312618672847748, 0.0, 0.0, 0.009010089561343193, 0.019329510629177094],
#                                         'R2 of streamflow': [0.4500880241394043, nan, 0.24559003114700317, 0.9967767000198364, nan, nan, 0.9693315029144287, 0.7414484024047852, nan, nan, 0.98923259973526, 0.9316816329956055, nan, 0.4399816393852234, nan, nan, 0.9858621954917908, 0.9798012971878052],
#                                         'KGE of streamflow': [0.23136926897050225, nan, 0.06242216242697585, 0.9418486056916499, nan, nan, 0.8303835611188187, 0.45588609612130826, nan, nan, 0.882994300993167, 0.7351942483393656, nan, 0.08914752553337868, nan, nan, 0.8745301287070183, 0.8498555916310028],
#                                         'FHV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], 'FLV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]}
# Epoch 2 Loss 0.3838 time 27.98 lr 1.0
# NestedNarx(
#   (dl_model): Narx(
#     (linearIn): Linear(in_features=2, out_features=64, bias=True)
#     (linearOut): Linear(in_features=64, out_features=1, bias=True)
#   )
# )
# Epoch 2 Valid Loss 0.3886 Valid Metric {'NSE of streamflow': [0.4500880241394043, nan, 0.24559003114700317, 0.9967767000198364, nan, nan, 0.9693315029144287, 0.7414484024047852, nan, nan, 0.98923259973526, 0.9316816329956055, nan, 0.4399816393852234, nan, nan, 0.9858621954917908, 0.9798012971878052],
#                                         'RMSE of streamflow': [0.06945981830358505, 0.0, 0.022734977304935455, 0.01656528376042843, 0.0, 0.0, 0.044577401131391525, 0.025807619094848633, 0.0, 0.0, 0.04439866170287132, 0.03567321226000786, 0.0, 0.4312618672847748, 0.0, 0.0, 0.009010089561343193, 0.019329510629177094],
#                                         'R2 of streamflow': [0.4500880241394043, nan, 0.24559003114700317, 0.9967767000198364, nan, nan, 0.9693315029144287, 0.7414484024047852, nan, nan, 0.98923259973526, 0.9316816329956055, nan, 0.4399816393852234, nan, nan, 0.9858621954917908, 0.9798012971878052],
#                                         'KGE of streamflow': [0.23136926897050225, nan, 0.06242216242697585, 0.9418486056916499, nan, nan, 0.8303835611188187, 0.45588609612130826, nan, nan, 0.882994300993167, 0.7351942483393656, nan, 0.08914752553337868, nan, nan, 0.8745301287070183, 0.8498555916310028],
#                                         'FHV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], 'FLV of streamflow': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]}
# Weights sucessfully loaded
# All processes are finished!
