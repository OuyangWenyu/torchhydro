"""
Author: Wenyu Ouyang
Date: 2022-09-09 14:47:42
LastEditTime: 2024-11-11 18:33:10
LastEditors: Wenyu Ouyang
Description: a script to run experiments for LSTM - CAMELS
FilePath: \torchhydro\experiments\run_camelslstm_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_US = [
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
VAR_T_CHOSEN_FROM_DAYMET = [
    # NOTE: prcp must be the first variable
    "prcp",
    "dayl",
    "srad",
    "swe",
    "tmax",
    "tmin",
    "vp",
]


def run_normal_dl(
    project_name,
    gage_id_file,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_US,
    var_t=VAR_T_CHOSEN_FROM_DAYMET,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    if train_period is None:
        train_period = ["1985-10-01", "1995-10-01"]
    if valid_period is None:
        valid_period = ["1995-10-01", "2000-10-01"]
    if test_period is None:
        test_period = ["2000-10-01", "2010-10-01"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),  # 17 + 7 = 24
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        scaler="DapengScaler",
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
        train_epoch=20,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 20,
        },
        gage_id_file=gage_id_file,
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# the gage_id.txt file is set by the user, it must be the format like:
# GAUGE_ID
# 01013500
# 01022500
# ......
# Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt"
# gage_id_file = "/mnt/d/minio/waterism/datasets-origin/camels/camels_us/gage_id.txt"
run_normal_dl(os.path.join("ndl", "explstm"), gage_id_file)


# Epoch 20 Loss 0.8337 time 4.49 lr 1.0
# CpuLstmModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (lstm): LstmCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 20 Valid Loss 0.9237 Valid Metric {'NSE of streamflow': [0.24244678020477295, -0.036179184913635254, -0.007669806480407715, -0.0780559778213501, -0.05985069274902344, -0.0077179670333862305, -0.06328582763671875, -0.06202042102813721, -0.13751983642578125, 0.010161876678466797], 'RMSE of streamflow': [1.80736243724823, 2.44464373588562, 2.341862678527832, 3.6298441886901855, 3.8003172874450684, 4.194015979766846, 6.63815450668335, 5.046813488006592, 3.151641607284546, 2.9736499786376953], 'R2 of streamflow': [0.24244678020477295, -0.036179184913635254, -0.007669806480407715, -0.0780559778213501, -0.05985069274902344, -0.0077179670333862305, -0.06328582763671875, -0.06202042102813721, -0.13751983642578125, 0.010161876678466797], 'KGE of streamflow': [0.13924946550200235, -0.031437955975069265, -0.05356836594699632, -0.21488607087441713, -0.19362044739420536, -0.145627078711843, -0.32808453020217154, -0.2884999593282622, -0.28120633888291, -0.15351603208830422], 'FHV of streamflow': [-72.1327896118164, -81.04639434814453, -80.39788055419922, -89.96695709228516, -89.39441680908203, -88.49397277832031, -94.32064056396484, -93.00979614257812, -92.5948715209961, -88.17268371582031], 'FLV of streamflow': [40.367164611816406, -1.421554684638977, 55.06676483154297, 40.552146911621094, 8.934036254882812, 22.379226684570312, 8.942217826843262, 12.878012657165527, -22.03927993774414, 40.07017135620117]}
# /home/yulili/code/torchhydro/torchhydro/trainers/deep_hydro.py:201: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   checkpoint = torch.load(weight_path, map_location=self.device)
# Weights sucessfully loaded
# All processes are finished!
