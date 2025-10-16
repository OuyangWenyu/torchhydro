"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2025-01-10 10:11:30
LastEditors: Wenyu Ouyang
Description: Train a model for 3775 basins
FilePath: /HydroForecastEval/scripts/train_googlefloodhub_camels_671basins_ear5land_less_param_new_rolling_large_horizon.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pandas as pd

import sys
#sys.path.append(r"C:\Users\Pengfei Qu\Desktop\torchhydro")
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels
#from hydrodataset.camels_aef import CamelsAef
from hydrodataset.camels_sk_aqua import CamelsSk
import torch
torch.backends.cudnn.enabled = False
# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("G:\data")
camels = CamelsSk(camels_dir)
# gage_id = camels.read_site_info()["gauge_id"].values.tolist()
gage_id = ['1002640', '1001630', '1002625', '1001655', '1002605', '1002610', '1002615', '1002650', '1001625', '1002685', '1002687', '1002698', '1002675', '1001670', '1002635', '1012630', '2001630', '1001660', '1005640', '1003620', '1014630', '2001658', '2012645', '1014620', '1005697', '1014640', '1007625', '1014650', '1007635', '1015645', '1013645', '1018610', '1018640', '1014680', '1018662', '1018680', '1018683', '1007685', '1019630', '2018630', '2018635', '2003610', '2015635', '1013655', '1006660', '1021650', '1021680', '2003670', '1015644', '2003690', '2018645', '1023660', '1006650', '1022640', '1023670', '2007620', '2007640', '1006672', '1006680', '1022630', '2013615', '2004605', '2007660', '2002655', '2002692', '2008645', '2004625', '1022680', '1006690', '2016650', '2002685', '2004640', '2018674', '1022650', '1022655', '2016680', '2018620', '2009620', '2009670', '2004650', '2003640', '2013650', '1006665', '1007680', '2004655', '2004695', '2018685', '2018680', '2019615', '2004635', '2004675', '2004680', '2012665', '2005660', '2013690', '2013685', '2008650', '2010650', '2008635', '2008630', '2011640', '2011650', '2011660', '1022648', '2004668', '1018623', '1018620', '1018625', '1018630', '1018665', '2019635', '1018670', '2012696', '1018675', '2018665', '2019655', '2012690', '2005680', '2008690', '2018655', '2006665', '1022670', '2019695', '2012625', '1018650', '1005695', '2006630', '1018655', '2017685', '1007640', '1018635', '2017620', '1018697', '1018693', '1018690', '1007655', '1018695', '2020615', '2014620', '2020650', '2006625', '2014690', '2014660', '2014640', '2022670', '2022660', '2022655', '2014680', '2020675', '1016660', '1007650', '2021675', '2010690', '1016650', '1007645', '1007617', '1007615', '1007605', '2022610', '2022640', '2022680', '2012652', '2020603', '2019610', '2021677', '2021685', '2021690', '2021665', '2021640', '2012628', '2012648', '2012640', '2012695', '2012653', '2012650', '2012670', '2012660']

gage_id = sorted([x for x in gage_id])

assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

length = 7
dim = 256
scaler = "DapengScaler"
# scaler = "StandardScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        #project_dir="D:\\torchhydro\\text2attr",
        sub=project_name,
        source_cfgs={
            "source_name": "camels_sk",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "SK"},
        },
        ctx=[1],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 15,
            "output_size": 1,
            "hidden_size": 256,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
        # gage_id=gage_id[5000:5009],
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=256,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="h",
        min_time_interval=1,
        var_t=["total_precipitation", "temperature_2m", "potential_evaporation", "surface_net_solar_radiation", "precip_obs"],
        scaler_params={
            "prcp_norm_cols": [
                "q_cms_obs"
            ],
            "gamma_norm_cols": [],
            "pbm_norm": False,
        },
        var_c=['dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx', 'run_mm_syr', 'inu_pc_smn', 'inu_pc_smx', 'pre_mm_s01', 'pre_mm_s02', 'pre_mm_s03', 'pet_mm_syr'],
        # scaler="DapengScaler",
        scaler=scaler,
        var_out=["q_cms_obs"],
        dataset="StreamflowDataset",
        train_epoch=2,
        save_epoch=1,
        train_period=["2010-01-01", "2019-12-31"],
        valid_period=["2010-01-01", "2019-12-31"],
        test_period=["2017-01-01", "2019-12-31"],
        # train_period=["1980-01-01", "1981-12-31"],
        # valid_period=["2010-01-01", "2013-12-31"],
        # test_period=["2014-01-01", "2015-12-31"],
        loss_func="RMSESum",
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0],
        #     "device": [2],
        #     "item_weight": [1],
        # },
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        # lr_scheduler={
        #     epoch: (
        #         0.5
        #         if 1 <= epoch <= 5
        #         else (
        #             0.2
        #             if 6 <= epoch <= 10
        #             else (
        #                 0.1
        #                 if 11 <= epoch <= 15
        #                 else 0.05 if 16 <= epoch <= 20 else 0.02
        #             )
        #         )
        #     )
        #     for epoch in range(1, 21)
        # },
        which_first_tensor="sequence",
        # calc_metrics=True,
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        rolling=0,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=2,
        model_type="Normal",
        #valid_batch_mode="train",
        # valid_batch_mode="test",
        #evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            #"eval_way": "1pace",
            # "pace_idx": -1,
            #"pace_idx": -1,
        #},
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
