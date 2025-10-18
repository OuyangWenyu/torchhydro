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
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels
from hydrodataset.camels_sk_aqua import CamelsSk
import torch
torch.backends.cudnn.enabled = False
# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "04"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("/ftproot/camels_data")
camels = CamelsSk(camels_dir)
gage_id = ['2008630', '2008635', '2008690', '2003640', '2008645', '2017620', '2014690', '2014620', '2014640', '2014660', '2012696', '2014680', '2020615', '2011660', '2011650', '2011640', '2020650', '2009620', '2009670', '2020675', '2013690', '2013685', '2022610', '2022640', '2018655', '2022680', '2018665', '2017685', '2012640', '2019610', '2007660', '2012665', '2007620', '2007640', '2016680', '2012650', '2012653', '2003690', '2002655', '2016650', '2004625', '2005680', '2012670', '2002692', '2019695', '2013650', '2003670', '2012628', '2019655', '2004605', '2012660', '2019635', '2020603', '2012695', '2006665', '2021640', '2019615', '2021665', '2004655', '2004640', '2018674', '2004650', '2004675', '2004680', '2002685', '2003610', '2018620', '2021685', '2021690', '2021677', '2021675', '2013615', '2006625', '2004695', '1005695', '1014680', '2010690', '2004668', '2012648', '2018685', '2018680', '1007680', '1014650', '2022655', '2022670', '2022660', '1015644', '2012625', '2006630', '2018645', '1014640', '2005660', '2010650', '1014620', '2004635', '1006660', '1018665', '1007615', '1007617', '2012690', '1006690', '2015635', '1007655', '2012652', '1014630', '2012645', '2001658', '1007605', '1006680', '1006672', '1007640', '1022630', '1018620', '1018630', '1018625', '1018623', '1022655', '1022650', '1001620', '1006650', '1007650', '1007645', '1002675', '1022648', '1002685', '1002687', '2001630', '2018635', '2018630', '1022640', '1022670', '1018675', '1002635', '1013655', '1016650', '1002698', '1022680', '1003620', '1018635', '1018695', '1023670', '1018650', '1016660', '1021650', '1021680', '1018693', '1018690', '1018697', '1018670', '1023660', '1018655', '1002650', '1006665', '1002640', '1005697', '1005640', '1001630', '1001655', '1002625', '1002605', '1002615', '1002610', '1015645', '1001670', '1001660', '1007625', '1007635', '1012630', '1013645', '1001625', '1018610', '1018640', '1007685', '1018662', '1018680', '1018683', '1019630']
gage_id = sorted([x for x in gage_id])

assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

length = 7
dim = 128
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

    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_sk",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "SK"},
        },
        ctx=[0,1,2],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 69,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        batch_size=256,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="h",
        min_time_interval=1,
        var_t=[
            "total_precipitation", 
            "temperature_2m", 
            "potential_evaporation", 
            "surface_net_solar_radiation", 
            "precip_obs"
        ],
        scaler_params={
            "prcp_norm_cols": [
                "q_cms_obs"
            ],
            "gamma_norm_cols": [],
            "pbm_norm": False,
        },
        var_c=[
            'dis_m3_pyr', 
            'dis_m3_pmn',
            'dis_m3_pmx',
            'run_mm_syr',
            'inu_pc_smn',
            'inu_pc_smx',
            'pre_mm_s01',
            'pre_mm_s02',
            'pre_mm_s03', 
            'pet_mm_syr',
        
        ],
        scaler=scaler,
        var_out=["q_cms_obs"],
        dataset="AlphaEarthDataset",
        train_epoch=20,
        save_epoch=1,
        train_period=["2016-01-01", "2017-12-31"],
        valid_period=["2018-01-01", "2018-12-31"],
        test_period=["2019-01-01", "2019-12-31"],
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },

        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        rolling=0,
        patience=2,
        model_type="Normal",
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)