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
gage_id = ["2008650"]

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
        f"camels_test", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        train_mode=False,
        stat_dict_file=os.path.join(f'results/camels/simplelstm_DapengScaler_128_0.4_ens_{hru_delete}/dapengscaler_stat.json'),
        project_dir=Path(__file__).resolve().parent,
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
        model_loader={
            "load_way": "pth",
            "pth_path": os.path.join(f'results/camels/simplelstm_DapengScaler_128_0.4_ens_{hru_delete}/best_model.pth'),
        },
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
        train_epoch=10,
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
        rolling=1,
        patience=2,
        model_type="Normal",
        valid_batch_mode="train",
        evaluator={
            "eval_way": "1pace",
            "pace_idx": -1,
        },
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
