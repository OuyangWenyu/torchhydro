"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2024-02-14 16:12:50
LastEditors: Wenyu Ouyang
Description: A test function for transfer learning
FilePath: \torchhydro\tests\test_transfer_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

DEVICE = 1
  
@pytest.mark.requires_data
@pytest.mark.skip(reason="Gages/Mopex/SupData4Camels data sources removed from torchhydro (moved to hydrodatasource)")
def test_transfer_gages_lstm_model():
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "simple_lstm_DapengScaler",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    
    project_name = "test_camels/exp_use_rain"
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
            ),
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "songliaorrevent",
                "offset_to_utc": True,
                # "net_rain_key": "net_rain",
                "obs_flow_key": "inflow",
                # "delta_t_hours": 3.0,
            },
        },
        # source_region="US",
        ctx=[DEVICE],
        gage_id=[
            'songliao_20800900', 
            'songliao_20810200', 
            'songliao_21100150', 
            'songliao_21110150', 
            'songliao_21113800', 
            'songliao_21401050', 
            'songliao_21401550', 
        ],
        model_type="TransLearn",
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 41,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
            "wconv": True,
        },
        
        batch_size=8,
        hindcast_length=0,
        forecast_length=20,
        frwin=20,
        min_time_unit="D",
        min_time_interval="1",
        var_t=[
            # "total_precipitation_hourly",
            "rain",
            # "flood_event",
            "temperature_2m",
            "surface_net_solar_radiation_hourly",
            "surface_net_thermal_radiation_hourly",
            "snowfall_hourly",
            "surface_pressure"
            ],
        t_rm_nan=False,
        var_c=[            
            "area", # shp_area
            "pet_mm_syr",
            "aet_mm_syr",
            "ele_mt_sav",
            "lka_pc_sse",
            "for_pc_sse",
            "slp_dg_sav",
            "lkv_mc_usu",
            "ero_kh_sav",
            "tmp_dc_syr",
            "riv_tc_usu",
            "gdp_ud_sav",
            "soc_th_sav",
            "kar_pc_sse",
            "ppd_pk_sav",
            "nli_ix_sav",
            "pst_pc_sse",
            "pac_pc_sse",
            "snd_pc_sav",
            "pop_ct_usu",
            "swc_pc_syr",
            "ria_ha_usu",
            "slt_pc_sav",
            "cly_pc_sav",
            "crp_pc_sse",
            "inu_pc_slt",
            "cmi_ix_syr",
            "snw_pc_syr",
            "ari_ix_sav",
            "ire_pc_sse",
            "rev_mc_usu",
            "inu_pc_smn",
            "urb_pc_sse",
            "prm_pc_sse",
            "gla_pc_sse"
            ],
        c_rm_nan=False,
        var_out=[
            "inflow",
            "flood_event"
            ],
        rs=1234,
        train_period=["2000-01-01","2014-12-31"],
        test_period=["2020-01-01","2023-12-31"],
        scaler="DapengScaler",
        sampler="KuaiSampler",
        # dataset="FloodEventDataset",
        dataset="FloodEventDataset",
        weight_path=weight_path,
        stat_dict_file = "/home/zlh/zhuanglaihong/torchhydro/results/test_camels/exp_use_rain/dapengscaler_stat.json",
        # weight_path_add={
        #     "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        # },
        continue_train=False,
        train_epoch=20,
        # te=20,
        save_epoch=1,
        train_mode=False,
        opt="Adam",
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "device": [DEVICE],
        },
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=1,
        calc_metrics = True,
        evaluator={"eval_way": "floodevent"},
        
    )
    cfg = default_config_file()
    update_cfg(cfg, args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
