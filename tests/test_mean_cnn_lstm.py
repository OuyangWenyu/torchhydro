"""
Author: Wenyu Ouyang
Date: 2023-07-25 16:47:19
LastEditTime: 2023-11-29 17:27:01
LastEditors: Wenyu Ouyang
Description: Test a full training and evaluating process
FilePath: \torchhydro\tests\test_train_camels_lstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import pandas as pd

show = pd.read_csv("data/gage_id_test.csv", dtype={'id':str})
gage_id = show['id'].values.tolist()

@pytest.fixture()
def config():
    project_name = "test_mean_lstm/exp_cnn_lstm_100_usbasins_3yearto1year"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="CNN1D",
        source_path=os.path.join(hds.ROOT_DIR, "lstm_data"),
        source_region="US",
        download=0,
        ctx=[2],
        model_name="HybridLSTMModel",
        model_hyperparam={
            # "n_input_features": 1,
            "input_size": 12,
            "output_size": 1,
            "hidden_size": 256,
            "cnn_output_size": 12,
            "forecast_length": 24,
        },
        # gage_id=[
        #     "21401550"
        # ],
        gage_id=gage_id,
        batch_size=128,
        rho=168, # seq_length forecast_history forecast_length=1 linearIn
        # var_t=["p_mean"],
        var_t=['convective_fraction', 'longwave_radiation', 'potential_energy', 'potential_evaporation', 'pressure', 'shortwave_radiation', 'specific_humidity', 'temperature', 'total_precipitation', 'wind_u', 'wind_v', 'qobs_mm_per_hour'],
        # var_c=["None"],
        var_out=['qobs_mm_per_hour'],
        dataset="CNN1D_Dataset",
        # dataset = "GPM_GFS_Dataset",
        sampler = "HydroSampler",
        scaler="DapengScaler",
        train_epoch=10,
        save_epoch=1,
        te=9,
        #1979
        train_period=[
            {"start": "2015-07-01", "end": "2015-09-29"},
            {"start": "2016-07-01", "end": "2016-09-29"},
            {"start": "2017-07-01", "end": "2017-09-29"},
            {"start": "2018-07-01", "end": "2018-09-29"},
        ],
        test_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
        ],
        valid_period=[
            {"start": "2018-07-01", "end": "2018-09-29"},
        ],

        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 1
        lr_scheduler={1: 1e-2, 2: 5e-3, 3: 1e-3},
        which_first_tensor="sequence",
        is_tensorboard=False,
    )
    update_cfg(config_data, args)
    return config_data


def test_train_evaluate(config):
    train_and_evaluate(config)
