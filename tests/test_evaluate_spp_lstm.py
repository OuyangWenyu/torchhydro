"""
Author: Xinzhuo Wu
Date: 2023-12-29 14:20:18
LastEditTime: 2023-12-29 11:05:57
LastEditors: Xinzhuo Wu
Description: A simple evaluate model test
FilePath: \torchhydro\tests\test_spp_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import pytest
import hydrodataset as hds
import warnings
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed, save_result

warnings.filterwarnings("ignore")


@pytest.fixture()
def config_data():
    project_name = "test_evalute_spp_lstm/ex1"
    train_path = os.path.join(os.getcwd(), "results", "test_spp_lstm", "ex2_0")
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 60,
            "dropout": 0.25,
        },
        gage_id=["86_21401550"],
        batch_size=256,
        var_t=["tp"],
        var_out=["streamflow"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        test_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
        ],  # 该范围为降水的时间范围，流量会整体往后推24h
        rolling=False,
        weight_path=os.path.join(train_path, "best_model.pth"),
        stat_dict_file=os.path.join(train_path, "GPM_GFS_Scaler_2_stat.json"),
        continue_train=False,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_evaluate_spp_lstm(config_data):
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    model = DeepHydro(data_source, config_data)
    test_acc = model.model_evaluate()
    print("summary test_accuracy", test_acc[0])
    save_result(
        config_data["data_cfgs"]["test_path"],
        "0",
        test_acc[1],
        test_acc[2],
    )
