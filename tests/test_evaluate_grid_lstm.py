"""
Author: Xinzhuo Wu
Date: 2023-12-29 14:20:18
LastEditTime: 2024-05-31 11:00:01
LastEditors: Wenyu Ouyang
Description: A simple evaluate model test
FilePath: \torchhydro\tests\test_evaluate_grid_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import warnings
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed
from torchhydro.trainers.resulter import Resulter

warnings.filterwarnings("ignore")


@pytest.fixture()
def config_data():
    project_name = "test_evalute_spp_lstm/ex1"
    train_path = os.path.join(os.getcwd(), "results", "test_spp_lstm", "ex1_0")
    args = cmd(
        sub=project_name,
        source="HydroGrid",
        source_path=[
            {
                "gpm": "basins-origin/hour_data/1h/grid_data/grid_gpm_data",
                "gfs": "basins-origin/hour_data/1h/grid_data/grid_gfs_data",
                "smap": "basins-origin/hour_data/1h/grid_data/grid_smap_data",
                "target": "basins-origin/hour_data/1h/mean_data/mean_data_target",
                "attributes": "basins-origin/attributes.nc",
            }
        ],
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "forecast_history": 168,
            "forecast_length": 24,
            "p_n_output": 1,
            "p_n_hidden_states": 60,
            "p_dropout": 0.25,
            "p_in_channels": 1,
            "p_out_channels": 8,
            "len_c": 15,
            "s_forecast_history": None,
            "s_n_output": 1,
            "s_n_hidden_states": 60,
            "s_dropout": 0.25,
            "s_in_channels": 8,
            "s_out_channels": 8,
        },
        gage_id=["21401550"],
        batch_size=256,
        var_t=[
            ["gpm_tp"],
            [
                "None",
            ],
            [
                "None",
            ],
        ],
        var_out=["streamflow"],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        dataset="GridDataset",
        sampler="HydroSampler",
        scaler="MutiBasinScaler",
        test_period=[
            ("2017-07-01", "2017-09-29"),
        ],
        rolling=False,
        weight_path=os.path.join(train_path, "best_model.pth"),
        stat_dict_file=os.path.join(train_path, "MutiBasinScaler_stat.json"),
        continue_train=False,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_evaluate_spp_lstm(config_data):
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    resulter = Resulter(config_data)
    model = DeepHydro(config_data)
    test_acc = model.model_evaluate()
    print("summary test_accuracy", test_acc[0])
    resulter.save_result(
        test_acc[0],
        test_acc[1],
    )
