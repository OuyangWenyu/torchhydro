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
from torchhydro.datasets.data_dict import s_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed, save_result

warnings.filterwarnings("ignore")


@pytest.fixture()
def config_data():
    project_name = "test_evalute_spp_lstm/ex1"
    train_path = os.path.join(os.getcwd(), "results", "test_spp_lstm", "ex3_0")
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        streamflow_source_path=r"C:\Users\Administrator\.hydrodataset\cache\merge_streamflow.nc",
        rainfall_source_path=r"C:\Users\Administrator\PycharmProjects\AIFloodForecast\test_data\biliuhe",
        attributes_path=r"C:\Users\Administrator\PycharmProjects\AIFloodForecast\test_data\camelsus_attributes.nc",
        gfs_source_path="",
        download=0,
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 60,
            "dropout": 0.25,
            "len_c": 15,
            "in_channels": 1,  # 卷积层输入通道数，等于len(var_t)
            "out_channels": 32,  # 输出通道数
        },
        gage_id=["86_21401550"],
        batch_size=256,
        var_t=[["tp"]],
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
        user='zxw'
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_evaluate_spp_lstm(config_data):
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    _name = data_cfgs["_name"]
     = s_dict[_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    model = DeepHydro(, config_data)
    test_acc = model.model_evaluate()
    print("summary test_accuracy", test_acc[0])
    save_result(
        config_data["data_cfgs"]["test_path"],
        "0",
        test_acc[1],
        test_acc[2],
    )
