"""
Author: Wenyu Ouyang
Date: 2024-05-22 09:34:45
LastEditTime: 2024-05-27 10:09:26
LastEditors: Wenyu Ouyang
Description: 
FilePath: \torchhydro\experiments\evaluate_with_gpm_streamflow.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import warnings
import logging
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed
from torchhydro.trainers.resulter import Resulter

logging.basicConfig(level=logging.INFO)

for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
show = pd.read_csv("data/basin_id(46+1).csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


def get_config_data():
    project_name = os.path.join("test_evaluate_seq2seq", "ex28")
    train_path = os.path.join(
        os.getcwd(), "results", "train_with_gpm_streamflow", "ex1"
    )
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
            },
        },
        ctx=[0],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 56,
            "hindcast_output_window": 1,
        },
        gage_id=gage_id,
        model_loader={"load_way": "best"},
        batch_size=1024,
        hindcast_length=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "gpm_tp",
            "sm_surface",
            "streamflow",
        ],
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
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        sampler="BasinBatchSampler",
        scaler="DapengScaler",
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        test_period=[("2015-06-01-01", "2016-05-31-01")],
        which_first_tensor="batch",
        rolling=56,
        weight_path=os.path.join(train_path, "best_model.pth"),
        stat_dict_file=os.path.join(train_path, "dapengscaler_stat.json"),
        train_mode=False,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def main():
    config_data = get_config_data()
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    resulter = Resulter(config_data)
    model = DeepHydro(config_data)
    results = model.model_evaluate()
    resulter.save_result(
        results[0],
        results[1],
    )


if __name__ == "__main__":
    main()
