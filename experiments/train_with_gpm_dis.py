"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-27 15:49:30
LastEditors: Wenyu Ouyang
Description: 
FilePath: \torchhydro\experiments\train_with_gpm_dis.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pathlib
import pandas as pd
import torch.multiprocessing as mp
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.trainers.deep_hydro import train_worker

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv("data/basin_id(498+24).csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


def main():
    config_data = create_config()
    test_seq2seq(config_data)


def create_config():
    project_name = "train_with_gpm_dis/ex1"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[0, 1, 2],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 17,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 3,
            "prec_window": 1,
        },
        model_loader={"load_way": "best"},
        # gage_id=gage_id,
        gage_id=["21400800", "21401550"],
        batch_size=1024,
        forecast_history=240,
        forecast_length=3,
        min_time_unit="H",
        min_time_interval=3,
        var_t=[
            "gpm_tp",
            "sm_surface",
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
        sampler="DistSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_period=[("2016-06-01-01", "2023-12-01-01")],
        test_period=[("2015-06-01-01", "2016-06-01-01")],
        valid_period=[("2015-06-01-01", "2016-06-01-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.003,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        rolling=False,
        static=False,
        early_stopping=True,
        patience=8,
        model_type="DDP_MTL",
    )

    update_cfg(config_data, args)

    return config_data


def test_seq2seq(config):
    world_size = len(config["training_cfgs"]["device"])
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
