"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2024-04-17 13:31:16
LastEditors: Xinzhuo Wu
Description:
FilePath: /torchhydro/tests/test_train_seq2seq.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pathlib

import pandas as pd
import pytest
import hydrodatasource.configs.config as hdscc
import xarray as xr
import torch.multiprocessing as mp

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import train_worker

# from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent.parent, "data/basin_id(all).csv"),
    dtype={"id": str},
)
gage_id = show["id"].values.tolist()


def test_merge_forcing_and_streamflow():
    data_name = "era5land"
    for id in gage_id:
        concat_ds_path = f"s3://basins-origin/hour_data/1h/mean_data/data_forcing_{data_name}_streamflow/data_forcing_streamflow_{id}.nc"
        if not hdscc.FS.exists(concat_ds_path):
            forcing_nc = f"s3://basins-origin/hour_data/1h/mean_data/data_forcing_{data_name}/data_forcing_{id}.nc"
            stream_nc = f"s3://basins-origin/hour_data/1h/mean_data/streamflow_basin/streamflow_{id}.nc"
            if hdscc.FS.exists(forcing_nc) and hdscc.FS.exists(stream_nc):
                forcing = xr.open_dataset(hdscc.FS.open(forcing_nc))
                stream = xr.open_dataset(hdscc.FS.open(stream_nc))
                forcing_times = pd.to_datetime(forcing.time.values)
                stream_times = pd.to_datetime(stream.time.values)
                common_times = pd.Index(stream_times).intersection(
                    pd.Index(forcing_times)
                )
                forcing_common = forcing.sel(time=common_times)
                stream_common = stream.sel(time=common_times)
                combined_dataset = xr.merge([forcing_common, stream_common])
                hdscc.FS.write_bytes(concat_ds_path, combined_dataset.to_netcdf())
            else:
                print(f"File {id} dosen't exists. Skipping overwrite.")
        else:
            print(f"File {concat_ds_path} already exists. Skipping overwrite.")


@pytest.fixture()
def config():
    project_name = "test_mean_seq2seq/ex26"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[0, 1, 2],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 20,
            "output_size": 2,
            "hidden_size": 128,
            "forecast_length": 168,
            "prec_window": 3,
            "interval": 3,
        },
        model_loader={"load_way": "best"},
        gage_id=[
            "21401550",  # 碧流河
            # "01181000",
            # "01411300",
            # "01414500",
            # "02016000",
            # "02018000",
            # "02481510",
            # "03070500",
            # "08324000",
            # "11266500",
            # "11523200",
            # "12020000",
            # "12167000",
            # "14185000",
            "14306500",
        ],
        port="10086",
        # gage_id=gage_id,
        batch_size=1024,
        rho=672,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
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
        dataset="ERA5LandDataset",
        sampler="DistSampler",  # 使用多卡训练时必须使用DistSampler
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_period=[
            ("2015-06-01", "2022-12-20"),
        ],
        test_period=[
            ("2023-02-01", "2023-11-30"),
        ],
        valid_period=[
            ("2023-02-01", "2023-11-30"),  # 目前只支持一个时段
        ],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
            # "limit_part": [1],
        },
        opt="Adam",
        lr_scheduler={
            # 多卡训练时，先验地将学习率设置成显卡数*单卡学习率
            "lr": 0.003,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        rolling=False,
        static=False,
        early_stopping=True,
        patience=5,
        ensemble=True,
        ensemble_items={
            "kfold": 9,
            "batch_sizes": [1024],
        },
        model_type="DDP_MTL",
        fill_nan=["no", "no"],
    )
    update_cfg(config_data, args)
    return config_data


def test_seq2seq(config):
    world_size = len(config["training_cfgs"]["device"])
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    # train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
