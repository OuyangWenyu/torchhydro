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
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent.parent, "data/basin_aus.csv"),
    dtype={"id": str},
)
gage_id = show["id"].values.tolist()
# print(gage_id)

def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        "train_with_camelsaus", "camelsaus_test"
    )
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": "/home/estelle/HydroDataCompiler-main/notebooks/camelsaus_test",
            "other_settings": {"time_unit": ["1D"]},
        },
        # sampler="BasinBatchSampler",
        # scaler="DapengScaler",
        ctx=[0],
        # dataset="StreamflowDataset",
        dataset="Seq2SeqDataset",
        sampler="BasinBatchSampler",
        scaler="DapengScaler",
        model_loader={"load_way": "best"},
        model_name="KuaiLSTM",
        model_hyperparam={
            "n_input_features": 16,
            "n_output_features": 1,
            "n_hidden_states": 30,
            "dr": 0.2,
        },
        gage_id=gage_id,
        train_period=["1980-01-01", "2020-01-01"],
        test_period=["2015-01-01", "2020-01-01"],
        valid_period=["2020-01-01", "2022-01-01"],
        batch_size=100,
        hindcast_length=30,
        forecast_length=30,
        min_time_unit="D",
        min_time_interval=1,
        var_out=["streamflow"],
        var_t=[
            "total_precipitation_hourly",
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
        train_epoch=30,
        save_epoch=1,
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        which_first_tensor="batch",
        calc_metrics=False,
        early_stopping=True,
        patience=10,
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)