"""
Author: Wenyu Ouyang / adapted for eval
Date: 2025-01-10
Description: Evaluate a trained MTSLSTM model on CAMELS-Hourly
"""

import logging
import os
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camelshourly import CamelsHourly


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

# ====== 数据集与流域 ======
camels_dir = "/Users/cylenlc/data/camels_hourly"
camels = CamelsHourly(camels_dir)
gage_id = ["01022500", "01031500"]
gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

PROJECT_DIR = "/Users/cylenlc/work/torchhydro/experiments"

scaler = "DapengScaler"
dim = 128
dr = 0.4
hru_delete = "01"
RUN_SUBDIR = f"camels/simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"

STAT_PATH = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "dapengscaler_stat.json")
PTH_PATH  = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "best_model.pth")

seeds = 111
ens = True

def config():
    # 默认配置
    config_data = default_config_file()

    args = cmd(
        train_mode=False,
        project_dir=PROJECT_DIR,
        stat_dict_file=STAT_PATH,
        sub=RUN_SUBDIR,
        source_cfgs={
            "source_name": "camels_hourly",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[0],
        model_name="MTSLSTM",
        model_hyperparam={
            # —— 与训练保持一致 —— #
            "input_sizes": 28,          # 只给一个数即可（自动扩成周/日两频）
            "hidden_sizes": [64, 64],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": False,
            # 自动构造低频 + 切片传递
            "auto_build_lowfreq": True,
            "build_factor": 7,          # 7 天聚 1 周
            "agg_reduce": "mean",
            # "per_feature_aggs": [...],
            "truncate_incomplete": True,
            "slice_transfer": True,
            "slice_use_ceil": True,
        },
        model_loader={
            "load_way": "pth",
            "pth_path": PTH_PATH,
        },

        gage_id=gage_id,
        batch_size=384,
        rs=seeds,
        ensemble=ens,                    # 若训练时用了集成评估同样生效（会对同一权重做重复 seed 推理）
        ensemble_items={"seeds": seeds},

        # 预测窗口设置（与训练一致即可）
        forecast_history=0,
        forecast_length=365,

        # 时间粒度（这里是以日为高频，内部会自动生成周频）
        min_time_unit="D",
        min_time_interval=1,

        # 动态气象变量（与训练一致）
        var_t=[
            "convective_fraction",
            "longwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "pressure",
            "shortwave_radiation",
            "specific_humidity",
            "temperature",
            "total_precipitation",
            "wind_u",
            "wind_v",
        ],

        # 归一化/标准化参数（与训练一致）
        scaler_params={
            "prcp_norm_cols": ["streamflow"],    # 目标/相关列按需列出
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },

        # 静态流域属性（与训练一致）
        var_c=[
            "elev_mean",
            "slope_mean",
            "area_gages2",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "geol_1st_class",
            "geol_2nd_class",
            "geol_porostiy",
            "geol_permeability",
        ],

        scaler=scaler,
        var_out=["qobs_mm_per_hour"],    # CAMELS-Hourly 的目标列（与你训练一致）

        dataset="CamelsHourlyDataset",

        # —— 评估不训练 —— #
        train_epoch=0,
        save_epoch=0,

        # 虽然不训练，仍给出时间段（有的管线会用到）
        train_period=["1990-01-01", "1991-12-31"],
        valid_period=["1990-01-01", "1991-12-31"],
        test_period=["1990-01-01", "1991-12-31"],

        # 损失、优化器配置在评估阶段不会用到，但给了也无妨
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={"lr_factor": 0.95},

        which_first_tensor="sequence",

        # 输出指标
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],

        early_stopping=False,
        rolling=0,
        patience=2,
        model_type="Normal",

        # —— 关键：在测试集上评估 —— #
        valid_batch_mode="test",
        evaluator={
            "eval_way": "once",
            "stride": 0
        },
    )

    update_cfg(config_data, args)
    return config_data


if __name__ == "__main__":
    cfgs = config()
    train_and_evaluate(cfgs)
