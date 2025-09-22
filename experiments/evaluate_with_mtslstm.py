"""
Author: Wenyu Ouyang / adapted for eval (MTSLSTM hourly-unified)
Date: 2025-01-10
Description: Evaluate a trained MTSLSTM model on CAMELS-Hourly (new model: per-feature down-aggregation, no upsampling)
"""

import logging
import os
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camelshourly import CamelsHourly

# 可选：若本文件在 experiments/ 下，确保上层在 sys.path 中
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

# ====== 日志 ======
logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

# ====== 数据集与流域 ======
camels_dir = "/Users/cylenlc/data/camels_hourly"
camels = CamelsHourly(camels_dir)
gage_id = ["01054200"]
gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# ====== 实验目录与已训练权重 ======
PROJECT_DIR = "/Users/cylenlc/work/torchhydro/experiments"
# ⚠️ 改成你训练新模型时的子目录名称
RUN_SUBDIR   = "camels/mtslstm_DapengScaler_h-unified_3freq_14dwin"

STAT_PATH = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "dapengscaler_stat.json")
BEST_PTH  = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "best_model.pth")

# ====== 模型多频配置（需与训练保持一致）======
# 周(0) / 日(1) / 时(2=最高频=输入)
FACS     = [7, 24]          # 周->日×7，日->时×24
SEQ_LENS = [2, 14, 14*24]   # 评估窗口的“定义步长”，供切片计算（这里与训练示例一致）

# 动态变量（小时粒度的原始输入顺序，必须与训练时 var_t 一致）
var_t = [
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
]

# 静态变量（顺序需与训练时 var_c 一致）
var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]

# ====== 特征频率桶与聚合方式（需与训练完全一致）======
# 频率桶：0=周, 1=日, 2=时（最高频）
bucket_map_dyn = {
    "convective_fraction": 2,   # 小时
    "longwave_radiation": 1,    # 天
    "potential_energy": 0,      # 周
    "potential_evaporation": 1,
    "pressure": 1,
    "shortwave_radiation": 1,
    "specific_humidity": 2,
    "temperature": 2,
    "total_precipitation": 2,   # 保留在小时；若向低频下采样，用 sum
    "wind_u": 2,
    "wind_v": 2,
}
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]

# 静态通常直接并到最高频（不参与下采样），这里统一放到 2（小时层）
feature_buckets_sta = [2] * len(var_c)

# 拼接后的 buckets：顺序 = 动态在前 + 静态在后（与数据管线中拼接顺序一致）
feature_buckets = feature_buckets_dyn + feature_buckets_sta

# 聚合方式：降水/径流 "sum"，其它 "mean"；静态恒值用 "mean"
agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

# ====== 评估配置 ======
scaler = "DapengScaler"
seeds = 111

def config():
    cfg = default_config_file()

    args = cmd(
        # —— 评估模式 —— #
        train_mode=False,
        project_dir=PROJECT_DIR,
        stat_dict_file=STAT_PATH,   # 训练保存的归一化统计
        sub=RUN_SUBDIR,             # 用训练时的子目录名，便于默认路径解析

        # 数据源
        source_cfgs={
            "source_name": "camels_hourly",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },

        ctx=[0],

        # —— 模型名与超参（必须与训练一致）—— #
        model_name="MTSLSTM",
        model_hyperparam={
            # 新模型（小时统一输入；按特征下采样聚合）
            "hidden_sizes": [64, 64, 64],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": True,

            # 关键：走新路径，不用 auto_build_lowfreq
            "auto_build_lowfreq": False,

            # 与训练一致的多频设置
            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,

            # 切片传递（NH 固定切片）
            "slice_transfer": True,
            "slice_use_ceil": True,

            # 以下兼容旧路径的冗余项，不影响新路径
            "build_factor": 24,
            "agg_reduce": "mean",
            "truncate_incomplete": True,
        },

        # —— 指定加载权重 —— #
        model_loader={
            # 方式一：直接加载 best（推荐，路径会按 project_dir/sub 去找）
            "load_way": "best",
            # 方式二：若想手动指定 .pth，改成：
            # "load_way": "pth",
            # "pth_path": BEST_PTH,
        },

        gage_id=gage_id,

        rolling=0,
        batch_size=1,
        forecast_history=0,
        forecast_length=SEQ_LENS[-1],

        # 时间粒度（小时）
        min_time_unit="h",
        min_time_interval=1,

        var_t=var_t,
        var_c=var_c,
        var_out=["qobs_mm_per_hour"],

        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },

        train_period=["1990-01-01", "1991-12-31"],
        valid_period=["1990-01-01", "1991-12-31"],
        test_period=["1990-01-01", "1991-12-31"],

        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=False,
        patience=2,
        model_type="Normal",

        valid_batch_mode="test",
        dataset="CamelsHourlyDataset"
    )
    setattr(args, "evaluator", {
        "eval_way": "once",
        "stride": 0,
        "return_key": "f2"
    })
    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    cfgs = config()
    # 小检查，避免路径问题
    if not os.path.isfile(STAT_PATH):
        logging.warning(f"[Warn] stat_dict_file 不存在：{STAT_PATH}")
    best_default = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "best_model.pth")
    if not os.path.isfile(best_default) and not os.path.isfile(BEST_PTH):
        logging.warning(f"[Warn] 未找到 best_model.pth：{best_default} 或 {BEST_PTH}")

    train_and_evaluate(cfgs)
