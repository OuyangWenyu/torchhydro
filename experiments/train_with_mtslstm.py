"""
Train MTSLSTM (hourly-unified input; no upsampling; per-feature down-aggregation).
"""

import logging
import os
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camelshourly import CamelsHourly

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = "/Users/cylenlc/data/camels_hourly"
camels = CamelsHourly(camels_dir)
gage_id = ['01022500', '01031500']
gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# ---- 频率设计：周(0) / 日(1) / 时(2=最高频=原始输入) ----
# 用 14 天窗口训练：T_hour = 14*24, T_day = 14, T_week = 2
FACS = [7, 24]               # 低->中×7，中->高×24
SEQ_LENS = [2, 14, 14*24]    # [周步数, 日步数, 小时步数]

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

var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]

# 频率桶：0=周, 1=日, 2=时（最高频）
bucket_map_dyn = {
    "convective_fraction": 2,     # 小时
    "longwave_radiation": 1,      # 天
    "potential_energy": 0,        # 周
    "potential_evaporation": 1,
    "pressure": 1,
    "shortwave_radiation": 1,
    "specific_humidity": 2,
    "temperature": 2,
    "total_precipitation": 2,
    "wind_u": 2,
    "wind_v": 2,
}
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]

# 静态全部放在小时层（不会做下采样），便于长度对齐
feature_buckets_sta = [2] * len(var_c)

# >>> 最终传给模型的 buckets（顺序必须和实际拼接顺序一致：动态在前，静态在后） <<<
feature_buckets = feature_buckets_dyn + feature_buckets_sta    # 长度应为 28

# 聚合方式：降水/径流 "sum"，其它 "mean"；静态用 "mean"
agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)

per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta  # 长度 28

# ---- 其它训练超参 ----
scaler = "DapengScaler"
seeds = 111
project_sub = f"camels/mtslstm_{scaler}_h-unified_3freq_14dwin"

def config():
    cfg = default_config_file()

    args = cmd(
        sub=project_sub,
        source_cfgs={
            "source_name": "camels_hourly",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[0],
        model_name="MTSLSTM",

        model_hyperparam={
            # 不再需要手写 input_sizes；让模型从 feature_buckets 内部推导各分支维度
            # "input_sizes": 省略/None,

            "hidden_sizes": [64, 64, 64],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": True,

            # "auto_build_lowfreq": False,

            "feature_buckets": feature_buckets,      # 长度 = len(var_t)
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,               # [7, 24] 低->中，中->高
            "seq_lengths": SEQ_LENS,                 # [2, 14, 14*24]

            # NH 风格的“切片传递”：先跑到分界点（按 seq_lengths & factors 算），再迁移状态
            "slice_transfer": True,
            "slice_use_ceil": True,  # 只要提供了 seq_lengths/factors，内部就用固定切片，不再依赖 ceil/floor
            # 下面几个是聚合时的冗余参数（仅在 fallback 或你强制用旧路径时有用）
            # "build_factor": 24,
            # "agg_reduce": "mean",
            # "truncate_incomplete": True,
        },

        # ----------------- 训练设置 -----------------
        gage_id=gage_id,

        # 两种训练方式任选其一：

        # A) 固定窗口滑动训练（推荐，便于 batch>1）：
        rolling=0,
        forecast_length=SEQ_LENS[-1],   # = 14*24 小时窗口
        batch_size=256,

        # B) 全序列训练（rolling=0）：若不同流域长度不一致，建议 batch_size=1 或写 padding-collate
        # rolling=0,
        # batch_size=1,

        rs=seeds,
        ensemble=True,
        ensemble_items={"seeds": seeds},

        min_time_unit="h",
        min_time_interval=1,

        var_t=var_t,
        var_c=[
            "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
            "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
            "soil_porosity","soil_conductivity","max_water_content",
            "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
        ],
        var_out=["qobs_mm_per_hour"],

        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": ["qobs_mm_per_hour"],
            "gamma_norm_cols": [],
            "pbm_norm": False,
        },

        # 训练/验证/测试时间
        train_epoch=2,
        save_epoch=1,
        train_period=["1990-01-01", "1991-12-31"],
        valid_period=["1990-01-01", "1991-12-31"],
        test_period=["1990-01-01", "1991-12-31"],

        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 1e-4},
        lr_scheduler={"lr_factor": 0.95},

        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        patience=2,
        model_type="Normal",

        # 把最好模型保存/加载
        model_loader={"load_way": "best"},
        dataset="CamelsHourlyDataset"
    )

    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    cfgs = config()
    train_and_evaluate(cfgs)
