"""
Author: Wenyu Ouyang / adapted for eval (MTSLSTM hourly-unified)
Date: 2025-01-10
Description:
    Evaluate a trained Multi-Time-Scale LSTM (MTSLSTM) model on the CAMELS-Hourly dataset.
    This version supports per-feature down-aggregation and avoids upsampling.

Features:
    - Hourly-unified input pipeline with per-feature aggregation.
    - Multi-frequency architecture (weekly/daily/hourly).
    - Loads pretrained weights (best checkpoint or user-specified).
    - Supports deterministic evaluation with fixed sequence lengths.
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

# ====== Logging setup ======
logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

# ====== Dataset and basin setup ======
camels_dir = "/Users/cylenlc/data/camels_hourly"
camels = CamelsHourly(camels_dir)
gage_id = ["01054200"]
gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# ====== Project directory and trained weights ======
PROJECT_DIR = "/Users/cylenlc/work/torchhydro/experiments"
RUN_SUBDIR   = "camels/mtslstm_DapengScaler_h-unified_3freq_14dwin"

STAT_PATH = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "dapengscaler_stat.json")
BEST_PTH  = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "best_model.pth")

# ====== Multi-frequency configuration (must match training) ======
FACS     = [7, 24]          # Weekly->Daily ×7, Daily->Hourly ×24
SEQ_LENS = [2, 14, 14*24]   # Sequence lengths per branch: weekly, daily, hourly

# Dynamic variables (must match training var_t order)
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

# Static variables (must match training var_c order)
var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]

# Frequency bucket mapping
bucket_map_dyn = {
    "convective_fraction": 2,   # Hourly
    "longwave_radiation": 1,    # Daily
    "potential_energy": 0,      # Weekly
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
feature_buckets_sta = [2] * len(var_c)  # All static vars at hourly branch
feature_buckets = feature_buckets_dyn + feature_buckets_sta

# Per-feature aggregation map
agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

# ====== Evaluation configuration ======
scaler = "DapengScaler"
seeds = 111


def config():
    """Build evaluation configuration for MTSLSTM.

    Returns:
        dict: Configuration dictionary for evaluation with
            - Data source (CAMELS-Hourly)
            - Model hyperparameters (frequency buckets, hidden sizes, dropout, etc.)
            - Evaluation mode setup (no training, load checkpoint)
    """
    cfg = default_config_file()

    args = cmd(
        # —— Evaluation mode —— #
        train_mode=False,
        project_dir=PROJECT_DIR,
        stat_dict_file=STAT_PATH,
        sub=RUN_SUBDIR,

        # —— Data source —— #
        source_cfgs={
            "source_name": "camels_hourly",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },

        ctx=[0],

        # —— Model setup (must match training) —— #
        model_name="MTSLSTM",
        model_hyperparam={
            "hidden_sizes": [64, 64, 64],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": False,

            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,

            # Slice transfer settings
            "slice_transfer": True,
            "slice_use_ceil": True,
        },

        # —— Model loader —— #
        model_loader={
            "load_way": "best",
            # Alternative (manual path):
            # "load_way": "pth",
            # "pth_path": BEST_PTH,
        },

        gage_id=gage_id,

        # —— Evaluation setup —— #
        rolling=0,
        batch_size=1,
        forecast_history=0,
        forecast_length=SEQ_LENS[-1],

        min_time_unit="h",
        min_time_interval=1,

        var_t=var_t,
        var_c=var_c,
        var_out=["qobs_mm_per_hour"],

        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": ["qobs_mm_per_hour"],
            "gamma_norm_cols": [],
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
    # Evaluator settings
    setattr(args, "evaluator", {
        "eval_way": "once",
        "stride": 0,
        "return_key": "f2"
    })
    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    """Main entry point for evaluation.

    - Builds configuration with `config()`.
    - Verifies required files (stat_dict, best_model checkpoint).
    - Calls `train_and_evaluate(cfgs)` in evaluation mode.
    """
    cfgs = config()
    if not os.path.isfile(STAT_PATH):
        logging.warning(f"[Warn] stat_dict_file not found: {STAT_PATH}")
    best_default = os.path.join(PROJECT_DIR, "results", RUN_SUBDIR, "best_model.pth")
    if not os.path.isfile(best_default) and not os.path.isfile(BEST_PTH):
        logging.warning(f"[Warn] best_model.pth not found at {best_default} or {BEST_PTH}")

    train_and_evaluate(cfgs)
