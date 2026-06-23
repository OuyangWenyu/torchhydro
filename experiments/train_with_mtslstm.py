"""
Train MTSLSTM (hourly-unified input; no upsampling; per-feature down-aggregation).

This script sets up and trains the Multi-Time-Scale LSTM (MTSLSTM) model on the
CAMELS-hourly dataset. It supports hourly-unified inputs, per-feature aggregation,
and allows optional pretraining for the daily branch.

Features:
    - Hourly-to-daily/weekly down-aggregation.
    - Configurable sequence lengths per frequency scale.
    - Pretraining support for daily branch weights.
    - Flexible training setup with sliding-window or full-sequence training.
"""

import logging
import os
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camelsh import Camelsh

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

# === Pretrained checkpoint path for daily branch ===
PRETRAIN_DAY_PTH = "/Users/cylenlc/work/torchhydro/experiments/best_model.pth"

# === Dataset setup ===
camels_dir = "/Users/cylenlc/data/camels_hourly"
camels = Camelsh(camels_dir)
gage_id = ['01022500', '01031500']
gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# === Frequency setup: weekly(0) / daily(1) / hourly(2=highest) ===
# Window size = 14 days
# T_hour = 14*24, T_day = 14, T_week = 2
FACS = [7, 24]               # low->mid ×7, mid->high ×24
SEQ_LENS = [2, 14, 14*24]    # [weekly steps, daily steps, hourly steps]

# === Dynamic variables (time series inputs) ===
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

# === Static variables (catchment attributes) ===
var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]

# === Frequency bucket mapping ===
# 0 = weekly, 1 = daily, 2 = hourly
bucket_map_dyn = {
    "convective_fraction": 2,
    "longwave_radiation": 1,
    "potential_energy": 0,
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

# Static features all placed in hourly branch (no down-aggregation)
feature_buckets_sta = [2] * len(var_c)

# Final buckets (dynamic + static)
feature_buckets = feature_buckets_dyn + feature_buckets_sta    # length = 28

# === Per-feature aggregation setup ===
agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta  # length = 28

# === Training setup ===
scaler = "DapengScaler"
seeds = 111
project_sub = f"camels/mtslstm_{scaler}_h-unified_3freq_14dwin"


def config():
    """Build configuration dictionary for training MTSLSTM.

    This function constructs the experiment configuration, including:
      - Data source setup (CAMELS-hourly dataset).
      - Model hyperparameters (frequency buckets, hidden sizes, dropout, etc.).
      - Training hyperparameters (window size, batch size, optimizer, etc.).
      - Pretrained checkpoint for daily branch.

    Returns:
        dict: Experiment configuration ready for `train_and_evaluate`.
    """
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
            "pretrained_day_path": PRETRAIN_DAY_PTH,
            "pretrained_lstm_prefix": "lstm.",
            "pretrained_head_prefix": "linearOut.",
            "hidden_sizes": [128, 128, 128],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": True,

            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,

            # NeuralHydrology-style slice transfer
            "slice_transfer": True,
            "slice_use_ceil": True,
        },

        # === Training parameters ===
        gage_id=gage_id,
        rolling=0,                          # sliding window
        forecast_length=SEQ_LENS[-1],       # 14 days (hourly steps)
        batch_size=256,

        rs=seeds,
        ensemble=True,
        ensemble_items={"seeds": seeds},

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

        # === Training/validation/testing periods ===
        train_epoch=20,
        save_epoch=1,
        train_period=["1990-01-01", "1991-12-31"],
        valid_period=["1990-01-01", "1991-12-31"],
        test_period=["1990-01-01", "1991-12-31"],

        # === Optimization ===
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 1e-4},
        lr_scheduler={"lr_factor": 0.95},

        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        patience=2,
        model_type="Normal",

        model_loader={"load_way": "best"},
        dataset="CamelsHourlyDataset"
    )

    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    cfgs = config()
    train_and_evaluate(cfgs)
