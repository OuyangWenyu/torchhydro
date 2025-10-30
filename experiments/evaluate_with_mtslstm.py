import logging
import os
import sys
from pathlib import Path

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
#from hydrodataset.camelshourly import CamelsHourly

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

PRETRAIN_DAY_PTH = '/Users/cylenlc/Desktop/simple_lstm_DapengScaler_128_0.4/best_model.pth'

# === Dataset setup ===
sanxia_dir = "/Volumes/Untitled/data/sanxia"
#camels = CamelsHourly(camels_dir)
gage_id = ['sanxia_60406350', 'sanxia_60406500', 'sanxia_60406700', 'sanxia_60407100', 'sanxia_60407200', 'sanxia_60701101', 'sanxia_60711600', 'sanxia_60713400', 'sanxia_60713630', 'sanxia_60713800', 'sanxia_60717050']

gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# === Frequency setup: daily(0) / hourly(1=highest) ===
# 14-day window → 14 steps (daily) or 14×24 steps (hourly)
FACS = [24]                 # daily → hourly ×24
SEQ_LENS = [14, 14 * 24]    # [daily steps, hourly steps]

# === Dynamic variables ===
# 低频变量放 daily，高频变量放 hourly
bucket_map_dyn = {
    "precipitation_obs": 1,
    "temperature_2m": 1,
    "surface_net_solar_radiation_hourly": 1,
    "snowfall_hourly": 1,
}

# var_t = list(bucket_map_dyn.keys())
var_t = [
    "precipitation_obs",
    "temperature_2m",
    "surface_net_solar_radiation_hourly",
    "snowfall_hourly",
]
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]

var_c = [
    "area",  # shp_area
    "pet_mm_syr",
    "aet_mm_syr",
    "ele_mt_sav",
    "lka_pc_sse",
    "for_pc_sse",
    "slp_dg_sav",
    "lkv_mc_usu",
    "ero_kh_sav",
    "tmp_dc_syr",
    "riv_tc_usu",
    "gdp_ud_sav",
    "soc_th_sav",
    "kar_pc_sse",
    "ppd_pk_sav",
    "nli_ix_sav",
    "pst_pc_sse",
    "pac_pc_sse",
    "snd_pc_sav",
    "pop_ct_usu",
    "swc_pc_syr",
    "ria_ha_usu",
    "slt_pc_sav",
    "cly_pc_sav",
    "crp_pc_sse",
    "inu_pc_slt",
    "cmi_ix_syr",
    "snw_pc_syr",
    "ari_ix_sav",
    "ire_pc_sse",
    "rev_mc_usu",
    "inu_pc_smn",
    "urb_pc_sse",
    "prm_pc_sse",
    "gla_pc_sse"
]
# 静态特征全部放在小时层（不参与下采样）
feature_buckets_sta = [1] * len(var_c)

# === Final feature buckets (dynamic + static) ===
feature_buckets = feature_buckets_dyn + feature_buckets_sta

# === Per-feature aggregation setup ===
# 日尺度 → mean 或 sum（降水），小时尺度 → mean
agg_map_dyn = {v: "mean" for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

# === Training setup ===
scaler = "DapengScaler"
seeds = 111
project_sub = f"camels/mtslstm_{scaler}_d-h_2freq_14dwin"


def config():
    """Build configuration dictionary for daily-hourly MTSLSTM training."""
    cfg = default_config_file()

    args = cmd(
        train_mode=False,
        stat_dict_file=r"/Users/cylenlc/work/torchhydro/experiments/results/camels/mtslstm_DapengScaler_d-h_2freq_14dwin/dapengscaler_stat.json",
        project_dir=r"experiments",
        sub=project_sub,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            #"other_settings": {"download": False, "region": "US"},
            "other_settings": {"time_unit": ["1h"]},
        },
        ctx=[0],
        model_name="MTSLSTM",

        model_hyperparam={
            #"pretrained_day_path": PRETRAIN_DAY_PTH,
            #"pretrained_lstm_prefix": "lstm.",
            #"pretrained_head_prefix": "linearOut.",
            "hidden_sizes": [128, 128],
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.4,
            "return_all": True,

            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,

            # 启用跨尺度状态传递（日 → 时）
            "slice_transfer": True,
            "slice_use_ceil": True,
        },
        model_loader={
            "load_way": "pth",
            "pth_path": f"/Users/cylenlc/work/torchhydro/experiments/results/camels/mtslstm_DapengScaler_d-h_2freq_14dwin/best_model.pth",
        },

        # === Training setup ===
        gage_id=gage_id,
        rolling=0,                           # sliding window
        forecast_length=SEQ_LENS[-1],        # 14 days × 24h
        batch_size=256,

        rs=seeds,
        ensemble=True,
        ensemble_items={"seeds": seeds},

        min_time_unit="h",
        min_time_interval=1,

        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],

        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": [
                #"streamflow",
            ],
            "gamma_norm_cols": [
                #"precipitation_obs",
            ],
            "pbm_norm": False,
        },

        # === Training/validation/testing periods ===
        train_epoch=20,
        save_epoch=1,
        train_period=["2021-01-01", "2023-12-31"],
        valid_period=["2022-01-01", "2022-12-31"],
        test_period=["2024-01-01", "2024-12-31"],

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

        #model_loader={"load_way": "best"},
        dataset="StreamflowDataset",
        evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            "eval_way": "1pace",
            # "pace_idx": -1,
            "pace_idx": -1,
        },
    )

    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    cfgs = config()
    train_and_evaluate(cfgs)

