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
sanxia_dir = "/Volumes/Untitled/data"
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
    "surface_net_solar_radiation_hourly":0,
    "snowfall_hourly": 0,
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
        sub=project_sub,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            "other_settings": {"dataset_name": "sanxia", "time_unit": ["1h"]},
        },
        ctx=[0],
        model_name="MTSLSTM",

        model_hyperparam={
            "pretrained_flag": True,
            "linear1_size": 41,
            "linear2_size": 128,
            "pretrained_day_path": PRETRAIN_DAY_PTH,
            "pretrained_lstm_prefix": "lstm.",
            "pretrained_head_prefix": "linearOut.",
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
                "streamflow",
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

        model_loader={"load_way": "best"},
        dataset="StreamflowDataset"
    )

    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    cfgs = config()
    train_and_evaluate(cfgs)



'''import logging
import os.path
import pandas as pd

import sys
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels
#from hydrodataset.camels_aef import CamelsAef

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

PRETRAIN_DAY_PTH = '/Users/cylenlc/work/torchhydro/experiments/best_model.pth'

# === Dataset setup ===
sanxia_dir = "/Volumes/Untitled/data/sanxia"
gage_id = ['sanxia_60406350', 'sanxia_60406500', 'sanxia_60406700', 'sanxia_60407100', 'sanxia_60407200', 'sanxia_60701101', 'sanxia_60711600', 'sanxia_60713400', 'sanxia_60713630', 'sanxia_60713800', 'sanxia_60717050']

gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

length = 7
dim = 128
scaler = "DapengScaler"
# scaler = "StandardScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        #project_dir="D:\\torchhydro\\text2attr",
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            #"other_settings": {"download": False, "region": "US"},
            "other_settings": {"time_unit": ["1h"]},
        },
        ctx=[1],
        model_name="SLSTM",
        model_hyperparam={
            "input_size": 39,
            "output_size": 1,
            "hidden_size": 128,
            "pretrained_path": PRETRAIN_DAY_PTH,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
        # gage_id=gage_id[5000:5009],
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=256,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="h",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "precipitation_obs",
            ],
            "pbm_norm": True,
        },
        var_t=[
            "precipitation_obs",
            "temperature_2m",
            "surface_net_solar_radiation_hourly",
            "snowfall_hourly",
        ],
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
    ],
        # scaler="DapengScaler",
        scaler=scaler,
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        train_epoch=20,
        save_epoch=1,
        train_period=["2021-01-02", "2024-12-31"],
        valid_period=["2022-01-01", "2022-12-31"],
        test_period=["2024-01-01", "2024-12-31"],
        # train_period=["1980-01-01", "1981-12-31"],
        # valid_period=["2010-01-01", "2013-12-31"],
        # test_period=["2014-01-01", "2015-12-31"],
        loss_func="RMSESum",
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0],
        #     "device": [2],
        #     "item_weight": [1],
        # },
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        # lr_scheduler={
        #     epoch: (
        #         0.5
        #         if 1 <= epoch <= 5
        #         else (
        #             0.2
        #             if 6 <= epoch <= 10
        #             else (
        #                 0.1
        #                 if 11 <= epoch <= 15
        #                 else 0.05 if 16 <= epoch <= 20 else 0.02
        #             )
        #         )
        #     )
        #     for epoch in range(1, 21)
        # },
        which_first_tensor="sequence",
        # calc_metrics=True,
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        rolling=0,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=2,
        model_type="Normal",
        #valid_batch_mode="train",
        # valid_batch_mode="test",
        #evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            #"eval_way": "1pace",
            # "pace_idx": -1,
            #"pace_idx": -1,
        #},
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)'''
