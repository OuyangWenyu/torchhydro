import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.resulter import Resulter
from torchhydro.trainers.trainer import set_random_seed


logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_pretrain_pth = Path(
    os.environ.get(
        "PRETRAIN_DAY_PTH",
        "/home/litianli/torchhydro-feature-MTSLSTM/pth_file/best_model2.pth",
    )
).expanduser()
PRETRAIN_DAY_PTH = str(
    _pretrain_pth if _pretrain_pth.is_absolute() else (PROJECT_ROOT / _pretrain_pth)
)
if not Path(PRETRAIN_DAY_PTH).exists():
    raise FileNotFoundError(
        f"预训练模型文件未找到，请检查路径：{Path(PRETRAIN_DAY_PTH).resolve()}"
    )

_stat_dict_pth = Path(
    os.environ.get(
        "STAT_DICT_FILE",
        str(Path(PROJECT_ROOT / "pth_file" / "dapengscaler_stat.json")),
    )
).expanduser()
STAT_DICT_FILE = str(
    _stat_dict_pth if _stat_dict_pth.is_absolute() else (PROJECT_ROOT / _stat_dict_pth)
)
if not Path(STAT_DICT_FILE).exists():
    _fallback_stat = Path(PRETRAIN_DAY_PTH).with_name("dapengscaler_stat.json")
    if _fallback_stat.exists():
        STAT_DICT_FILE = str(_fallback_stat)
    else:
        raise FileNotFoundError(
            f"Scaler统计量文件未找到，请检查路径：{Path(STAT_DICT_FILE).resolve()}"
        )

_sanxia_dir = Path(os.environ.get("SANXIA_DATA_DIR", "/home/litianli/Dataset/")).expanduser()
sanxia_dir = str(_sanxia_dir if _sanxia_dir.is_absolute() else (PROJECT_ROOT / _sanxia_dir))
if not Path(sanxia_dir).exists():
    raise FileNotFoundError(f"数据目录未找到，请检查路径：{Path(sanxia_dir).resolve()}")

csv_path = Path(os.environ.get("SANXIA_GAGE_CSV", "data/223_floodevents.csv")).expanduser()
csv_path = csv_path if csv_path.is_absolute() else (PROJECT_ROOT / csv_path)
try:
    gage_df = pd.read_csv(csv_path)
    gage_id = gage_df["GAGE_ID"].dropna().tolist()
except FileNotFoundError:
    raise FileNotFoundError(f"CSV文件未找到，请检查路径：{csv_path.absolute()}")

gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

FIXED_HIDDEN_SIZES = [128, 128]
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

FACS = [24]
SEQ_LENS = [4, 96]

bucket_map_dyn = {
    "precipitation_obs": 1,
    "temperature_2m": 1,
    "surface_net_solar_radiation_hourly": 0,
    "snowfall_hourly": 0,
}

var_t = [
    "precipitation_obs",
    "temperature_2m",
    "surface_net_solar_radiation_hourly",
    "snowfall_hourly",
]
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]

var_c = [
    "area",
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
    "gla_pc_sse",
]
feature_buckets_sta = [0] * len(var_c)
feature_buckets = feature_buckets_dyn + feature_buckets_sta

agg_map_dyn = {
    "precipitation_obs": "sum",
    "temperature_2m": "mean",
    "surface_net_solar_radiation_hourly": "mean",
    "snowfall_hourly": "sum",
}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

scaler = "DapengScaler"
seeds = 111


def transfer_only_evaluate(cfgs: dict) -> None:
    set_random_seed(cfgs["training_cfgs"]["random_seed"])
    resulter = Resulter(cfgs)

    deephydro = DeepHydro(cfgs)
    test_dataloader = deephydro._get_dataloader(
        cfgs["training_cfgs"], cfgs["data_cfgs"], mode="infer"
    )
    pred_xr, obs_xr = deephydro.inference(test_dataloader, return_inputs=False)

    resulter.save_cfg(deephydro.cfgs)
    resulter.save_result(pred_xr, obs_xr)
    resulter.eval_result(pred_xr, obs_xr)


def config():
    hidden_str = "_".join(map(str, FIXED_HIDDEN_SIZES))
    project_sub = f"newbasin_223_LongSeq_223_2020_new/mtslstm_{scaler}_hidden_{hidden_str}_transfer_only"

    cfg = default_config_file()
    args = cmd(
        train_mode=False,
        continue_train=0,
        stat_dict_file=STAT_DICT_FILE,
        sub=project_sub,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            "other_settings": {
                "time_unit": ["1h"],
                "dataset_name": "223_FloodEvents_105_final",
                "delta_t_hours": 1.0,
            },
        },
        ctx=[1],
        model_name="MTSLSTM",
        model_hyperparam={
            "pretrained_flag": True,
            "pretrained_copy_to_all_branches": True,
            "linear1_size": 41,
            "linear2_size": 128,
            "pretrained_day_path": PRETRAIN_DAY_PTH,
            "pretrained_lstm_prefix": "lstm.",
            "pretrained_head_prefix": "linearOut.",
            "hidden_sizes": FIXED_HIDDEN_SIZES,
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.4,
            "return_all": True,
            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,
            "slice_transfer": True,
            "slice_use_ceil": True,
        },
        gage_id=gage_id,
        rolling=0,
        hindcast_length=72,
        forecast_length=24,
        warmup_length=0,
        batch_size=BATCH_SIZE,
        rs=seeds,
        ensemble=False,
        min_time_unit="h",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [],
            "pbm_norm": False,
        },
        test_period=["2020-01-01", "2024-12-31"],
        train_period=["2020-01-01", "2024-12-31"],
        valid_period=["2020-01-01", "2024-12-31"],
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        model_type="Normal",
        model_loader={"load_way": "pth", "pth_path": PRETRAIN_DAY_PTH},
        dataset="StreamflowDataset",
        evaluator={"eval_way": "once", "stride": 0},
    )

    update_cfg(cfg, args)
    return cfg


if __name__ == "__main__":
    print(f"⚠️  固定隐藏层结构：{FIXED_HIDDEN_SIZES}（{len(FIXED_HIDDEN_SIZES)}层）")
    print(f"批次大小：{BATCH_SIZE}\n")

    cfg = config()
    transfer_only_evaluate(cfg)

    print("\n所有实验完成！")
