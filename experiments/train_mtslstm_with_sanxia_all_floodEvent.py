import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


# === Logging setup ===
logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

PRETRAIN_DAY_PTH = "/home/litianli/torchhydro-feature-MTSLSTM/pth_file/best_model2.pth"

# === Dataset setup ===
sanxia_dir = "/home/litianli/Dataset/"
csv_path = Path("data/106_floodevents.csv")
try:
    gage_df = pd.read_csv(csv_path)
    gage_id = gage_df["GAGE_ID"].dropna().tolist()
except FileNotFoundError:
    raise FileNotFoundError(f"CSV文件未找到，请检查路径：{csv_path.absolute()}")

gage_id = sorted(gage_id)
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

# === 超参数设置（固定隐藏层，仅调整其他参数） ===
FIXED_HIDDEN_SIZES = [128, 128]  # 固定2层128单元（可按需修改）
LRS = [1e-3,2e-3,5e-4]  # 学习率候选
BATCH_SIZES = [ 128, 64, 36]  # 批次大小候选
EPOCHS = [20]  # 训练轮数候选
LR_FACTORS = [0.9, 0.95, 0.98]  # 学习率衰减因子候选++

# === 生成超参数组合（不含隐藏层，仅其他参数的笛卡尔积） ===
hyperparam_combinations = list(product(LRS, BATCH_SIZES, EPOCHS, LR_FACTORS))

# === 固定参数 ===
FACS = [24]
# Sample length = warmup(0) + hindcast(30) + forecast(72) = 102 hours
# Daily: 102 / 24 = 4.25 → use 5 days to ensure coverage
SEQ_LENS = [5, 102]  # Low-freq 5 days, High-freq 102 hours


# 动态特征配置
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

# 静态特征配置
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

"""var_c = [
    "area",
    "pet_mm_syr",
    "aet_mm_syr",
    "ele_mt_sav",
]"""
feature_buckets_sta = [0] * len(var_c)  # Static features to Low-freq (daily) branch

feature_buckets = feature_buckets_dyn + feature_buckets_sta

# 特征聚合方式
agg_map_dyn = {
    "precipitation_obs": "sum",  # 降水用 sum
    "temperature_2m": "mean",  # 温度用 mean
    "surface_net_solar_radiation_hourly": "mean",
    "snowfall_hourly": "sum",  # 雪量通常也用 sum
}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

scaler = "DapengScaler"
seeds = 111


def get_config(hyperparams):
    """生成配置（固定隐藏层，仅代入其他超参数）"""
    lr, batch_size, epoch, lr_factor = hyperparams  # 解析当前组合（无隐藏层）
    hidden_str = "_".join(map(str, FIXED_HIDDEN_SIZES))  # 固定隐藏层字符串

    # 实验名称（包含所有调整的超参数）
    exp_name = (
        f"mtslstm_{scaler}_hidden_{hidden_str}_"
        f"lr_{lr:.0e}_batch_{batch_size}_epoch_{epoch}_lrf_{lr_factor}"
    )
    project_sub = (
        f"newbasin_223_FloodEvents_form_106/{exp_name}"  # 两级目录结构
    )

    cfg = default_config_file()
    args = cmd(
        stat_dict_file="pth_file/dapengscaler_stat.json",
        sub=project_sub,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": sanxia_dir,
            "other_settings": {
                "time_unit": ["1h"],
                "dataset_name": "223_FloodEvents_106",
                "delta_t_hours": 1.0,
            },
        },
        ctx=[0],
        model_name="MTSLSTM",
        model_hyperparam={
            # "input_size": 39,
            # "hidden_size": 128,
            "pretrained_day_path": PRETRAIN_DAY_PTH,
            "pretrained_lstm_prefix": "lstm.",
            "pretrained_head_prefix": "linearOut.",
            "hidden_sizes": FIXED_HIDDEN_SIZES,  # 固定隐藏层x
            "output_size": 1,
            "shared_mtslstm": False,
            "transfer": "linear",
            "dropout": 0.1,
            "return_all": False,
            "feature_buckets": feature_buckets,
            "per_feature_aggs_map": per_feature_aggs_map,
            "frequency_factors": FACS,
            "seq_lengths": SEQ_LENS,
            "slice_transfer": True,
            "slice_use_ceil": True,
        },
        # 训练参数（动态代入超参数）
        gage_id=gage_id,
        rolling=0,
        hindcast_length=30,  # hindcast period: 30 hours
        forecast_length=72,  # forecast period: 72 hours (matches original SEQ_LENS[-1]=72)
        warmup_length=720,
        batch_size=batch_size,
        rs=seeds,
        ensemble=True,
        ensemble_items={"seeds": seeds},
        min_time_unit="h",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow", "flood_event"],
        scaler=scaler,
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [],
            "pbm_norm": False,
        },
        train_epoch=epoch,
        save_epoch=1,
        train_period=["2014-01-01", "2023-07-31"],
        valid_period=["2023-07-02", "2024-12-31"],
        test_period=["2023-07-02", "2024-12-31"],
        loss_func="FloodLoss",
        opt="Adam",
        opt_param={"lr": lr},
        lr_scheduler={"lr_factor": lr_factor},
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        patience=2,
        model_type="Normal",
        model_loader={"load_way": "best"},
        dataset="FloodEventDataset",
        evaluator={"eval_way": "floodevent"},
        peak_aligned_sampling={
            "enabled": False,  # Enable peak-aligned sampling
            "slide_step": 1,  # Sliding window step size
            "peak_selection": "first",  # "first" or "highest"
        },
    )

    update_cfg(cfg, args)
    # ---------------------------
    # 兼容性修复：确保 cfg 中包含 FloodEventDataset / trainer 期望的 evaluation 配置项
    # 放在 `update_cfg(cfg, args)` 之后，`return cfg` 之前
    # ---------------------------

    # 1) 确保老代码期望的 cfg["evaluation"] 存在，并填入缺省字段，防止 KeyError
    if "evaluation" not in cfg or not isinstance(cfg["evaluation"], dict):
        cfg["evaluation"] = {}
    # 默认值（与 FloodEventDataset 中被访问的字段对齐）
    cfg["evaluation"].setdefault("hrwin", None)  # hour window
    cfg["evaluation"].setdefault("frwin", None)  # flood window
    cfg["evaluation"].setdefault("mode", "normal")

    # 2) 兼容我们此前使用/检查过的 evaluation_cfgs 结构（trainer / collate 查找时可能用到）
    if "evaluation_cfgs" not in cfg or not isinstance(cfg["evaluation_cfgs"], dict):
        cfg["evaluation_cfgs"] = {}
    # ensure top-level key "evaluator" exists and has eval_way
    cfg["evaluation_cfgs"].setdefault("evaluator", {})
    cfg["evaluation_cfgs"]["evaluator"].setdefault("eval_way", "floodevent")
    # also set a recover_mode used previously
    cfg["evaluation_cfgs"]["evaluator"].setdefault("recover_mode", "bybasins")
    cfg["evaluation_cfgs"]["evaluator"].setdefault("stride", 1)

    # 3) 兼容某些代码可能直接在 training_cfgs 里寻找 evaluation_cfgs（冗余写入，不影响其它逻辑）
    # 如果你后续的 trainer 从 training_cfgs 里读取，下面这步会把同一份结构复制一份到 args-like dict（如果存在）
    try:
        # args (来自 cmd) 通常是 args variable; 不强制存在，防错
        if "args" in locals() and isinstance(args, dict):
            args.setdefault("evaluation_cfgs", {})
            args["evaluation_cfgs"].setdefault("evaluator", {})
            args["evaluation_cfgs"]["evaluator"].setdefault(
                "eval_way", cfg["evaluation_cfgs"]["evaluator"]["eval_way"]
            )
    except Exception:
        # 安全兜底，不影响流程
        pass

    return cfg


if __name__ == "__main__":
    total_exp = len(hyperparam_combinations)
    print(f"⚠️  固定隐藏层结构：{FIXED_HIDDEN_SIZES}（{len(FIXED_HIDDEN_SIZES)}层）")
    print(f"\n总实验数：{total_exp}（仅调整学习率、批次大小、轮数、衰减因子）\n")

    # 循环执行所有实验
    for exp_idx, hyperparams in enumerate(hyperparam_combinations, 1):
        lr, batch_size, epoch, lr_factor = hyperparams
        print(f"\n{'='*60}")
        print(f"实验 {exp_idx}/{total_exp}")
        print(f"超参数：")
        print(f"  学习率：{lr:.0e}")
        print(f"  批次大小：{batch_size}")
        print(f"  训练轮数：{epoch}")
        print(f"  学习率衰减因子：{lr_factor}")
        print(f"{'='*60}\n")

        cfg = get_config(hyperparams)
        train_and_evaluate(cfg)

    print("\n所有实验完成！")
