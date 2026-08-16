"""
Standalone script to run SHAP analysis on a previously trained model.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import pandas as pd
from torchhydro.configs.config import get_config_file
from torchhydro.trainers.deep_hydro import DeepHydro

def main():
    # 1. 指定已训练模型的输出目录 (包含 config.json 和 .pth 权重文件)
    case_dir = r"results/newbasin_223_FloodEvents_for_shap/mtslstm_DapengScaler_hidden_128_128_lr_1e-03_batch_128_epoch_20_lrf_0.95"
    basin_ids = None
    metric_path = os.path.join(case_dir, "metric_streamflow.csv")
    if basin_ids is None and os.path.exists(metric_path):
        try:
            metric_df = pd.read_csv(metric_path)
            if "basin_id" in metric_df.columns:
                basin_ids = metric_df["basin_id"].dropna().astype(str).tolist()
        except Exception:
            basin_ids = None

    if not os.path.exists(case_dir):
        print(f"Error: Directory {case_dir} does not exist. Please train the model first.")
        return

    # 2. 从目录中加载已有的实验配置
    print(f"Loading configuration from {case_dir}...")
    cfg = get_config_file(case_dir)

    # 3. 修改配置：关闭训练模式，仅加载测试集进行推理和 SHAP 分析
    cfg["training_cfgs"]["train_mode"] = False
    
    # 确保加载最佳模型进行 SHAP 分析
    if "evaluation_cfgs" not in cfg:
        cfg["evaluation_cfgs"] = {}
    cfg["evaluation_cfgs"]["model_loader"] = {"load_way": "best"}

    # 注入 SHAP 分析的配置
    training_cfgs = cfg.get("training_cfgs", {})
    seq_len = None
    if (
        isinstance(training_cfgs.get("hindcast_length"), int)
        and isinstance(training_cfgs.get("forecast_length"), int)
    ):
        seq_len = training_cfgs["hindcast_length"] + training_cfgs["forecast_length"]
    base_shap_cfgs = {
        "enable": True,
        "max_sample": 1000,
        "sequence_length": seq_len,
        "method": "gradient",
        "nsamples": 50,
        "background_size": 100,
        "diagnose": True,
        "diagnose_modes": ["zero", "permute_samples"],
        "diagnose_max_samples": 128,
        "diagnose_topk": 15,
        "sequence_enable": True,
        "sequence_max_sample": 64,
        "sequence_background_size": 24,
        "sequence_nsamples": 50,
        "time_agg": "abs_sum",
        "plot_mode": "split_only",
        "clean_extra": True,
        "hybrid_dynamic_mode": "seq_bucket0",
    }

    if not basin_ids:
        shap_save_dir = os.path.join(case_dir, "shap_results_standalone")
        cfg["evaluation_cfgs"]["shap_cfgs"] = {**base_shap_cfgs, "save_dir": shap_save_dir}
        print("Initializing DeepHydro for SHAP analysis (all basins)...")
        trainer = DeepHydro(cfg)
        print("Running evaluation and SHAP analysis...")
        trainer.model_evaluate()
        print("SHAP analysis finished successfully!")
        print(f"Please check the plots at: {shap_save_dir}")
        return

    topk_env = os.getenv("SHAP_TOPK")
    if isinstance(topk_env, str) and topk_env.isdigit():
        k = int(topk_env)
        if k > 0:
            basin_ids = basin_ids[:k]

    print(f"Found {len(basin_ids)} basins in metric_streamflow.csv (may be limited), running per-basin SHAP...")
    for basin_id in basin_ids:
        cfg_b = copy.deepcopy(cfg)
        cfg_b["data_cfgs"]["object_ids"] = [str(basin_id)]
        shap_save_dir = os.path.join(case_dir, "shap_results_by_basin", str(basin_id))
        cfg_b["evaluation_cfgs"]["shap_cfgs"] = {**base_shap_cfgs, "save_dir": shap_save_dir}
        print(f"Initializing DeepHydro for SHAP analysis (basin={basin_id})...")
        trainer = DeepHydro(cfg_b)
        print(f"Running evaluation and SHAP analysis (basin={basin_id})...")
        trainer.model_evaluate()
        print(f"[OK] basin={basin_id} -> {shap_save_dir}")

if __name__ == "__main__":
    main()
