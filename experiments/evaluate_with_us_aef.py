"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2025-01-10 10:11:30
LastEditors: Wenyu Ouyang
Description: Train a model for 3775 basins
FilePath: /HydroForecastEval/scripts/train_googlefloodhub_camels_671basins_ear5land_less_param_new_rolling_large_horizon.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import argparse
import logging
import os.path
import pandas as pd

import sys
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))


def _parse_cli():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gage-id-file", default=None)
    p.add_argument("--gage-ids", default=None)
    p.add_argument("--train-hru-delete", default=None)
    p.add_argument("--eval-hru-delete", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--ctx", default=None)
    p.add_argument("--project-dir", default=None)
    p.add_argument("--training-project-dir", default=None)
    p.add_argument("--camels-dir", default=None)
    p.add_argument("--train-epoch", type=int, default=None)
    args, _ = p.parse_known_args()
    return args


CLI = _parse_cli()

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from path_config import get_paths
    _PATHS = get_paths()
except Exception:
    _PATHS = {}


train_hru_delete = CLI.train_hru_delete or "01"
eval_hru_delete = CLI.eval_hru_delete or train_hru_delete
hru_delete = eval_hru_delete

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = CLI.camels_dir or _PATHS.get("camels_dir") or os.path.join("/Volumes/Untitled/data")
camels = Camels(camels_dir)

if CLI.gage_id_file:
    with open(CLI.gage_id_file, "r", encoding="utf-8") as _f:
        gage_id = [ln.strip() for ln in _f if ln.strip()]
elif CLI.gage_ids:
    gage_id = [s.strip() for s in CLI.gage_ids.split(",") if s.strip()]
else:
    gage_id = ["01013500", "01022500", "01030500", "01031500", "01047000"]
gage_id = sorted([x for x in gage_id])
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

_train_results_root = (CLI.training_project_dir
                       or _PATHS.get("project_dir")
                       or "/Users/cylenlc/work/torchhydro")
_eval_project_dir = (CLI.project_dir
                     or _PATHS.get("eval_project_dir")
                     or r"/Users/cylenlc/work/torchhydro/experiments")

length = 7
dim = 128
scaler = "DapengScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels_test", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        train_mode=False,
        stat_dict_file=f"{_train_results_root}/results/camels/simplelstm_{scaler}_{dim}_{dr}_ens_{train_hru_delete}/dapengscaler_stat.json",
        project_dir=_eval_project_dir,
        sub=project_name,
        source_cfgs={"dataset_id": "camels_us"},
        ctx=[int(x) for x in CLI.ctx.split(",")] if CLI.ctx else [0],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 71,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={
            "load_way": "pth",
            "pth_path": f"{_train_results_root}/results/camels/simplelstm_{scaler}_{dim}_{dr}_ens_{train_hru_delete}/best_model.pth",
        },
        gage_id=gage_id,
        batch_size=256,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval=1,
        var_t=[
            "precipitation",
            "daylight_duration",
            "solar_radiation",
            "temperature_max",
            "temperature_min",
            "vapor_pressure",
            "potential_evapotranspiration",
        ],
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": ["precipitation", "potential_evapotranspiration"],
            # "one_hot_cols": ["dom_land_cover", "geol_1st_class", "geol_2nd_class"],
            "pbm_norm": False,
        },
        var_c=[
            "elev_mean",
            "slope_mean",
            "area",
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
        var_out=["streamflow"],
        dataset=CLI.dataset or "AlphaEarthDataset",
        train_epoch=CLI.train_epoch or 10,
        save_epoch=1,
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
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
        rolling=1,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=2,
        model_type="Normal",
        valid_batch_mode="train",
        # valid_batch_mode="test",
        evaluator={
            # "eval_way": "once",
            #  "stride": 0,
            "eval_way": "1pace",
            # "pace_idx": -1,
            "pace_idx": -1,
        },
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


if __name__ == "__main__":
    configs = config()
    train_and_evaluate(configs)
