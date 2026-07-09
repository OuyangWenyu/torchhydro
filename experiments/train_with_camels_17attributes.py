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
import sys
from pathlib import Path
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))


def _parse_cli():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gage-id-file", default=None,
                   help="text file with one basin id per line (overrides hardcoded list)")
    p.add_argument("--gage-ids", default=None,
                   help="comma-separated basin ids (overrides hardcoded list)")
    p.add_argument("--hru-delete", default=None,
                   help="suffix used in the experiment sub-folder; default '01'")
    p.add_argument("--dataset", default=None,
                   help="dataset class name; default 'StreamflowDataset'")
    p.add_argument("--ctx", default=None,
                   help="comma-separated GPU device ids; default '0'")
    p.add_argument("--project-dir", default=None,
                   help="project dir for results/; default from paths.yaml")
    p.add_argument("--camels-dir", default=None,
                   help="CAMELS data root; default from paths.yaml")
    p.add_argument("--data-source", choices=["cloud", "local"], default="cloud",
                   help="read data from cloud zarr or local CAMELS files")
    p.add_argument("--train-epoch", type=int, default=None,
                   help="override training epochs (debug)")
    args, _ = p.parse_known_args()
    return args


CLI = _parse_cli()

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from path_config import get_paths
    _PATHS = get_paths()
except Exception:
    _PATHS = {}


hru_delete = CLI.hru_delete or "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = CLI.camels_dir or _PATHS.get("camels_dir")
if CLI.data_source == "local":
    from hydrodataset.camels import Camels

    camels_dir = camels_dir or os.path.join("/Volumes/Untitled/data")
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

length = 7
dim = 128
scaler = "DapengScaler"
# scaler = "StandardScaler"
dr = 0.4
seeds = 111
ens = True


def config():
    project_name = os.path.join(
        f"camels", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )
    config_data = default_config_file()

    args = cmd(
        project_dir=CLI.project_dir or _PATHS.get("project_dir"),
        sub=project_name,
        source_cfgs={
            "dataset_id": "camels_us",
            "other_settings": {
                "source": CLI.data_source,
                "download": False,
                "region": "US",
            },
        },
        ctx=[int(x) for x in CLI.ctx.split(",")] if CLI.ctx else [0],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 24,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
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
        dataset=CLI.dataset or "StreamflowDataset",
        train_epoch=CLI.train_epoch or 20,
        save_epoch=1,
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        early_stopping=True,
        rolling=0,
        patience=2,
        model_type="Normal",
    )

    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
