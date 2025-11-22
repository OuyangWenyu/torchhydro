"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2025-01-10 10:11:30
LastEditors: Wenyu Ouyang
Description: Train a model for 3775 basins
FilePath: /HydroForecastEval/scripts/train_googlefloodhub_camels_671basins_ear5land_less_param_new_rolling_large_horizon.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

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

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("/Volumes/Untitled/data")
camels = Camels(camels_dir)
# gage_id = camels.read_site_info()["gauge_id"].values.tolist()
gage_id = [
    "01013500",
    "01031500",
    "01052500",
    "01055000",
    "01078000",
    "01121000",
    "01162500",
    "01365000",
    "01439500",
    "01516500",
    "01543500",
    "01544500",
    "01550000",
    "01568000",
    "01583500",
    "01591400",
    "01644000",
    "01666500",
    "01669520",
    "02011400",
    "02016000",
    "02017500",
    "02028500",
    "02046000",
    "02051000",
    "02051500",
    "02056900",
    "02059500",
    "02065500",
    "02082950",
    "02111180",
    "02143000",
    "02149000",
    "02152100",
    "02216180",
    "02245500",
    "02349900",
    "02369800",
    "02372250",
    "02395120",
    "02430085",
    "02465493",
    "02472500",
    "02481510",
    "03028000",
    "03066000",
    "03070500",
    "03076600",
    "03186500",
    "03291780",
    "03338780",
    "03340800",
    "03455500",
    "03500240",
    "03574500",
    "04015330",
    "04074950",
    "04161580",
    "04213075",
    "05056000",
    "05057000",
    "05413500",
    "05414000",
    "05488200",
    "05495500",
    "05501000",
    "05508805",
    "05593575",
    "06037500",
    "06224000",
    "06289000",
    "06339100",
    "06441500",
    "06450500",
    "06468170",
    "06746095",
    "06784000",
    "06803510",
    "06876700",
    "06903400",
    "06921070",
    "07149000",
    "07208500",
    "07315200",
    "07315700",
    "07362100",
    "07375000",
    "08014500",
    "08050800",
    "08070200",
    "08104900",
    "08164300",
    "08171300",
    "08175000",
    "08178880",
    "08198500",
    "08200000",
    "08271000",
    "08377900",
    "08380500",
    "09047700",
    "09107000",
    "09312600",
    "09430500",
    "09484000",
    "09497980",
    "10234500",
    "10244950",
    "10348850",
    "10396000",
    "11162500",
    "11180960",
    "11253310",
    "11381500",
    "11473900",
    "11478500",
    "11522500",
    "11523200",
    "12025000",
    "12054000",
    "12092000",
    "12114500",
    "12141300",
    "12167000",
    "12374250",
    "12377150",
    "12411000",
    "13018300",
    "13023000",
    "13313000",
    "14138900",
    "14154500",
    "14166500",
    "14185900",
    "14362250",
]

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
        f"camels_test", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )
    config_data = default_config_file()

    args = cmd(
        train_mode=False,
        stat_dict_file=r"/Users/cylenlc/work/torchhydro/experiments/results/camels/simplelstm_DapengScaler_128_0.4_ens_01/dapengscaler_stat.json",
        project_dir=r"/Users/cylenlc/work/torchhydro/experiments",
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[0],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 88,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={
            "load_way": "pth",
            "pth_path": f"/Users/cylenlc/work/torchhydro/experiments/results/camels/simplelstm_DapengScaler_128_0.4_ens_{hru_delete}/best_model.pth",
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
            "pbm_norm": False,
            # "one_hot_cols": ["dom_land_cover", "geol_1st_class", "geol_2nd_class"],
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
        # scaler="DapengScaler",
        scaler=scaler,
        var_out=["streamflow"],
        dataset="MergedDataset",
        train_epoch=10,
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
        shap_cfgs={
            "enable": True,
            "save_dir": "/Users/cylenlc/追加实验/CamelsAttr+AEF/第一折",
            "max_sample": 3000,
            "background_size": 100,
        },
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


if __name__ == "__main__":
    configs = config()
    train_and_evaluate(configs)
