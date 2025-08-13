import logging
import os.path
import pandas as pd

import sys
from pathlib import Path
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.camels import Camels

# Get the project directory of the py file

# import the module using a relative path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

hru_delete = "01"

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

camels_dir = os.path.join("D:\stream_data", "camels", "camels_us")
camels = Camels(camels_dir)
gage_id = ['01031500', '01030500', '01047000', '01052500', '04057800', '04043050', '04040500', '01055000', '01137500', '01134500', '04015330', '04045500', '04127918', '01139000', '04056500', '05129115', '01022500', '04027000', '01078000', '04296000', '01054200', '01057000', '04256000', '04057510', '05131500', '04024430', '04059500', '05393500', '04074950', '04063700', '01139800', '01162500', '01144000', '01142500', '01169000', '01170100', '05362000', '04127997', '01181000', '01187300', '04124000', '01350080', '01350140', '01350000', '01073000', '01510000', '04122500', '01435000', '01414500', '01333000', '04224775', '01552000', '01413500', '03011800', '03028000', '03015500', '01434025', '01548500', '01545600', '04122200', '04216418', '03010655', '01415000', '01549500', '04233000', '04221000', '01423000', '03021350', '03026500', '01550000', '01552500', '01543000', '01532000']

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
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        f"camels", f"simplelstm_{scaler}_{dim}_{dr}_ens_{hru_delete}"
    )

    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[1],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 71,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        batch_size=384,
        rs=seeds,
        ensemble=ens,
        ensemble_items={"seeds": seeds},
        forecast_history=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval=1,
        var_t=["prcp", "dayl", "srad", "tmax", "tmin", "vp", "PET"],
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },
        var_c=[
            "p_mean",
            "pet_mean",
            "p_seasonality",
            "frac_snow",
            "aridity",
            "high_prec_freq",
            "high_prec_dur",
            "low_prec_freq",
            "low_prec_dur",
            "elev_mean",
            "slope_mean",
            "area_gages2",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "gvf_max",
            "gvf_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "soil_depth_pelletier",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "sand_frac",
            "silt_frac",
            "clay_frac",
            "geol_1st_class",
            "glim_1st_class_frac",
            "geol_2nd_class",
            "glim_2nd_class_frac",
            "carbonate_rocks_frac",
            "geol_porostiy",
            "geol_permeability",
        ],
        scaler=scaler,
        var_out=["streamflow"],
        dataset="AlphaEarthDataset",
        train_epoch=20,
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
        model_type="Normal"
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
