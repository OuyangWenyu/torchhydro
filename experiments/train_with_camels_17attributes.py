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
sys.path.append(r"C:\Users\Pengfei Qu\Desktop\torchhydro")
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

camels_dir = os.path.join("D:\stream_data", "camels", "camels_us")
camels = Camels(camels_dir)
# gage_id = camels.read_site_info()["gauge_id"].values.tolist()
gage_id = ['14301000', '01030500', '07056000', '01547700', '12013500', '01557500', '03237280', '03173000', '03069500', '03070500', '14306500', '01187300', '04040500', '01567500', '03182500', '03237500', '07068000', '04216418', '03213700', '03488000', '01440000', '12025000', '12035000', '03180500', '03281500', '03604000', '07196900', '01606500', '05129115', '01439500', '04024430', '03300400', '04059500', '03066000', '14305500', '02038850', '02196000', '04233000', '05131500', '14303200', '03281100', '01031500', '05408000', '14306340', '03473000', '02371500', '02051500', '02046000', '08013000', '07359610', '02472000', '02070000', '14400000', '03471500', '11468500', '01664000', '02051000', '04027000', '02016000', '02372250', '06919500', '02125000', '08014500', '07071500', '01634500', '02018000', '01548500', '02082950', '02027000', '02193340', '03164000', '11481200', '03498500', '02014000', '01543500', '02069700', '01632000', '03384450', '14309500', '14325000', '14154500', '02013000', '01047000', '02017500', '08025500', '03021350', '03280700', '02212600', '01451800', '01552000', '11162500', '04213075', '03159540', '04213000', '02408540', '02011400', '02450250', '05489000', '01518862', '08066200', '03015500', '02464000', '01350080', '01543000', '14166500', '03144000', '02140991', '04015330', '05362000', '01440400', '02112120', '02464146', '02464360', '01052500', '02415000', '01550000', '01542810', '01181000', '04221000', '03010655', '03050000', '01544500', '01350000', '02056900', '01435000', '01350140', '01516500', '05488200', '02011460', '01568000', '01510000', '02065500', '02469800', '02028500', '02177000', '01413500', '01552500', '07362100', '01423000', '05495000', '07340300', '02427250', '01365000', '07292500', '03187500', '02422500', '07067000', '07362587', '07373000', '03340800', '02015700', '05487980', '05466500', '12056500', '01415000', '13338500', '07376000', '07263295', '01434025', '02374500', '01170100', '01134500', '01667500', '07335700', '02342933', '01414500', '03186500', '12043000', '01144000', '01605500', '03439000', '03028000', '11480390', '08023080', '04045500', '03140000', '03238500', '03026500', '06452000', '05503800', '04074950', '06934000', '04057800', '02178400', '12040500', '07295000', '03346000', '02064000', '02327100', '07346045', '07014500', '02096846', '11478500', '01580000', '07291000', '04063700', '03338780', '01073000', '01539000', '01549500', '05393500', '02315500', '03479000', '03500000', '01333000', '01055000', '06911900', '06888500', '02395120', '12411000', '03011800', '01545600', '03357350', '12145500', '01594950', '03049000', '04197100', '01532000', '07167500', '04224775', '06360500', '12054000', '02128000', '04122200', '01596500', '01583500', '01591400', '02081500', '02324400', '01666500', '07375000', '01620500', '04122500', '11284400', '06447000', '13340000', '02092500', '06441500', '07060710', '02363000', '01586610', '11532500', '04105700', '07290650', '02384540', '01658500', '03076600', '04056500', '01613050', '02430615', '11528700', '02053800', '01169000', '03285000', '02108000', '03078000', '06354000', '05057000', '01139000', '03165000', '02314500', '02479560', '06889500', '02231000', '11476600', '06878000', '01057000', '02465493', '04185000', '08086212', '06601000', '04127918', '03049800', '03574500', '02479300', '11475560', '02472500', '02059500', '12073500', '06910800', '01644000', '01669520', '12414500', '07066000', '06876700', '02481510', '11451100', '04057510', '02053200', '14362250', '04124000', '11522500', '06903400', '12041200', '08029500', '02481000', '01411300', '08070200', '06889200', '14158790', '04127997', '07180500', '06892000', '05585000', '07197000', '08066300', '01486000', '06784000', '05123400', '02361000', '03170000', '05412500', '08070000', '07301500', '02137727', '11124500', '11180500', '01142500', '11141280', '05591550', '12388400', '05056000', '02027500', '12010000', '05592050', '06885500', '01669000', '05592575', '02369800', '02430085', '12167000', '02143040', '06853800', '05062500', '11482500', '12048000', '11148900', '06906800', '02310947', '02152100', '11180960', '04296000', '14316700', '02074500', '06803530', '12390700', '07315700', '03592718', '01118300', '07315200', '14185900', '12115000', '06353000', '06453600', '09484600', '02350900', '04256000', '02149000', '12114500', '03366500', '08164300', '07261000', '11473900', '07057500', '05556500', '07184000', '12147500', '08164000', '06404000', '02077200', '08086290', '07299670', '06847900', '08079600', '04161580', '08104900', '01162500', '02479155', '01195100', '12358500', '07145700', '09494000', '02111500', '08101000', '08175000', '07149000', '01121000', '06879650', '06477500', '02055100', '12375900', '13340600', '05584500', '02221525', '02143000', '02118500', '02111180', '01022500', '07148400', '07226500']
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

    # project_name = os.path.join("train_googleflood", "exp1_lstm_googlefloodwochina")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        #project_dir="D:\\torchhydro\\text2attr",
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": camels_dir,
            "other_settings": {"download": False, "region": "US"},
        },
        ctx=[1],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 24,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        model_loader={"load_way": "best"},
        # gage_id=gage_id[5000:5009],
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
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
                # "streamflow_input",
                "streamflow",
            ],
            "gamma_norm_cols": ["prcp", "PET"],
            "pbm_norm": False,
        },
        var_c=[
            "elev_mean",
            "slope_mean",
            "area_gages2",
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
        dataset="StreamflowDataset",
        train_epoch=20,
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
train_and_evaluate(configs)
