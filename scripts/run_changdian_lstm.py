"""
Author: Wenyu Ouyang
Date: 2022-09-09 14:47:42
LastEditTime: 2024-11-11 18:33:10
LastEditors: Wenyu Ouyang
Description: a script to run experiments for LSTM - CAMELS
FilePath: \torchhydro\experiments\run_camelslstm_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

VAR_C_CHOSEN_FROM_CHANGDIAN_CHAI = [
    "sgr_dk_sav",  # 河流坡度
    "p_mean",  # 年平均降雨
    "lit_cl_smj",  # 土壤地质
    "wet_cl_smj",  # 湿地
    "glc_cl_smj",  # 土地利用
    "area",  # shp area calculated by the shapefile
    "pet_mm_syr",  # Potential Evapotranspiration: Category = Climate; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {yr} annual average
    "aet_mm_syr",  # Actual Evapotranspiration: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
    "ele_mt_sav",  # Elevation: Category = Physiography; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "lka_pc_sse",  # Limnicity (Percent Lake Area): Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "for_pc_sse",  # Forest Cover Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "slp_dg_sav",  # Terrain Slope: Category = Physiography; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "lkv_mc_usu",  # Lake Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
    "ero_kh_sav",  # Soil Erosion: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "tmp_dc_syr",  # Air Temperature: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
    "riv_tc_usu",  # River Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
    "gdp_ud_sav",  # Gross Domestic Product: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "soc_th_sav",  # Organic Carbon Content in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "kar_pc_sse",  # Karst Area Extent: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "ppd_pk_sav",  # Population Density: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "nli_ix_sav",  # Nighttime Lights: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "pst_pc_sse",  # Pasture Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "pac_pc_sse",  # Protected Area Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "snd_pc_sav",  # Sand Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "pop_ct_usu",  # Population Count: Category = Anthropogenic; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
    "swc_pc_syr",  # Soil Water Content: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
    "ria_ha_usu",  # River Area: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
    "slt_pc_sav",  # Silt Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "cly_pc_sav",  # Clay Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "crp_pc_sse",  # Cropland Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "inu_pc_slt",  # Inundation Extent: Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {lt} long-term maximum
    "cmi_ix_syr",  # Climate Moisture Index: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
    "snw_pc_syr",  # Snow Cover Extent: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
    "ari_ix_sav",  # Global Aridity Index: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
    "ire_pc_sse",  # Irrigated Area Extent (Equipped): Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "rev_mc_usu",  # Reservoir Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
    "inu_pc_smn",  # Inundation Extent: Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {mn} annual minimum
    "urb_pc_sse",  # Urban Extent: Category = Anthropogenic; Spatial Extent = {s} in reach catchment; Dimensions = {se} spatial extent (%)
    "prm_pc_sse",  # Permafrost Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
    "gla_pc_sse",  # Glacier Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
]
VAR_T_CHOSEN_FROM_CHANGDIAN_CHAI = [
    # NOTE: prcp must be the first variable
    "total_precipitation_hourly",
    "potential_evaporation_hourly",
    "snowfall_hourly",
    "snowmelt_hourly",
    "snow_depth_water_equivalent",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_net_solar_radiation_hourly",
    # "surface_net_solar_radiation",
]
DATASET_DIR = "/home/mahaoran/data/"
data_path = os.path.join(DATASET_DIR, "changdian_chai")
# dataset = SelfMadeHydroDataset(data_path=data_path, time_unit=["1D"])


def run_normal_dl(
    project_name,
    gage_id_file,
    var_c=VAR_C_CHOSEN_FROM_CHANGDIAN_CHAI,
    var_t=VAR_T_CHOSEN_FROM_CHANGDIAN_CHAI,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    if train_period is None:
        train_period = ["2021-1-01", "2024-12-31"]
        # train_period = ["2020-1-01", "2024-12-31"]
    if valid_period is None:
        valid_period = ["2021-1-01", "2024-12-31"]
    if test_period is None:
        test_period = ["2021-1-01", "2024-12-31"]
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        # source_cfgs={
        #     "source_name": "camels_us",
        #     "source_path": os.path.join(
        #         # SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
        #         SETTING["local_data_path"]["datasets-origin"]
        #     ),
        # },
        # source_cfgs={
        #     "source_name": "selfmadehydrodataset",
        #     "source_path": data_path,
        #     "other_settings": {
        #         "time_unit": ["1D"],
        #         "dataset_name": "era5chinaandcamels",
        #     },
        # },
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["1D"],
            },
        },
        ctx=[2],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": len(var_c) + len(var_t),
            "output_size": 1,
            "hidden_size": 512,
        },
        loss_func="RMSESum",
        # loss_func="MultiOutLoss",
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0, 0],
        #     "device": [2],  # =ctx
        #     "item_weight": [1, 0],
        #     "limit_part": [1],
        # },
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        # scaler_params={
        #     "prcp_norm_cols": [
        #         "streamflow",
        #     ],
        #     "gamma_norm_cols": [
        #         "total_precipitation_hourly",
        #     ],
        #     "pbm_norm": False,
        # },
        batch_size=64,
        hindcast_length=0,
        forecast_length=365,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adam",
        rs=1234,
        train_epoch=30,
        save_epoch=1,
        opt_param= {"lr": 0.0007},
        lr_scheduler={
            "lr": 0.0007,
            "lr_factor": 0.98,
        },
        model_loader={
            "load_way": "specified",
            "test_epoch": 20,
        },
        # gage_id = ["sanxia_60406350"], # 彝良流量
        # gage_id = ["sanxia_60406500"], # 箐口塘流量
        # gage_id = ["sanxia_60406700"], # 豆沙关水文站流量
        # gage_id = ["sanxia_60407100"], # 横江流量
        # gage_id = ["sanxia_60407200"], # 牛街水文站流量
        # gage_id = ["sanxia_12020000"], # 三皇庙
        # gage_id = ["changdian_61700"], # 三皇庙
        gage_id_file=gage_id_file,
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# the gage_id.txt file is set by the user, it must be the format like:
# GAUGE_ID
# 01013500
# 01022500
# ......
# Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
run_normal_dl(os.path.join("changdian230", "obsrain_85basin_area2w_lstm_newdata"), "/home/mahaoran/data/gage_id/gage_id_changdian85_area2w.csv")
