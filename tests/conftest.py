import os
import pytest

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
import logging
import pandas as pd


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure the logger"""
    logging.basicConfig(level=logging.INFO)
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)


@pytest.fixture(scope="session")
def basin4test():
    """Read the basin ID list, only choose final 5 basins as test data"""
    show = pd.read_csv("data/basin_id(46+1).csv", dtype={"id": str})
    return show["id"].values.tolist()[-5:]


@pytest.fixture()
def config_data():
    return default_config_file()


@pytest.fixture()
def args():
    project_name = os.path.join("test_camels", "exp1")
    data_dir = SETTING["local_data_path"]["datasets-origin"]
    source_path = os.path.join(data_dir, "camels", "camels_us")
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": source_path,
        },
        ctx=[-1],
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        batch_size=8,
        hindcast_length=0,
        forecast_length=20,
        min_time_unit="D",
        min_time_interval="1",
        var_t=["prcp", "dayl", "srad", "tmax", "tmin", "vp"],
        # var_c=["None"],
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        model_loader={"load_way": "specified", "test_epoch": 2},
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        # key is epoch, start from 0, each value means the decay rate
        lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        which_first_tensor="sequence",
    )


@pytest.fixture()
def mtl_args():
    project_name = os.path.join("test_camels", "expmtl001")
    data_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    data_interim_dir = SETTING["local_data_path"]["datasets-interim"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_names": [
                "usgs4camels",
                "modiset4camels",
                "nldas4camels",
                "smap4camels",
            ],
            "source_paths": [
                os.path.join(data_origin_dir, "camels", "camels_us"),
                os.path.join(data_interim_dir, "camels_us", "modiset4camels"),
                os.path.join(data_interim_dir, "camels_us", "nldas4camels"),
                os.path.join(data_interim_dir, "camels_us", "smap4camels"),
            ],
        },
        ctx=[0],
        model_type="MTL",
        model_name="KuaiLSTMMultiOut",
        model_hyperparam={
            "n_input_features": 23,
            "n_output_features": 2,
            "n_hidden_states": 256,
            "layer_hidden_size": 128,
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": [0],
            "item_weight": [0.5, 0.5],
        },
        # gage_id_file=os.path.join("results", "test_camels", "camels_us_mtl_2001_2021_flow_screen.csv"),
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        batch_size=100,
        hindcast_length=0,
        forecast_length=365,  # batch_size=100, rho=365,
        var_t=[
            "temperature",
            "specific_humidity",
            "shortwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "total_precipitation",
        ],
        var_t_type=["nldas"],
        var_out=["streamflow", "ET"],
        var_to_source_map={
            "temperature": "nldas4camels",
            "specific_humidity": "nldas4camels",
            "shortwave_radiation": "nldas4camels",
            "potential_energy": "nldas4camels",
            "potential_evaporation": "nldas4camels",
            "total_precipitation": "nldas4camels",
            "streamflow": "usgs4camels",
            "ET": "modiset4camels",
            "elev_mean": "usgs4camels",
            "slope_mean": "usgs4camels",
            "area_gages2": "usgs4camels",
            "frac_forest": "usgs4camels",
            "lai_max": "usgs4camels",
            "lai_diff": "usgs4camels",
            "dom_land_cover_frac": "usgs4camels",
            "dom_land_cover": "usgs4camels",
            "root_depth_50": "usgs4camels",
            "soil_depth_statsgo": "usgs4camels",
            "soil_porosity": "usgs4camels",
            "soil_conductivity": "usgs4camels",
            "max_water_content": "usgs4camels",
            "geol_1st_class": "usgs4camels",
            "geol_2nd_class": "usgs4camels",
            "geol_porostiy": "usgs4camels",
            "geol_permeability": "usgs4camels",
        },
        train_period=["2001-04-01", "2011-04-01"],
        test_period=["2016-04-01", "2017-04-01"],
        dataset="FlexDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=2,
        fill_nan=["no", "mean"],
        model_loader={"load_way": "specified", "test_epoch": 2},
    )


@pytest.fixture()
def s2s_args(basin4test):
    project_name = os.path.join("test_seq2seq", "gpmsmapexp1")
    return cmd(
        sub=project_name,
        # TODO: Update the source_path to the correct path
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
            "other_settings": {"time_unit": ["3h"]},
        },
        ctx=[0],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 17,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            # number of min-time-intervals to predict; horizon
            "forecast_length": 56,
            # Number of preceding streamflow time steps included in the output,
            # which must be less than or equal to hindcast_length, and is recommended to be set to 1
            "hindcast_output_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=basin4test,
        batch_size=512,
        # historical number of min-time-intervals; 240 means 240 * 3H = 720H
        hindcast_length=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval="3",
        var_t=[
            "gpm_tp",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        sampler="BasinBatchSampler",
        scaler="DapengScaler",
        train_epoch=1,
        save_epoch=1,
        train_period=[("2016-06-01-01", "2016-12-31-01")],
        test_period=[("2015-06-01-01", "2015-10-31-01")],
        valid_period=[("2015-06-01-01", "2015-10-31-01")],
        # loss_func="QuantileLoss",
        # loss_param={"quantiles":[0.2,0.8]},
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        calc_metrics=False,
        early_stopping=True,
        patience=8,
        model_type="MTL",
        fill_nan=["no", "no"],
    )


@pytest.fixture()
def trans_args(basin4test):
    project_name = os.path.join("test_trans", "gpmsmapexp1")
    return cmd(
        sub=project_name,
        # TODO: Update the source_path to the correct path
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
            "other_settings": {"time_unit": ["3h"]},
        },
        ctx=[0],
        model_name="Transformer",
        model_hyperparam={
            "n_encoder_inputs": 17,
            "n_decoder_inputs": 16,
            "n_decoder_output": 2,
            "channels": 256,
            "num_embeddings": 512,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.1,
            "hindcast_output_window": 0,
        },
        model_loader={"load_way": "best"},
        gage_id=basin4test,
        batch_size=128,
        hindcast_length=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval="3",
        var_t=[
            "gpm_tp",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="TransformerDataset",
        sampler="BasinBatchSampler",
        scaler="DapengScaler",
        train_epoch=10,
        save_epoch=1,
        train_period=[("2016-06-01-01", "2016-12-31-01")],
        test_period=[("2015-06-01-01", "2015-10-31-01")],
        valid_period=[("2015-06-01-01", "2015-10-31-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="sequence",
        calc_metrics=False,
        early_stopping=True,
        patience=8,
        model_type="MTL",
        fill_nan=["no", "no"],
    )


@pytest.fixture()
def dpl_args():
    project_name = os.path.join("test_camels", "expdpl001")
    data_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    train_period = ["1985-10-01", "1986-04-01"]
    # valid_period = ["1995-10-01", "2000-10-01"]
    valid_period = None
    test_period = ["2000-10-01", "2001-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(data_origin_dir, "camels", "camels_us"),
        },
        ctx=[0],
        model_type="MTL",
        # model_name="DplLstmXaj",
        model_name="DplAttrXaj",
        model_hyperparam={
            # "n_input_features": 25,
            "n_input_features": 17,
            "n_output_features": 15,
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 30,
            "param_limit_func": "clamp",
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "PET",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=50,
        hindcast_length=0,
        forecast_length=60,
        var_t=[
            "prcp",
            "PET",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        # NOTE: The second variable is not necessary, or not used in the model. But to keep the same length with model output, we add a dummy variable.
        var_out=["streamflow", "ET"],
        n_output=2,
        fill_nan=["no", "no"],
        target_as_input=0,
        constant_only=1,
        train_epoch=2,
        model_loader={
            "load_way": "specified",
            "test_epoch": 2,
        },
        warmup_length=30,
        opt="Adadelta",
        which_first_tensor="sequence",
    )


@pytest.fixture()
def seq2seq_config():
    project_name = os.path.join("train_with_gpm", "ex_test")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
            },
        },
        ctx=[0],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 17,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 56,
            "hindcast_output_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=128,
        hindcast_length=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "precipitationCal",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_mode=True,
        train_period=["2016-06-01-01", "2016-08-01-01"],
        test_period=["2015-06-01-01", "2015-08-01-01"],
        valid_period=["2015-06-01-01", "2015-08-01-01"],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="batch",
        rolling=56,
        calc_metrics=False,
        early_stopping=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        model_type="MTL",
    )

    # update the config data
    update_cfg(config_data, args)

    return config_data


@pytest.fixture()
def dpl4hbv_selfmadehydrodataset_args():
    project_name = os.path.join("test", "expdpl4hbv")
    train_period = ["2014-10-01", "2018-10-01"]
    valid_period = ["2017-10-01", "2021-10-01"]
    # valid_period = None
    test_period = ["2017-10-01", "2021-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {"time_unit": ["1D"]},
        },
        model_type="Normal",
        ctx=[0],
        model_name="DplLstmHbv",
        model_hyperparam={
            "n_input_features": 6,
            # "n_input_features": 19,
            "n_output_features": 14,
            "n_hidden_states": 64,
            "kernel_size": 15,
            "warmup_length": 365,
            "param_limit_func": "clamp",
            "param_test_way": "final",
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_hourly",
                "potential_evaporation_hourly",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            # "camels_01013500",
            # "camels_01022500",
            # "camels_01030500",
            # "camels_01031500",
            # "camels_01047000",
            # "camels_01052500",
            # "camels_01054200",
            # "camels_01055000",
            # "camels_01057000",
            # "camels_01170100",
            "changdian_61561"
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=300,
        hindcast_length=0,
        forecast_length=365,
        var_t=[
            # although the name is hourly, it might be daily according to your choice
            "total_precipitation_hourly",
            "potential_evaporation_hourly",
            "temperature_2m",
            "snow_depth_water_equivalent",
            "snowfall_hourly",
            "dewpoint_temperature_2m",
        ],
        var_c=[
            # "sgr_dk_sav",
            # "pet_mm_syr",
            # "slp_dg_sav",
            # "for_pc_sse",
            # "pre_mm_syr",
            # "slt_pc_sav",
            # "swc_pc_syr",
            # "soc_th_sav",
            # "cly_pc_sav",
            # "ari_ix_sav",
            # "snd_pc_sav",
            # "ele_mt_sav",
            # "area",
            # "tmp_dc_syr",
            # "crp_pc_sse",
            # "lit_cl_smj",
            # "wet_cl_smj",
            # "snw_pc_syr",
            # "glc_cl_smj",
        ],
        # NOTE: although we set total_evaporation_hourly as output, it is not used in the training process
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        # train_epoch=100,
        train_epoch=2,
        save_epoch=10,
        model_loader={
            "load_way": "specified",
            # "test_epoch": 100,
            "test_epoch": 2,
        },
        warmup_length=365,
        opt="Adadelta",
        which_first_tensor="sequence",
        # train_mode=0,
        # weight_path="C:\\Users\\wenyu\\code\\torchhydro\\results\\test_camels\\expdpl61561201\\10_September_202402_32PM_model.pth",
        # continue_train=0,
    )


@pytest.fixture()
def dpl4xaj_selfmadehydrodataset_args():
    project_name = os.path.join("test_camels", "expdpl61561201")
    train_period = ["2014-10-01", "2018-10-01"]
    valid_period = ["2017-10-01", "2021-10-01"]
    # valid_period = None
    test_period = ["2017-10-01", "2021-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {"time_unit": ["1D"]},
        },
        model_type="MTL",
        ctx=[1],
        # model_name="DplLstmXaj",
        # model_name="DplAttrXaj",
        model_name="DplNnModuleXaj",
        model_hyperparam={
            "n_input_features": 6,
            # "n_input_features": 19,
            "n_output_features": 15,
            "n_hidden_states": 64,
            "kernel_size": 15,
            "warmup_length": 365,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
            "et_output": 1,
            "param_var_index": [],
        },
        # loss_func="RMSESum",
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_hourly",
                "potential_evaporation_hourly",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            # "camels_01013500",
            # "camels_01022500",
            # "camels_01030500",
            # "camels_01031500",
            # "camels_01047000",
            # "camels_01052500",
            # "camels_01054200",
            # "camels_01055000",
            # "camels_01057000",
            # "camels_01170100",
            "changdian_61561"
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=300,
        hindcast_length=0,
        forecast_length=365,
        var_t=[
            # although the name is hourly, it might be daily according to your choice
            "total_precipitation_hourly",
            "potential_evaporation_hourly",
            "snow_depth_water_equivalent",
            "snowfall_hourly",
            "dewpoint_temperature_2m",
            "temperature_2m",
        ],
        var_c=[
            # "sgr_dk_sav",
            # "pet_mm_syr",
            # "slp_dg_sav",
            # "for_pc_sse",
            # "pre_mm_syr",
            # "slt_pc_sav",
            # "swc_pc_syr",
            # "soc_th_sav",
            # "cly_pc_sav",
            # "ari_ix_sav",
            # "snd_pc_sav",
            # "ele_mt_sav",
            # "area",
            # "tmp_dc_syr",
            # "crp_pc_sse",
            # "lit_cl_smj",
            # "wet_cl_smj",
            # "snw_pc_syr",
            # "glc_cl_smj",
        ],
        # NOTE: although we set total_evaporation_hourly as output, it is not used in the training process
        var_out=["streamflow", "total_evaporation_hourly"],
        n_output=2,
        # TODO: if chose "mean", metric results' format is different, this should be refactored
        fill_nan=["no", "no"],
        target_as_input=0,
        constant_only=0,
        # train_epoch=100,
        train_epoch=2,
        save_epoch=10,
        model_loader={
            "load_way": "specified",
            # "test_epoch": 100,
            "test_epoch": 2,
        },
        warmup_length=365,
        opt="Adadelta",
        which_first_tensor="sequence",
        # train_mode=0,
        # weight_path="C:\\Users\\wenyu\\code\\torchhydro\\results\\test_camels\\expdpl61561201\\10_September_202402_32PM_model.pth",
        # continue_train=0,
    )
