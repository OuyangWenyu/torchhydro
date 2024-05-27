import os
import pytest

from torchhydro.configs.config import cmd, default_config_file
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
        rho=20,
        min_time_type="D",
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
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
        rho=365,  # batch_size=100, rho=365,
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
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_gpm_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[0],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 17,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 168,
            "prec_window": 1,  # 将前序径流一起作为输出，选择的时段数，该值需小于等于rho，建议置为1
        },
        model_loader={"load_way": "best"},
        gage_id=basin4test,
        batch_size=512,
        rho=720,
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
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=1,
        save_epoch=1,
        train_period=[("2016-06-01", "2016-12-31")],
        test_period=[("2015-06-01", "2015-10-31")],
        valid_period=[("2015-06-01", "2015-10-31")],
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
        rolling=False,
        long_seq_pred=False,
        early_stopping=True,
        patience=8,
        model_type="MTL",
        fill_nan=["no", "no"],
    )
