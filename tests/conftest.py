import os
import pytest
from typing import List, Dict, Any
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
import logging


@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configure the logger"""
    logging.basicConfig(level=logging.INFO)
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)


@pytest.fixture(scope="session")
def basin4test() -> List[str]:
    """Read the basin ID list, only choose final 5 basins as test data"""
    # For now, use a hardcoded list to avoid dependency on a local file
    return [
        "01055000",
        "01057000",
        "01170100",
    ]


@pytest.fixture()
def config_data() -> Dict[str, Any]:
    return default_config_file()


@pytest.fixture()
def args() -> Any:
    project_name = os.path.join("test_camels", "exp1")
    source_path = SETTING["local_data_path"]["datasets-origin"]
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
        var_t=[
            "precipitation",
            "daylight_duration",
            "solar_radiation",
            "temperature_max",
            "temperature_min",
            "vapor_pressure",
        ],
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
def mtl_args() -> Any:
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
def trans_args(basin4test: List[str]) -> Any:
    var_c_target = [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "frac_forest",
        "lai_max",
        "lai_diff",
    ]
    var_c_source = [
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
    ]
    var_t_target = [
        "daylight_duration",
        "precipitation",
        "solar_radiation",
    ]
    var_t_source = [
        "precipitation",
        "daylight_duration",
        "solar_radiation",
        "temperature_max",
        "temperature_min",
        "vapor_pressure",
    ]
    # This config is from test_transfer_learning.py.
    # It requires a pre-trained model specified by weight_path, which is not available in the repo.
    # Setting weight_path to None to make it runnable as a unit test, but the test logic
    # that relies on it will likely fail. The test should be marked as requires_data or skipped.
    weight_path = None
    project_name = "test_camels/exp4"
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": SETTING["local_data_path"]["datasets-origin"],
        },
        ctx=[0],
        model_type="TransLearn",
        model_name="KaiLSTM",
        model_hyperparam={
            "linear_size": len(var_c_target) + len(var_t_target),
            "n_input_features": len(var_c_source) + len(var_t_source),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        loss_func="RMSESum",
        batch_size=5,
        hindcast_length=0,
        forecast_length=20,
        rs=1234,
        train_period=["2010-10-01", "2011-10-01"],
        test_period=["2011-10-01", "2012-10-01"],
        scaler="DapengScaler",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },
        continue_train=True,
        train_epoch=20,
        save_epoch=10,
        var_t=var_t_target,
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=basin4test,
    )


@pytest.fixture()
def dpl_args() -> Any:
    project_name = os.path.join("test_camels", "expdpl001")
    train_period = ["1985-10-01", "1986-04-01"]
    # valid_period = ["1995-10-01", "2000-10-01"]
    valid_period = None
    test_period = ["2000-10-01", "2001-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": SETTING["local_data_path"]["datasets-origin"],
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
                StandardVariable.PRECIPITATION,
                StandardVariable.EVAPOTRANSPIRATION,
                StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
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
            StandardVariable.PRECIPITATION,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
            StandardVariable.DAYLIGHT_DURATION,
            StandardVariable.SOLAR_RADIATION,
            StandardVariable.SNOW_WATER_EQUIVALENT,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.VAPOR_PRESSURE,
        ],
        # var_t_type=["nldas"],
        var_out=["streamflow"],
        var_c=[
            "DRAIN_SQKM",
            "PPTAVG_BASIN",
            "T_AVG_BASIN",
            "T_MAX_BASIN",
            "T_MAXSTD_BASIN",
            "T_MIN_BASIN",
            "T_MINSTD_BASIN",
            "RH_BASIN",
            "PET",
            "GEOL_REEDBUSH_DOM",
            "GEOL_REEDBUSH_DOM_PCT",
            "BFI_AVE",
            "PERDUN",
            "PERHOR",
            "TOPWET",
            "CONTACT",
            "RUNAVE7100",
            "DEVNLCD06",
            "FORESTNLCD06",
            "PLANTNLCD06",
            "IMPNLCD06",
            "HGA",
            "HGB",
            "HGAD",
            "HGC",
            "HGD",
            "HGAC",
            "HGBD",
            "HGCD",
            "HGBC",
            "HGVAR",
            "AWCAVE",
            "PERMAVE",
            "BDAVE",
            "WTDEPAVE",
            "ROCKDEPAVE",
            "ELEV_MEAN_M_BASIN",
            "ELEV_MAX_M_BASIN",
            "ELEV_MIN_M_BASIN",
            "ELEV_MEDIAN_M_BASIN",
            "ELEV_STD_M_BASIN",
            "RRMEAN",
            "RRMEDIAN",
            "SLOPE_PCT",
        ],
        var_to_source_map={
            "dayl(s)": "gages",
            "prcp(mm/day)": "gages",
            "srad(W/m2)": "gages",
            "swe(mm)": "gages",
            "tmax(C)": "gages",
            "tmin(C)": "gages",
            "vp(Pa)": "gages",
            "map(mm/day)": "mopex",
            "pe(mm/day)": "mopex",
            "DRAIN_SQKM": "gages",
            "PPTAVG_BASIN": "gages",
            "T_AVG_BASIN": "gages",
            "T_MAX_BASIN": "gages",
            "T_MAXSTD_BASIN": "gages",
            "T_MIN_BASIN": "gages",
            "T_MINSTD_BASIN": "gages",
            "RH_BASIN": "gages",
            "PET": "gages",
            "GEOL_REEDBUSH_DOM": "gages",
            "GEOL_REEDBUSH_DOM_PCT": "gages",
            "BFI_AVE": "gages",
            "PERDUN": "gages",
            "PERHOR": "gages",
            "TOPWET": "gages",
            "CONTACT": "gages",
            "RUNAVE7100": "gages",
            "DEVNLCD06": "gages",
            "FORESTNLCD06": "gages",
            "PLANTNLCD06": "gages",
            "IMPNLCD06": "gages",
            "HGA": "gages",
            "HGB": "gages",
            "HGAD": "gages",
            "HGC": "gages",
            "HGD": "gages",
            "HGAC": "gages",
            "HGBD": "gages",
            "HGCD": "gages",
            "HGBC": "gages",
            "HGVAR": "gages",
            "AWCAVE": "gages",
            "PERMAVE": "gages",
            "BDAVE": "gages",
            "WTDEPAVE": "gages",
            "ROCKDEPAVE": "gages",
            "ELEV_MEAN_M_BASIN": "gages",
            "ELEV_MAX_M_BASIN": "gages",
            "ELEV_MIN_M_BASIN": "gages",
            "ELEV_MEDIAN_M_BASIN": "gages",
            "ELEV_STD_M_BASIN": "gages",
            "RRMEAN": "gages",
            "RRMEDIAN": "gages",
            "SLOPE_PCT": "gages",
        },
        sampler="KuaiSampler",
        n_output=1,
        opt="Adam",
        train_epoch=2,
        fill_nan=["no", "mean"],
        model_loader={"load_way": "specified", "test_epoch": 2},
        lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        which_first_tensor="sequence",
    )


@pytest.fixture()
def fusion_lstm_args():
    project_name = os.path.join("test_fusion", "exp001")
    multi_source_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    multi_source_interim_dir = SETTING["local_data_path"]["datasets-interim"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_names": [
                "gages",
                "mopex",
            ],
            "source_paths": [
                os.path.join(multi_source_origin_dir, "gages"),
                os.path.join(multi_source_interim_dir, "mopex"),
            ],
        },
        ctx=[0],
        model_type="Normal",
        # model_name="ChuFusion",
        model_name="CoupledLSTM",
        # model_hyperparam={
        #     "num_sources": 4,
        #     "out_channels": 2,
        #     "transformer_heads": 4,
        #     "transformer_layers": 2,
        #     "n_input_features": 23,
        #     "n_output_features": 2,
        #     "n_hidden_states": 64,
        #     "layer_hidden_size": 32,
        #     "kernel_size_hydrograph": 3,
        #     "warmup_length": 365,
        #     "param_limit_func": "sigmoid",
        #     "param_test_way": "final",
        #     "source_book": "HF",
        #     "source_type": "sources",
        #     "kernel_size": 3,
        #     "padding": 1,
        # },
        model_hyperparam={
            "precipitation_fusion_input_feature": 2,
            "precipitation_fusion_output_feature": 1,
            "precipitation_fusion_hidden_states": 256,
            "flow_prediction_input_feature": 51,
            "flow_prediction_output_feature": 1,
            "flow_prediction_hidden_states": 256,
        },
        loss_func="RMSESum",
        gage_id_file=os.path.join(multi_source_origin_dir, "fusion_gage_id.csv"),
        batch_size=32,
        forecast_history=0,
        forecast_length=365,
        # rho=365,
        var_t=[
            "prcp",
            "map",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        # var_t_type=["nldas"],
        var_out=["streamflow"],
        var_c=[
            "DRAIN_SQKM",
            "PPTAVG_BASIN",
            "T_AVG_BASIN",
            "T_MAX_BASIN",
            "T_MAXSTD_BASIN",
            "T_MIN_BASIN",
            "T_MINSTD_BASIN",
            "RH_BASIN",
            "PET",
            "GEOL_REEDBUSH_DOM",
            "GEOL_REEDBUSH_DOM_PCT",
            "BFI_AVE",
            "PERDUN",
            "PERHOR",
            "TOPWET",
            "CONTACT",
            "RUNAVE7100",
            "DEVNLCD06",
            "FORESTNLCD06",
            "PLANTNLCD06",
            "IMPNLCD06",
            "HGA",
            "HGB",
            "HGAD",
            "HGC",
            "HGD",
            "HGAC",
            "HGBD",
            "HGCD",
            "HGBC",
            "HGVAR",
            "AWCAVE",
            "PERMAVE",
            "BDAVE",
            "WTDEPAVE",
            "ROCKDEPAVE",
            "ELEV_MEAN_M_BASIN",
            "ELEV_MAX_M_BASIN",
            "ELEV_MIN_M_BASIN",
            "ELEV_MEDIAN_M_BASIN",
            "ELEV_STD_M_BASIN",
            "RRMEAN",
            "RRMEDIAN",
            "SLOPE_PCT",
        ],
        var_to_source_map={
            "dayl(s)": "gages",
            "prcp(mm/day)": "gages",
            "srad(W/m2)": "gages",
            "swe(mm)": "gages",
            "tmax(C)": "gages",
            "tmin(C)": "gages",
            "vp(Pa)": "gages",
            "map(mm/day)": "mopex",
            "pe(mm/day)": "mopex",
            "DRAIN_SQKM": "gages",
            "PPTAVG_BASIN": "gages",
            "T_AVG_BASIN": "gages",
            "T_MAX_BASIN": "gages",
            "T_MAXSTD_BASIN": "gages",
            "T_MIN_BASIN": "gages",
            "T_MINSTD_BASIN": "gages",
            "RH_BASIN": "gages",
            "PET": "gages",
            "GEOL_REEDBUSH_DOM": "gages",
            "GEOL_REEDBUSH_DOM_PCT": "gages",
            "BFI_AVE": "gages",
            "PERDUN": "gages",
            "PERHOR": "gages",
            "TOPWET": "gages",
            "CONTACT": "gages",
            "RUNAVE7100": "gages",
            "DEVNLCD06": "gages",
            "FORESTNLCD06": "gages",
            "PLANTNLCD06": "gages",
            "IMPNLCD06": "gages",
            "HGA": "gages",
            "HGB": "gages",
            "HGAD": "gages",
            "HGC": "gages",
            "HGD": "gages",
            "HGAC": "gages",
            "HGBD": "gages",
            "HGCD": "gages",
            "HGBC": "gages",
            "HGVAR": "gages",
            "AWCAVE": "gages",
            "PERMAVE": "gages",
            "BDAVE": "gages",
            "WTDEPAVE": "gages",
            "ROCKDEPAVE": "gages",
            "ELEV_MEAN_M_BASIN": "gages",
            "ELEV_MAX_M_BASIN": "gages",
            "ELEV_MIN_M_BASIN": "gages",
            "ELEV_MEDIAN_M_BASIN": "gages",
            "ELEV_STD_M_BASIN": "gages",
            "RRMEAN": "gages",
            "RRMEDIAN": "gages",
            "SLOPE_PCT": "gages",
        },
        train_period=["1983-10-01", "1993-09-30"],
        test_period=["1993-10-01", "2003-09-30"],
        dataset="PrecipitationFusionDataset",
        # sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=1,
        opt="Adam",
        train_epoch=50,
        save_epoch=2,
        fill_nan=["no"],
        model_loader={"load_way": "specified", "test_epoch": 50},
        lr_scheduler={0: 1, 10: 0.5, 20: 0.2},
        which_first_tensor="sequence",
    )


@pytest.fixture()
def mopex_lstm_args():
    project_name = "test_mopex/exp001"
    source_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    source_interim_dir = SETTING["local_data_path"]["datasets-interim"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_names": [
                "gages",
                "mopex",
            ],
            "source_paths": [
                os.path.join(source_origin_dir, "gages"),
                os.path.join(source_origin_dir, "mopex"),
            ],
        },
        ctx=[2],
        model_type="Normal",
        # model_name="ChuFusion",
        model_name="KuaiLSTM",
        # model_hyperparam={
        #     "num_sources": 4,
        #     "out_channels": 2,
        #     "transformer_heads": 4,
        #     "transformer_layers": 2,
        #     "n_input_features": 23,
        #     "n_output_features": 2,
        #     "n_hidden_states": 64,
        #     "layer_hidden_size": 32,
        #     "kernel_size_hydrograph": 3,
        #     "warmup_length": 365,
        #     "param_limit_func": "sigmoid",
        #     "param_test_way": "final",
        #     "source_book": "HF",
        #     "source_type": "sources",
        #     "kernel_size": 3,
        #     "padding": 1,
        # },
        model_hyperparam={
            "n_input_features": 51,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        gage_id_file=os.path.join(source_origin_dir, "fusion_gage_id.csv"),
        batch_size=32,
        forecast_history=0,
        forecast_length=365,
        # rho=365,
        var_t=[
            "map",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        # var_t_type=["nldas"],
        var_out=["streamflow"],
        var_c=[
            "DRAIN_SQKM",
            "PPTAVG_BASIN",
            "T_AVG_BASIN",
            "T_MAX_BASIN",
            "T_MAXSTD_BASIN",
            "T_MIN_BASIN",
            "T_MINSTD_BASIN",
            "RH_BASIN",
            "PET",
            "GEOL_REEDBUSH_DOM",
            "GEOL_REEDBUSH_DOM_PCT",
            "BFI_AVE",
            "PERDUN",
            "PERHOR",
            "TOPWET",
            "CONTACT",
            "RUNAVE7100",
            "DEVNLCD06",
            "FORESTNLCD06",
            "PLANTNLCD06",
            "IMPNLCD06",
            "HGA",
            "HGB",
            "HGAD",
            "HGC",
            "HGD",
            "HGAC",
            "HGBD",
            "HGCD",
            "HGBC",
            "HGVAR",
            "AWCAVE",
            "PERMAVE",
            "BDAVE",
            "WTDEPAVE",
            "ROCKDEPAVE",
            "ELEV_MEAN_M_BASIN",
            "ELEV_MAX_M_BASIN",
            "ELEV_MIN_M_BASIN",
            "ELEV_MEDIAN_M_BASIN",
            "ELEV_STD_M_BASIN",
            "RRMEAN",
            "RRMEDIAN",
            "SLOPE_PCT",
        ],
        var_to_source_map={
            "dayl(s)": "gages",
            "prcp(mm/day)": "gages",
            "srad(W/m2)": "gages",
            "swe(mm)": "gages",
            "tmax(C)": "gages",
            "tmin(C)": "gages",
            "vp(Pa)": "gages",
            "map(mm/day)": "mopex",
            "pe(mm/day)": "mopex",
            "DRAIN_SQKM": "gages",
            "PPTAVG_BASIN": "gages",
            "T_AVG_BASIN": "gages",
            "T_MAX_BASIN": "gages",
            "T_MAXSTD_BASIN": "gages",
            "T_MIN_BASIN": "gages",
            "T_MINSTD_BASIN": "gages",
            "RH_BASIN": "gages",
            "PET": "gages",
            "GEOL_REEDBUSH_DOM": "gages",
            "GEOL_REEDBUSH_DOM_PCT": "gages",
            "BFI_AVE": "gages",
            "PERDUN": "gages",
            "PERHOR": "gages",
            "TOPWET": "gages",
            "CONTACT": "gages",
            "RUNAVE7100": "gages",
            "DEVNLCD06": "gages",
            "FORESTNLCD06": "gages",
            "PLANTNLCD06": "gages",
            "IMPNLCD06": "gages",
            "HGA": "gages",
            "HGB": "gages",
            "HGAD": "gages",
            "HGC": "gages",
            "HGD": "gages",
            "HGAC": "gages",
            "HGBD": "gages",
            "HGCD": "gages",
            "HGBC": "gages",
            "HGVAR": "gages",
            "AWCAVE": "gages",
            "PERMAVE": "gages",
            "BDAVE": "gages",
            "WTDEPAVE": "gages",
            "ROCKDEPAVE": "gages",
            "ELEV_MEAN_M_BASIN": "gages",
            "ELEV_MAX_M_BASIN": "gages",
            "ELEV_MIN_M_BASIN": "gages",
            "ELEV_MEDIAN_M_BASIN": "gages",
            "ELEV_STD_M_BASIN": "gages",
            "RRMEAN": "gages",
            "RRMEDIAN": "gages",
            "SLOPE_PCT": "gages",
        },
        train_period=["1983-10-01", "1993-09-30"],
        test_period=["1993-10-01", "2003-09-30"],
        dataset="MopexPrecipitationGagesAttrFusionDataset",
        # sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=1,
        opt="Adam",
        train_epoch=50,
        save_epoch=2,
        fill_nan=["no"],
        model_loader={"load_way": "specified", "test_epoch": 2},
        # lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        which_first_tensor="sequence",
    )


@pytest.fixture()
def gages_lstm_args():
    project_name = "test_gages/exp001"
    source_origin_dir = SETTING["local_data_path"]["datasets-origin"]
    source_interim_dir = SETTING["local_data_path"]["datasets-interim"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "gages",
            "source_path": os.path.join(source_origin_dir, "gages"),
        },
        ctx=[2],
        model_type="Normal",
        # model_name="ChuFusion",
        model_name="KuaiLSTM",
        # model_hyperparam={
        #     "num_sources": 4,
        #     "out_channels": 2,
        #     "transformer_heads": 4,
        #     "transformer_layers": 2,
        #     "n_input_features": 23,
        #     "n_output_features": 2,
        #     "n_hidden_states": 64,
        #     "layer_hidden_size": 32,
        #     "kernel_size_hydrograph": 3,
        #     "warmup_length": 365,
        #     "param_limit_func": "sigmoid",
        #     "param_test_way": "final",
        #     "source_book": "HF",
        #     "source_type": "sources",
        #     "kernel_size": 3,
        #     "padding": 1,
        # },
        model_hyperparam={
            "n_input_features": 51,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        gage_id_file=os.path.join(source_origin_dir, "fusion_gage_id.csv"),
        batch_size=32,
        forecast_history=0,
        forecast_length=365,
        # rho=365,
        var_t=[
            "prcp",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        # var_t_type=["nldas"],
        var_out=["streamflow"],
        var_c=[
            "DRAIN_SQKM",
            "PPTAVG_BASIN",
            "T_AVG_BASIN",
            "T_MAX_BASIN",
            "T_MAXSTD_BASIN",
            "T_MIN_BASIN",
            "T_MINSTD_BASIN",
            "RH_BASIN",
            "PET",
            "GEOL_REEDBUSH_DOM",
            "GEOL_REEDBUSH_DOM_PCT",
            "BFI_AVE",
            "PERDUN",
            "PERHOR",
            "TOPWET",
            "CONTACT",
            "RUNAVE7100",
            "DEVNLCD06",
            "FORESTNLCD06",
            "PLANTNLCD06",
            "IMPNLCD06",
            "HGA",
            "HGB",
            "HGAD",
            "HGC",
            "HGD",
            "HGAC",
            "HGBD",
            "HGCD",
            "HGBC",
            "HGVAR",
            "AWCAVE",
            "PERMAVE",
            "BDAVE",
            "WTDEPAVE",
            "ROCKDEPAVE",
            "ELEV_MEAN_M_BASIN",
            "ELEV_MAX_M_BASIN",
            "ELEV_MIN_M_BASIN",
            "ELEV_MEDIAN_M_BASIN",
            "ELEV_STD_M_BASIN",
            "RRMEAN",
            "RRMEDIAN",
            "SLOPE_PCT",
        ],
        var_to_source_map={
            "dayl(s)": "gages",
            "prcp(mm/day)": "gages",
            "srad(W/m2)": "gages",
            "swe(mm)": "gages",
            "tmax(C)": "gages",
            "tmin(C)": "gages",
            "vp(Pa)": "gages",
            "map(mm/day)": "mopex",
            "pe(mm/day)": "mopex",
            "DRAIN_SQKM": "gages",
            "PPTAVG_BASIN": "gages",
            "T_AVG_BASIN": "gages",
            "T_MAX_BASIN": "gages",
            "T_MAXSTD_BASIN": "gages",
            "T_MIN_BASIN": "gages",
            "T_MINSTD_BASIN": "gages",
            "RH_BASIN": "gages",
            "PET": "gages",
            "GEOL_REEDBUSH_DOM": "gages",
            "GEOL_REEDBUSH_DOM_PCT": "gages",
            "BFI_AVE": "gages",
            "PERDUN": "gages",
            "PERHOR": "gages",
            "TOPWET": "gages",
            "CONTACT": "gages",
            "RUNAVE7100": "gages",
            "DEVNLCD06": "gages",
            "FORESTNLCD06": "gages",
            "PLANTNLCD06": "gages",
            "IMPNLCD06": "gages",
            "HGA": "gages",
            "HGB": "gages",
            "HGAD": "gages",
            "HGC": "gages",
            "HGD": "gages",
            "HGAC": "gages",
            "HGBD": "gages",
            "HGCD": "gages",
            "HGBC": "gages",
            "HGVAR": "gages",
            "AWCAVE": "gages",
            "PERMAVE": "gages",
            "BDAVE": "gages",
            "WTDEPAVE": "gages",
            "ROCKDEPAVE": "gages",
            "ELEV_MEAN_M_BASIN": "gages",
            "ELEV_MAX_M_BASIN": "gages",
            "ELEV_MIN_M_BASIN": "gages",
            "ELEV_MEDIAN_M_BASIN": "gages",
            "ELEV_STD_M_BASIN": "gages",
            "RRMEAN": "gages",
            "RRMEDIAN": "gages",
            "SLOPE_PCT": "gages",
        },
        train_period=["1983-10-01", "1993-09-30"],
        test_period=["1993-10-01", "2003-09-30"],
        dataset="StreamflowDataset",
        # sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=1,
        opt="Adam",
        train_epoch=50,
        save_epoch=2,
        fill_nan=["no"],
        model_loader={"load_way": "specified", "test_epoch": 2},
        # lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        # target_as_input=0,
        # constant_only=1,
        # warmup_length=30,
        which_first_tensor="sequence",
    )


@pytest.fixture()
def seq2seq_config() -> Dict[str, Any]:
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
        ctx=[2],
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
        gage_id=["21400800", "21401550", "21401300", "21401900"],
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
def dpl4hbv_selfmadehydrodataset_args() -> Any:
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
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "FDSources",
            },
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
def dpl4xaj_selfmadehydrodataset_args() -> Any:
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


@pytest.fixture()
def BALSTM_config():
    project_name = os.path.join("train_with_LongTerm", "balstm")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": os.path.join(
                # gmspa means ​​Global Monthly Streamflow Predictors Archive​
                SETTING["local_data_path"]["datasets-interim"],
                "gmspa",
            ),
            "other_settings": {
                "time_unit": ["1M"],
                "trange4cache": ["1800-01-01", "2024-12-31"],
            },
        },
        ctx=[0],
        model_name="BALSTM",
        model_hyperparam={
            "output_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.4,
            "input_size_dyn": 13,
            "input_size_glo": 106,
            "output_size": 1,
            "input_size_sta": 10,  # len(var_c)
        },
        model_loader={"load_way": "best"},
        gage_id=[
            "grdc_4234010",
            "grdc_6975140",
            "grdc_6976400",
            "grdc_6976450",
            "grdc_6978250",
        ],
        batch_size=32,
        forecast_history=10,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=[
            "d2m",
            "pev",
            "ro",
            "slhf",
            "sp",
            "sro",
            "swvl",
            "tp",
            "u10",
            "t2m",
            "v10",
            "sd",
            "sshf",
        ],
        var_c=[
            "sgr_dk_sav",
            "glc_pc_s06",
            "glc_pc_s07",
            "nli_ix_sav",
            "glc_pc_s04",
            "glc_pc_s05",
            "glc_pc_s02",
            "glc_pc_s03",
            "glc_pc_s01",
            "pet_mm_syr",
        ],
        var_out=["streamflow"],
        dataset="BALSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=1,
        save_epoch=1,
        train_period=["1951-01-01", "1970-12-31"],
        test_period=["1971-01-01", "1972-10-31"],
        # test_period=None,
        # valid_period=["1961-01-01", "1963-12-31"],
        valid_period=None,
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.0001},
        which_first_tensor="batch",
        rolling=False,
        calc_metrics=True,
        early_stopping=True,
        patience=1,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


@pytest.fixture()
def selfmadehydrodataset_args() -> Any:
    project_name = os.path.join("test_selfmadehydrodataset", "exp1")
    DEVICE = -1
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "songliaorrevent",
                "offset_to_utc": True,  # if you use Chinese dataset with start time 08:00, you need to set it to True
            },
        },
        ctx=[DEVICE],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 16,
        },
        gage_id=["songliao_20800900"],
        batch_size=8,
        hindcast_length=0,
        forecast_length=20,
        # for flood event dataset, we need to set the forecast length for testing
        frwin=20,
        min_time_unit="D",
        min_time_interval="1",
        var_t=["rain"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        dataset="FloodEventDataset",
        scaler="DapengScaler",
        variable_length_cfgs={
            # whether to use variable length training
            "use_variable_length": True,
            # variable length type:
            # - "fixed": use predefined lengths (replaces old multi_length_training)
            # - "dynamic": automatic padding with mask (replaces old mask_cfgs)
            "variable_length_type": "dynamic",
            # for "fixed" type: specify exact sequence lengths to use
            "fixed_lengths": None,
            # Pad strategy: "Pad" or "multi_table" (multi_table not fully tested yet)
            "pad_strategy": "Pad",
        },
        train_epoch=2,
        save_epoch=1,
        model_loader={"load_way": "specified", "test_epoch": 2},
        train_period=["1980-01-01", "2010-12-31"],
        valid_period=["2011-01-01", "2015-12-31"],
        test_period=["2016-01-01", "2020-12-31"],
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "device": [DEVICE],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},
    )


@pytest.fixture()
def selfmadehydrodataset_dpl4xaj_args() -> Any:
    project_name = os.path.join("test_selfmadehydrodataset", "dpl4xajexp1")
    DEVICE = -1
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["1D"],
                "dataset_name": "songliaorrevent",
                "offset_to_utc": True,  # if you use Chinese dataset with start time 08:00, you need to set it to True
            },
        },
        ctx=[DEVICE],
        model_name="DplLstmXaj",
        model_hyperparam={
            "n_input_features": 2,
            "n_output_features": 15,
            "n_hidden_states": 16,
            "kernel_size": 15,
            "warmup_length": 30,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
            "return_et": False,
        },
        gage_id=["songliao_21401550"],
        batch_size=8,
        warmup_length=30,
        hindcast_length=0,
        forecast_length=20,
        # for flood event dataset, we need to set the forecast length for testing
        frwin=20,
        min_time_unit="D",
        min_time_interval="1",
        var_t=["rain", "potential_evaporation_hourly"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        target_as_input=False,
        dataset="FloodEventDplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "inflow",
            ],
            "gamma_norm_cols": [
                "rain",
                "potential_evaporation_hourly",
            ],
            "pbm_norm": True,
        },
        variable_length_cfgs={
            # whether to use variable length training
            "use_variable_length": True,
            # variable length type:
            # - "fixed": use predefined lengths (replaces old multi_length_training)
            # - "dynamic": automatic padding with mask (replaces old mask_cfgs)
            "variable_length_type": "dynamic",
            # for "fixed" type: specify exact sequence lengths to use
            "fixed_lengths": None,
            # Pad strategy: "Pad" or "multi_table" (multi_table not fully tested yet)
            "pad_strategy": "Pad",
        },
        train_epoch=2,
        save_epoch=1,
        model_loader={"load_way": "specified", "test_epoch": 2},
        train_period=["1980-01-01", "2010-12-31"],
        valid_period=["2011-01-01", "2015-12-31"],
        test_period=["2016-01-01", "2020-12-31"],
        loss_func="HybridFloodloss",
        loss_param={
            "mae_weight": 0.5,
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},
    )


@pytest.fixture()
def flood_event_datasource_args() -> Any:
    """
    Configuration for FloodEventDatasource with enhanced data support and score weighting
    """
    project_name = os.path.join("test_flood_event_score", "exp20251211_real")
    DEVICE = -1
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
                "dataset_name": "songliaorrevent",
                "net_rain_key": "net_rain",
                "obs_flow_key": "inflow",
                "delta_t_hours": 3.0,
            },
        },
        ctx=[DEVICE],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 1,
            "output_size": 1,
            "hidden_size": 16,
        },
        gage_id=["songliao_21401550"],
        batch_size=8,
        warmup_length=0,  
        hindcast_length=56,
        forecast_length=24,
        frwin=24,
        min_time_unit="h",
        min_time_interval="3",
        var_t=["rain"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        target_as_input=False,  # 不将target作为输入特征
        dataset="FloodEventDataset",
        scaler="DapengScaler",
        variable_length_cfgs={
            "use_variable_length": True,
            "variable_length_type": "dynamic",
            "fixed_lengths": None,
            "pad_strategy": "Pad",
        },
        train_epoch=100,
        save_epoch=1,
        model_loader={"load_way": "specified", "test_epoch": 2},
        train_period=[
            "1980-01-01-02", "2017-12-31-23",   # 实测数据
            # ["2017-01-01-02", "2017-12-31-23"],
        ],
        valid_period=["2018-01-01-02", "2024-12-31-23"],
        test_period=["2018-01-01-02", "2023-12-31-23"],
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "device": [DEVICE],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},
        peak_aligned_sampling={
            "enabled": True,          # Enable peak-aligned sampling
            "slide_step": 1,          # Sliding window step size
            "peak_selection": "highest" # "first" or "highest"
        },
        # train_mode= False,
        # stat_dict_file = "D:\project\songliao\torchhydro\results\test_flood_event_train\exp20251211_real\dapengscaler_stat.json"
    )
    return args


@pytest.fixture()
def flood_event_datasource_score_args() -> Any:
    """Configuration for FloodEventDatasource with score threshold filtering

    This config is designed for NC files (v1.2+) that use score_threshold filtering
    during data loading. Data quality is controlled by threshold (default: 60),
    and all loaded data has equal weight during training.

    Features:
    - Uses augmented data from run_data_postprocessing.py (score >= threshold)
    - No score-based weighting (all data weighted equally)
    - Compatible with NC files containing: rain, inflow, flood_event
    - Supports both real historical data and augmented data
    """
    project_name = os.path.join("test_flood_event_Seq2Seq_mask", "exp20251215_real")
    DEVICE = -1
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
                "dataset_name": "songliaorrevent",
                "net_rain_key": "net_rain",
                "obs_flow_key": "inflow",
                "delta_t_hours": 3.0,
            },
        },
        ctx=[DEVICE],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 1,           # encoder输入特征数（降雨）
            "de_input_size": 2,           # decoder输入维度 = 降雨(1) + 上一步输出(1)
            "output_size": 1,             # 输出特征数（只预测径流，flood_event仅用于loss加权）
            "hidden_size": 16,            # LSTM隐藏层大小
            "forecast_length": 24,        # 预测步长（必须与training_cfgs.forecast_length一致）
            "hindcast_output_window": 56, # 后验期输出窗口（输出完整后验期）
            "teacher_forcing_ratio": 0.5, # teacher forcing比例（训练时使用真实值的概率）
        },
        gage_id=["songliao_21401550"],
        batch_size=8,
        warmup_length=0,        # 预热期长度
        hindcast_length=56,      # 模型用于预测的历史观测数据,7天
        forecast_length=24,     # 模型向未来预测的时间步长度1天,决定样本数量
        frwin=24,               # 测试/验证阶段使用的预测窗口长度
        min_time_unit="h",
        min_time_interval="3",
        var_t=["rain"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        target_as_input=False,  # target作为输入特征
        dataset="FloodEventSeq2SeqDataset",
        scaler="DapengScaler",
        variable_length_cfgs={
            "use_variable_length": True,
            "variable_length_type": "dynamic",
            "fixed_lengths": None,
            "pad_strategy": "Pad",
        },
        train_epoch=20,  # 训练轮数
        save_epoch=1,
        model_loader={"load_way": "best"},  # 使用最佳模型
        # 训练周期：混合真实数据和增强数据
        train_period=[
            "1984-01-01-02", "2017-12-31-23",  # 真实历史数据
            # ["2026-07-01-02", "3150-12-31-23"],  # 增强数据
        ],
        valid_period=["2018-01-01-02", "2024-12-30-23"],  # 真实数据
        test_period=["2018-01-01-02", "2023-12-31-23"],  # 真实数据
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            # Score-based weighting parameters (for augmented data with quality scores 1-100)
            # Uncomment below to enable score-based weighting:
            "use_score_weighting": False,    # Enable score-based weighting (flood_mask: -1/0=non-flood, 1-100=scores)
            "score_weight_min": 0.5,        # Weight for lowest quality (score=1)
            "score_weight_max": 3.0,        # Weight for highest quality (score=100)
            "score_alpha": 3.0,             # Control high-quality emphasis (>1: emphasize high scores)
            "device": [DEVICE],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},  # FloodEventSeq2SeqDataset 支持洪水期评估
        peak_aligned_sampling={
            "enabled": True,          # Enable peak-aligned sampling
            "slide_step": 1,          # Sliding window step size
            "peak_selection": "first" # "first" or "highest"
        },
        # train_mode= False,
        # stat_dict_file = "D:\project\songliao\torchhydro\results\test_flood_event_train\exp20251211_score\dapengscaler_stat.json"
    )
    return args


@pytest.fixture()
def flood_event_seq2seq_noforecast_args() -> Any:
    """Configuration for FloodEventSeq2SeqDataset with Seq2SeqNoForecast model (ablation study).

    This config uses the new Seq2SeqNoForecast model which does NOT use forecast period
    rainfall as input. The decoder only uses autoregressive predictions.

    Key differences from standard Seq2Seq:
    - model_name: "Seq2SeqNoForecast"
    - de_input_size: 1 (only previous output, no rainfall)
    - Dataset still returns decoder_input but model ignores it
    """
    project_name = os.path.join("test_flood_event_Seq2SeqNoForecast", "exp20251215_ablation")
    DEVICE = -1
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "floodeventdatasource",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {
                "time_unit": ["3h"],
                "dataset_name": "songliaorrevent",
                "net_rain_key": "net_rain",
                "obs_flow_key": "inflow",
                "delta_t_hours": 3.0,
            },
        },
        ctx=[DEVICE],
        model_name="Seq2SeqNoForecast",  # Use new model
        model_hyperparam={
            "en_input_size": 1,           # encoder输入特征数（降雨）
            "de_input_size": 1,           # decoder输入维度（将被模型忽略，内部使用output_size=1）
            "output_size": 1,             # 输出特征数（只预测径流，flood_event仅用于loss加权）
            "hidden_size": 16,            # LSTM隐藏层大小
            "forecast_length": 24,        # 预测步长（必须与training_cfgs.forecast_length一致）
            "hindcast_output_window": 56, # 后验期输出窗口（输出完整后验期）
            "teacher_forcing_ratio": 0.5, # teacher forcing比例（训练时使用真实值的概率）
        },
        hindcast_output_window=56,  # 数据集配置：后验期输出窗口
        gage_id=["songliao_21401550"],
        batch_size=8,
        warmup_length=0,
        hindcast_length=56,
        forecast_length=24,
        frwin=24,
        min_time_unit="h",
        min_time_interval="3",
        var_t=["rain"],
        t_rm_nan=False,
        var_c=["None"],
        c_rm_nan=False,
        var_out=["inflow", "flood_event"],
        target_as_input=False,
        dataset="FloodEventSeq2SeqDataset",
        scaler="DapengScaler",
        variable_length_cfgs={
            "use_variable_length": True,
            "variable_length_type": "dynamic",
            "fixed_lengths": None,
            "pad_strategy": "Pad",
        },
        train_epoch=20,
        save_epoch=1,
        model_loader={"load_way": "best"},
        train_period=[
            "1984-01-01-02", "2017-12-31-23",  # 真实历史数据
        ],
        valid_period=["2018-01-01-02", "2024-12-30-23"],
        test_period=["2018-01-01-02", "2023-12-31-23"],
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "use_score_weighting": False,
            "score_weight_min": 0.5,
            "score_weight_max": 3.0,
            "score_alpha": 3.0,
            "device": [DEVICE],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },
        which_first_tensor="sequence",
        valid_batch_mode="train",
        rolling=-1,
        evaluator={"eval_way": "floodevent"},
        peak_aligned_sampling={
            "enabled": True,
            "slide_step": 1,
            "peak_selection": "first"
        },
    )
