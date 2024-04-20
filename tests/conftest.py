import os
import pytest

from torchhydro.configs.config import cmd, default_config_file
from torchhydro import SETTING


@pytest.fixture()
def config_data():
    return default_config_file()


@pytest.fixture()
def args():
    project_name = "test_camels/exp1"
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
    project_name = "test_camels/expmtl001"
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
            "n_hidden_states": 64,
            "layer_hidden_size": 32,
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": [0],
            "item_weight": [1.0, 0.0],
            "limit_part": [1],
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
        batch_size=5,
        rho=30,  # batch_size=100, rho=365,
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
        train_period=["2015-04-01", "2016-04-01"],
        test_period=["2016-04-01", "2017-04-01"],
        dataset="FlexDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=2,
        fill_nan=["no", "mean"],
        model_loader={"load_way": "specified", "test_epoch": 2},
    )
