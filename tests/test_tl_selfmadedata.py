"""
Author: Wenyu Ouyang
Date: 2023-10-05 16:16:48
LastEditTime: 2024-07-10 17:15:26
LastEditors: Wenyu Ouyang
Description: Transfer learning for local basins with hydro_opendata
FilePath: /torchhydro/tests/test_tl_selfmadedata.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def var_c_target():
    return [
        "p_mean",
        "pet_mean",
        "Area",
        "geol_class_1st",
        "elev",
        # "SNDPPT",
    ]


@pytest.fixture()
def var_c_source():
    return [
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


@pytest.fixture()
def var_t_target():
    # mainly from ERA5LAND
    return ["total_precipitation", "potential_evaporation", "temperature_2m"]


@pytest.fixture()
def var_t_source():
    return ["dayl", "prcp", "srad", "tmax", "tmin", "vp"]


def test_transfer_gages_lstm_model(
    var_c_source, var_c_target, var_t_source, var_t_target
):
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    project_name = "test_camels/exptl4cc"
    args = cmd(
        sub=project_name,
        source="SelfMadeCAMELS",
        # cc means China continent
        source_path=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "camels_cc_v2"
        ),
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
        forecast_history=0,
        forecast_length=20,
        rs=1234,
        train_period=["2014-10-01", "2019-10-01"],
        test_period=["2019-10-01", "2021-10-01"],
        scaler="DapengScaler",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },
        continue_train=True,
        train_epoch=20,
        te=20,
        save_epoch=10,
        var_t=var_t_target,
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=[
            "86_changdian61561",
            "86_changdian62618",
        ],
    )
    cfg = default_config_file()
    update_cfg(cfg, args)
    train_and_evaluate(cfg)
    print("All processes are finished!")


@pytest.fixture()
def dpl_selfmadehydrodataset_args():
    project_name = os.path.join("test_camels", "expdpl002")
    train_period = ["1985-10-01", "1986-04-01"]
    # valid_period = ["1995-10-01", "2000-10-01"]
    valid_period = None
    test_period = ["2000-10-01", "2001-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": "/mnt/c/Users/wenyu/OneDrive/Research/research_topic_advancement/research_progress_plan/data4dpl/dplARdata",
        },
        ctx=[0],
        # model_name="DplLstmXaj",
        model_name="DplAttrXaj",
        model_hyperparam={
            # "n_input_features": 25,
            "n_input_features": 19,
            "n_output_features": 15,
            "n_hidden_states": 256,
            "kernel_size": 15,
            "warmup_length": 30,
            "param_limit_func": "clamp",
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_sum",
                "potential_evaporation_sum",
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
        forecast_history=0,
        forecast_length=60,
        var_t=[
            "total_precipitation_sum",
            "potential_evaporation_sum",
            "snow_depth_water_equivalent",
            "surface_net_solar_radiation_sum",
            "surface_pressure",
            "temperature_2m",
        ],
        var_c=[
            "sgr_dk_sav",
            "pet_mm_syr",
            "slp_dg_sav",
            "for_pc_sse",
            "pre_mm_syr",
            "slt_pc_sav",
            "swc_pc_syr",
            "soc_th_sav",
            "cly_pc_sav",
            "ari_ix_sav",
            "snd_pc_sav",
            "ele_mt_sav",
            "area",
            "tmp_dc_syr",
            "crp_pc_sse",
            "lit_cl_smj",
            "wet_cl_smj",
            "snw_pc_syr",
            "glc_cl_smj",
        ],
        var_out=["streamflow"],
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


def test_dpl_selfmadehydrodataset(dpl_selfmadehydrodataset_args):
    cfg = default_config_file()
    update_cfg(cfg, dpl_selfmadehydrodataset_args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
