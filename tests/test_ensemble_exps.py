import pytest
import os
from hydroutils import hydro_file

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import ensemble_train_and_evaluate


@pytest.fixture()
def var_c_target():
    return [
        "p_mean",
        "pet_mean",
        "Area",
        "geol_class_1st",
        "elev",
        "SNDPPT",
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


@pytest.fixture()
def gage_id():
    return [
        "61561",
    ]


def test_run_lstm_cross_val(var_c_target, var_t_target, gage_id):
    config = default_config_file()
    project_name = "test_camels/expcccv61561"
    kfold = 2
    train_period = ["2018-10-01", "2021-10-01"]
    valid_period = ["2015-10-01", "2018-10-01"]
    args = cmd(
        sub=project_name,
        source="SelfMadeCAMELS",
        source_path=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "camels_cc"
        ),
        ctx=[0],
        model_name="KuaiLSTM",
        model_hyperparam={
            "n_input_features": len(var_c_target) + len(var_t_target),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        # opt_param=opt_param,
        loss_func="RMSESum",
        train_period=train_period,
        test_period=valid_period,
        batch_size=20,
        forecast_history=0,
        forecast_length=365,
        scaler="DapengScaler",
        dataset="StreamflowDataset",
        continue_train=True,
        warmup_length=0,
        train_epoch=10,
        te=10,
        var_t=var_t_target,
        var_t_type="era5land",
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=gage_id,
        ensemble=True,
        ensemble_items={
            "kfold": kfold,
            "batch_sizes": [20, 50],
        },
    )
    update_cfg(config, args)
    ensemble_train_and_evaluate(config)
    print("All processes are finished!")


def test_run_cross_val_tlcamelsus2cc(
    var_c_source, var_c_target, var_t_source, var_t_target, gage_id
):
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = hydro_file.get_lastest_file_in_a_dir(weight_dir)
    project_name = "test_camels/exptl4cccv61561"
    train_period = ["2018-10-01", "2021-10-01"]
    valid_period = ["2015-10-01", "2018-10-01"]
    args = cmd(
        sub=project_name,
        source="SelfMadeCAMELS",
        # cc means China continent
        source_path=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "camels_cc"
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
        train_period=train_period,
        test_period=valid_period,
        scaler="DapengScaler",
        # sampler="KuaiSampler",
        dataset="StreamflowDataset",
        weight_path=weight_path,
        weight_path_add={
            "freeze_params": ["lstm.b_hh", "lstm.b_ih", "lstm.w_hh", "lstm.w_ih"]
        },
        continue_train=True,
        train_epoch=10,
        te=10,
        var_t=var_t_target,
        var_c=var_c_target,
        var_out=["streamflow"],
        gage_id=gage_id,
        ensemble=True,
        ensemble_items={
            "kfold": 2,
            "batch_sizes": [20, 50],
        },
    )
    cfg = default_config_file()
    update_cfg(cfg, args)
    ensemble_train_and_evaluate(cfg)
    print("All processes are finished!")
