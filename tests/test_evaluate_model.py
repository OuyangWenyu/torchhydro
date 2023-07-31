import os
import hydrodataset as hds
import pytest
from hydroutils.hydro_file import get_lastest_file_in_a_dir
from hydroutils.hydro_plot import plot_ts
from torchhydro.datasets.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.evaluator import evaluate_model
from torchhydro.trainers.time_model import PyTorchForecast
from torchhydro.trainers.trainer import set_random_seed


@pytest.fixture()
def config_data():
    project_name = "test_camels/exp2"
    weight_dir = os.path.join(
        os.getcwd(),
        "results",
        "test_camels",
        "exp1",
    )
    weight_path = get_lastest_file_in_a_dir(weight_dir)
    args = cmd(
        sub=project_name,
        download=0,
        source_path=os.path.join(hds.ROOT_DIR, "camels", "camels_us"),
        source_region="US",
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
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
        batch_size=5,
        rho=20,  # batch_size=100, rho=365,
        var_t=["dayl", "prcp", "srad", "tmax", "tmin", "vp"],
        var_out=["streamflow"],
        data_loader="KuaiDataset",
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
        },
        train_epoch=5,
        save_epoch=1,
        te=5,
        train_period=["2000-10-01", "2001-10-01"],
        test_period=["2001-10-01", "2002-10-01"],
        loss_func="RMSESum",
        opt="Adadelta",
        which_first_tensor="sequence",
        weight_path=weight_path,
    )
    config_data = default_config_file()
    update_cfg(config_data, args)
    return config_data


def test_evaluate_model(config_data):
    random_seed = config_data["training_params"]["random_seed"]
    set_random_seed(random_seed)
    data_params = config_data["data_params"]
    data_source_name = data_params["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_params["data_path"], data_params["download"]
    )
    model = PyTorchForecast(
        config_data["model_params"]["model_name"], data_source, config_data
    )
    eval_log, preds_xr, obss_xr = evaluate_model(model)
    print(eval_log)
    plot_ts(
        [preds_xr["time"].values, obss_xr["time"].values],
        [
            preds_xr["streamflow"].sel(basin="01013500").values,
            obss_xr["streamflow"].sel(basin="01013500").values,
        ],
        leg_lst=["pred", "obs"],
        fig_size=(6, 4),
        xlabel="Date",
        ylabel="Streamflow",
    )
