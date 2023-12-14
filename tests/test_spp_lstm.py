import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex6"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[1],
        model_name="SPPLSTM",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 80,
        },
        gage_id=[
            '21401550'
            # "05584500",
            # "01544500",
            # "01423000",
        ],
        batch_size=256,
        var_t=["tp"],
        var_out=["streamflow"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=50,
        save_epoch=1,
        te=50,
        # train_period=["2020-07-10", "2020-08-31"],
        # test_period=["2020-09-01", "2020-09-15"],
        # valid_period=["2020-09-16", "2020-09-29"],
        train_period=["2016-07-20", "2017-07-20"],
        test_period=["2016-07-20", "2017-07-20"],
        valid_period=["2016-07-20", "2017-07-20"],
        loss_func="RMSESum",
        opt="Adam",
        # explainer="shap",
        lr_scheduler={1: 1e-4, 2: 5e-5, 3: 1e-5},
        which_first_tensor="sequence",
        is_tensorboard=False,
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    train_and_evaluate(config)

# test_spp_lstm(config)