import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex11"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[2],
        model_name="SPPLSTM2",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 80,
        },
        gage_id=[
            "21401550"
        ],
        batch_size=256,
        var_t=["tp"],
        var_out=["streamflow"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=100,
        save_epoch=1,
        te=100,
        train_period=["2017-01-01", "2020-12-31"],
        test_period=["2021-01-01", "2021-12-31"],
        valid_period=["2021-01-01", "2021-12-31"],
        # train_period=["2017-07-26", "2017-07-26"],
        # test_period=["2017-07-22", "2017-07-22"],
        # valid_period=["2017-07-22", "2017-07-22"],
        # train_period=["2018-07-01", "2018-09-30"],
        # test_period=["2018-07-01", "2018-09-30"],
        # valid_period=["2018-07-01", "2018-09-30"],
        # train_period=["2017-07-20", "2017-07-21"],
        # test_period=["2017-07-22", "2017-07-22"],
        # valid_period=["2017-07-22", "2017-07-22"],
        loss_func="RMSESum",
        opt="Adam",
        # explainer="shap",
        lr_scheduler={1: 1e-2},
        which_first_tensor="sequence",
        early_stopping=True,
        patience=10,
        rolling=False,
        # ensemble=True,
        # ensemble_items={
        #     "kfold": 5,
        #     "batch_sizes": [512],
        # },
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    train_and_evaluate(config)
    # ensemble_train_and_evaluate(config)
