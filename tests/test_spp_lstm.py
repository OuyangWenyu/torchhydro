import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex11_2021"
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
            "n_hidden_states": 60,
            "dropout": 0.2,
        },
        gage_id=["21401550"],
        batch_size=256,
        var_t=["tp"],
        var_out=["streamflow"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=10,
        save_epoch=1,
        te=10,
        train_period=[
            {"start": "2017-07-01", "end": "2017-09-30"},
            {"start": "2018-07-01", "end": "2018-09-30"},
            {"start": "2019-07-01", "end": "2019-09-30"},
            {"start": "2020-07-01", "end": "2020-09-30"},
        ],
        test_period=[{"start": "2021-07-01", "end": "2021-09-30"}],
        valid_period=[
            {"start": "2021-07-01", "end": "2021-09-30"},
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={1: 1e-2, 3: 1e-3},
        lr_factor=0.5,
        lr_patience=2,
        weight_decay=1e-5,  # L2正则化衰减权重
        lr_val_loss=False,  # False则用NSE作为指标，而不是val loss,来更新lr、model、早退
        which_first_tensor="sequence",
        early_stopping=True,
        patience=5,
        rolling=False,  # evaluate 不采用滚动预测
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
