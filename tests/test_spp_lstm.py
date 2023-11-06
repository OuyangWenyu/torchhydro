import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture()
def config():
    project_name = "test_spp_lstm/ex1"
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[-1],
        model_name="SPPLSTM",
        model_hyperparam={
            "seq_length": 8,
            "forecast_length": 3,
            "n_output": 1,
            "n_hidden_states": 80,
        },
        gage_id=[
            "05584500",
            "01544500",
            "01423000",
        ],
        # batch_size有一些限制，不能超过一个流域用于训练的item个数，比如1个流域只有6个item,batch_size需小于6
        batch_size=4,
        var_t=["tp"],  # precipitationCal
        var_out=["waterlevel"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=1,
        save_epoch=1,
        te=1,
        train_period=["2017-01-10", "2017-01-10"],
        test_period=["2017-01-12", "2017-01-12"],
        # valid_period=["2017-01-13", "2017-01-13"],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={1: 5e-4, 2: 1e-4, 3: 1e-5},
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_spp_lstm(config):
    train_and_evaluate(config)
