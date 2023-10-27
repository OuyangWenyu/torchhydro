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
            "n_output": 3,
            "n_hidden_states": 80,
        },
        gage_id=[
            "05584500",
            # "01544500",
        ],
        batch_size=16,
        var_t=["precipitationCal"],
        var_out=["waterlevel"],
        dataset="GPM_GFS_Dataset",
        sampler="KuaiSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=80,
        save_epoch=1,
        te=80,
        train_period=["2017-01-03", "2017-01-03"],
        test_period=["2017-01-03", "2017-01-03"],
        valid_period=["2017-01-03", "2017-01-03"],
        loss_func="RMSESum",
        opt="Adam",
        # lr_scheduler={1: 1e-2, 2: -5e-3, 3: 1e-3},
        lr_scheduler={1: 5e-4, 2: 1e-4, 3: 1e-5},
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data

def test_spp_lstm(config):
    train_and_evaluate(config)
