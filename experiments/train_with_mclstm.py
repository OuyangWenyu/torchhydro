import os
import logging
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


def create_mclstm_config():
    config_data = default_config_file()

    args = cmd(
        sub=os.path.join("results", "mclstm_camels"),
        source_cfgs={"dataset_id": "camels_us"},
        ctx=[-1],
        model_name="MCLSTM",
        model_hyperparam={
            "input_size": 2,
            "output_size": 1,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
            "initial_forget_bias": 0.0,
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
        train_epoch=2,
        save_epoch=1,
        hindcast_length=0,
        forecast_length=20,
        min_time_unit="D",
        min_time_interval="1",
        var_t=["prcp", "pet"],
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        sampler="KuaiSampler",
        scaler="DapengScaler",
        model_loader={"load_way": "specified", "test_epoch": 2},
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
        which_first_tensor="sequence",
    )

    update_cfg(config_data, args)
    return config_data


def train_mclstm():
    cfg = create_mclstm_config()
    train_and_evaluate(cfg)


if __name__ == "__main__":
    train_mclstm()
