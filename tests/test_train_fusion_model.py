
from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_train_evaluate(fusion_lstm_args, config_data):
    update_cfg(config_data, fusion_lstm_args)
    train_and_evaluate(config_data)
