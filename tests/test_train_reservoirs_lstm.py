from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.trainers.trainer import evaluate_model


def test_train_evaluate(reservoir_lstm_args, config_data):
    update_cfg(config_data, reservoir_lstm_args)
    train_and_evaluate(config_data)


def test_train_evaluate_regulation(reservoir_regulation_lstm_args, config_data):
    update_cfg(config_data, reservoir_regulation_lstm_args)
    train_and_evaluate(config_data)


def test_train_evaluate_inflow(reservoir_inflow_lstm_args, config_data):
    update_cfg(config_data, reservoir_inflow_lstm_args)
    train_and_evaluate(config_data)
