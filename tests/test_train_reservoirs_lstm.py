import pytest

from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


# The reservoir_lstm_args / reservoir_regulation_lstm_args / reservoir_inflow_lstm_args
# fixtures referenced below were never defined in tests/conftest.py (introduced
# by 612767f but the fixtures were forgotten). Reconstruct the reservoir data
# source + model config to re-enable these data-dependent training tests.
pytestmark = pytest.mark.skip(
    reason="reservoir_lstm_args fixtures were never defined in conftest; "
    "needs reservoir data source + model config to be reconstructed"
)


@pytest.mark.requires_data
def test_train_evaluate(reservoir_lstm_args, config_data):
    update_cfg(config_data, reservoir_lstm_args)
    train_and_evaluate(config_data)


@pytest.mark.requires_data
def test_train_evaluate_regulation(reservoir_regulation_lstm_args, config_data):
    update_cfg(config_data, reservoir_regulation_lstm_args)
    train_and_evaluate(config_data)


@pytest.mark.requires_data
def test_train_evaluate_inflow(reservoir_inflow_lstm_args, config_data):
    update_cfg(config_data, reservoir_inflow_lstm_args)
    train_and_evaluate(config_data)
