import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from torchhydro.trainers.train_utils import torch_single_train


class _RegressionDataset(Dataset):
    training_cfgs = {"variable_length_cfgs": None}

    def __init__(self, nan_target=False):
        self.x = torch.arange(1, 9, dtype=torch.float32).unsqueeze(-1)
        self.y = 2 * self.x
        if nan_target:
            self.y[-1] = torch.nan

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def _train_once(loss_check_interval):
    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(_RegressionDataset(), batch_size=2, shuffle=False)
    loss, iterations = torch_single_train(
        model,
        optimizer,
        torch.nn.MSELoss(),
        loader,
        device=torch.device("cpu"),
        which_first_tensor="batch",
        mixed_precision="off",
        loss_check_interval=loss_check_interval,
    )
    return loss, iterations, model.state_dict()


@pytest.mark.parametrize("loss_check_interval", [0, 2, 100])
def test_deferred_loss_checks_preserve_training_result(loss_check_interval):
    expected_loss, expected_iterations, expected_state = _train_once(1)
    actual_loss, actual_iterations, actual_state = _train_once(loss_check_interval)

    assert actual_iterations == expected_iterations
    assert actual_loss == pytest.approx(expected_loss, rel=1e-6, abs=1e-6)
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name])


@pytest.mark.parametrize("loss_check_interval", [0, 2, 100])
def test_deferred_loss_checks_still_reject_nan(loss_check_interval):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(
        _RegressionDataset(nan_target=True), batch_size=2, shuffle=False
    )

    with pytest.raises(ValueError, match="nan loss detected"):
        torch_single_train(
            model,
            optimizer,
            torch.nn.MSELoss(),
            loader,
            device=torch.device("cpu"),
            which_first_tensor="batch",
            mixed_precision="off",
            loss_check_interval=loss_check_interval,
        )


def test_negative_loss_check_interval_is_rejected():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(_RegressionDataset(), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        torch_single_train(
            model,
            optimizer,
            torch.nn.MSELoss(),
            loader,
            device=torch.device("cpu"),
            which_first_tensor="batch",
            loss_check_interval=-1,
        )
