import math

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


def _make_training_objects(nan_target=False):
    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(
        _RegressionDataset(nan_target=nan_target), batch_size=2, shuffle=False
    )
    return model, optimizer, loader


def test_training_returns_finite_epoch_loss_and_iteration_count():
    model, optimizer, loader = _make_training_objects()

    loss, iterations = torch_single_train(
        model,
        optimizer,
        torch.nn.MSELoss(),
        loader,
        device=torch.device("cpu"),
        which_first_tensor="batch",
        mixed_precision="off",
    )

    assert iterations == len(loader)
    assert math.isfinite(loss)
    assert loss > 0


def test_training_rejects_nan_loss_before_parameter_update():
    model, optimizer, loader = _make_training_objects(nan_target=True)
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }

    with pytest.raises(ValueError, match="nan loss detected"):
        torch_single_train(
            model,
            optimizer,
            torch.nn.MSELoss(),
            loader,
            device=torch.device("cpu"),
            which_first_tensor="batch",
            mixed_precision="off",
        )

    # The first three finite batches update the model, while the final NaN
    # batch is rejected before backward/optimizer.step.
    assert any(
        not torch.equal(value, parameters_before[name])
        for name, value in model.named_parameters()
    )
    for value in model.parameters():
        assert torch.isfinite(value).all()
