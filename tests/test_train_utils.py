import os
import pytest
from torchhydro.trainers.train_utils import read_pth_from_model_loader


def test_read_pth_from_model_loader_specified():
    model_loader = {"load_way": "specified", "test_epoch": 5}
    model_pth_dir = "/path/to/models"
    expected_path = os.path.join(model_pth_dir, "model_Ep5.pth")
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_best():
    model_loader = {"load_way": "best"}
    model_pth_dir = "/path/to/models"
    expected_path = os.path.join(model_pth_dir, "best_model.pth")
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_latest(mocker):
    model_loader = {"load_way": "latest"}
    model_pth_dir = "/path/to/models"
    latest_file = "latest_model.pth"
    mocker.patch(
        "torchhydro.trainers.train_utils.get_lastest_file_in_a_dir",
        return_value=latest_file,
    )
    expected_path = latest_file
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_pth():
    model_loader = {"load_way": "pth", "pth_path": "/path/to/models/custom_model.pth"}
    model_pth_dir = "/path/to/models"
    expected_path = "/path/to/models/custom_model.pth"
    assert read_pth_from_model_loader(model_loader, model_pth_dir) == expected_path


def test_read_pth_from_model_loader_invalid():
    model_loader = {"load_way": "invalid"}
    model_pth_dir = "/path/to/models"
    with pytest.raises(ValueError, match="Invalid load_way"):
        read_pth_from_model_loader(model_loader, model_pth_dir)
