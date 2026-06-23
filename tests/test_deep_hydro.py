"""
Author: Wenyu Ouyang
Date: 2024-05-01 10:34:15
LastEditTime: 2025-11-07 08:36:12
LastEditors: Wenyu Ouyang
Description: Unit tests for the DeepHydro class
FilePath: \torchhydro\tests\test_deep_hydro.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import os
import torch
from torch.utils.data import Dataset
from torchhydro.datasets.sampler import BasinBatchSampler, KuaiSampler
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.trainers.train_logger import TrainLogger
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DistributedSampler


# Mock dataset class using random data
class MockDataset(Dataset):
    def __init__(self, cfgs, is_tra_val_te):
        super(MockDataset, self).__init__()
        self.data_cfgs = cfgs["data_cfgs"]
        self.training_cfgs = cfgs["training_cfgs"]

    @property
    def ngrid(self):
        return 10

    @property
    def nt(self):
        return 100

    @property
    def rho(self):
        return self.training_cfgs.get("hindcast_length", 0)

    @property
    def warmup_length(self):
        return self.training_cfgs.get("warmup_length", 0)

    @property
    def horizon(self):
        return self.training_cfgs.get("forecast_length", 0)

    def __len__(self):
        return self.ngrid * (self.nt - self.horizon + 1)

    def __getitem__(self, idx):
        x = torch.randn(self.rho, self.data_cfgs["input_features"])
        # The output shape must match the label shape for loss calculation
        y = torch.randn(self.rho, self.data_cfgs.get("output_features", 1))
        return x, y


@pytest.fixture()
def dummy_data_cfgs():
    test_path = "results/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    return {
        "dataset": "MockDataset",
        "model_mode": "normal",
        "input_features": 10,
        "output_features": 1,
        "t_range_valid": None,
        "case_dir": test_path,
        "sampler": "KuaiSampler",
        "input_size_encoder2": 1,
        "object_ids": ["02051500", "21401550"],
    }


@pytest.fixture()
def dummy_training_cfgs():
    return {
        "batch_size": 5,
        "hindcast_length": 30,
        "forecast_length": 1,
        "warmup_length": 0,
    }


@pytest.fixture()
def dummy_train_cfgs(dummy_data_cfgs, dummy_training_cfgs):
    # Combine data and training cfgs for a full config
    # This more closely resembles the final config structure
    cfgs = {
        "training_cfgs": {
            "early_stopping": False,
            "patience": 4,
            "epochs": 1,
            "start_epoch": 1,
            "which_first_tensor": "batch",
            "device": [-1],  # Assuming CPU device
            "train_mode": True,
            "criterion": "RMSE",
            "optimizer": "Adam",
            "optim_params": {"lr": 0.01},
            "lr_scheduler": {0: 0.5, 10: 0.1},
            "batch_size": 5,
            "save_epoch": 1,
            "hindcast_length": 30,
            "forecast_length": 1,
            "warmup_length": 0,
            "variable_length_cfgs": {"use_variable_length": False},
        },
        "data_cfgs": dummy_data_cfgs,
        "model_cfgs": {
            "model_type": "Normal",
            "model_name": "CpuLSTM",
            "weight_path": None,
            "continue_train": False,
            "model_hyperparam": {
                "n_input_features": 10,
                "n_output_features": 1,
                "n_hidden_states": 64,
            },
        },
        "evaluation_cfgs": {
            "model_loader": {"load_way": "specified", "test_epoch": 1},
            "metrics": ["NSE", "RMSE", "KGE"],
            "fill_nan": "no",
        },
    }
    # Add training cfgs to data cfgs for MockDataset compatibility
    cfgs["data_cfgs"]["training_cfgs"] = cfgs["training_cfgs"]
    return cfgs


@pytest.fixture()
def deep_hydro(dummy_train_cfgs):
    datasets_dict["MockDataset"] = MockDataset
    return DeepHydro(dummy_train_cfgs)


def test_model_train(deep_hydro):
    deep_hydro.model_train()
    assert deep_hydro.model.state_dict() is not None


@pytest.mark.skip(
    reason="Plotting logic is complex and incompatible with the mock model's forward pass."
)
def test_plot_model_structure(deep_hydro, dummy_train_cfgs):
    opt = torch.optim.SGD(deep_hydro.model.parameters(), lr=0.01)
    model_filepath = dummy_train_cfgs["data_cfgs"]["case_dir"]
    train_logger = TrainLogger(model_filepath, dummy_train_cfgs, opt)
    train_logger.plot_model_structure(deep_hydro.model)


def test_get_scheduler_lambda_lr(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {"lr": 0.001}
    opt = Adam(deep_hydro.model.parameters())
    scheduler = deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)
    assert isinstance(scheduler, LambdaLR)


def test_get_scheduler_lambda_lr_with_epochs(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {0: 1.0, 10: 0.1}
    opt = Adam(deep_hydro.model.parameters())
    scheduler = deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)
    assert isinstance(scheduler, LambdaLR)


def test_get_scheduler_exponential_lr(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {"lr_factor": 0.9}
    opt = Adam(deep_hydro.model.parameters())
    scheduler = deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)
    assert isinstance(scheduler, ExponentialLR)


def test_get_scheduler_reduce_lr_on_plateau(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {
        "lr_factor": 0.9,
        "lr_patience": 5,
    }
    opt = Adam(deep_hydro.model.parameters())
    scheduler = deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)
    assert isinstance(scheduler, ReduceLROnPlateau)


def test_get_scheduler_invalid_config(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {"invalid_key": 0.9}
    opt = Adam(deep_hydro.model.parameters())
    with pytest.raises(ValueError, match="Invalid lr_scheduler configuration"):
        deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)


def test_get_sampler_basin_batch_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "BasinBatchSampler"
    sampler = deep_hydro._get_sampler(
        dummy_train_cfgs["data_cfgs"],
        dummy_train_cfgs["training_cfgs"],
        deep_hydro.traindataset,
    )
    assert isinstance(sampler, BasinBatchSampler)


def test_get_sampler_kuai_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "KuaiSampler"
    sampler = deep_hydro._get_sampler(
        dummy_train_cfgs["data_cfgs"],
        dummy_train_cfgs["training_cfgs"],
        deep_hydro.traindataset,
    )
    assert isinstance(sampler, KuaiSampler)


def test_get_sampler_invalid_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "InvalidSampler"
    with pytest.raises(
        NotImplementedError, match="Sampler InvalidSampler not implemented yet"
    ):
        deep_hydro._get_sampler(
            dummy_train_cfgs["data_cfgs"],
            dummy_train_cfgs["training_cfgs"],
            deep_hydro.traindataset,
        )
