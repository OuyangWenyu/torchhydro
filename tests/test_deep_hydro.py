"""
Author: Wenyu Ouyang
Date: 2024-05-01 10:34:15
LastEditTime: 2024-11-04 18:26:24
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
    def __init__(self, data_cfgs, is_tra_val_te):
        super(MockDataset, self).__init__()
        self.data_cfgs = data_cfgs  # Store the passed configuration for later use
        # Simulate other configuration and setup steps

    @property
    def ngrid(self):
        return 10

    @property
    def nt(self):
        return 100

    def __len__(self):
        return self.ngrid * (self.nt - self.data_cfgs["forecast_length"] + 1)

    def __getitem__(self, idx):
        # Use the stored configurations to generate mock data
        rho = self.data_cfgs["forecast_length"]
        x = torch.randn(rho, self.data_cfgs["input_features"])
        y = torch.randn(rho, self.data_cfgs["output_features"])
        return x, y


@pytest.fixture()
def dummy_data_cfgs():
    test_path = "results/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    return {
        "dataset": "MockDataset",
        "input_features": 10,
        "output_features": 1,
        # "t_range_valid": ["2010-01-01", "2010-12-31"],
        "t_range_valid": None,
        "case_dir": test_path,
        "sampler": "KuaiSampler",
        "batch_size": 5,
        "hindcast_length": 0,
        "forecast_length": 30,
        "warmup_length": 0,
        "object_ids": ["02051500", "21401550"],
    }


def test_using_mock_dataset(dummy_data_cfgs):
    datasets_dict["MockDataset"] = MockDataset
    is_tra_val_te = True
    dataset_name = "MockDataset"

    dataset = datasets_dict[dataset_name](dummy_data_cfgs, is_tra_val_te)

    assert len(dataset) == 710
    sample_x, sample_y = dataset[0]
    print(sample_x[0].shape)
    print(sample_x[1].shape)
    print(sample_x[2].shape)
    print(sample_y.shape)
    assert sample_x.shape == (
        dummy_data_cfgs["forecast_length"],
        dummy_data_cfgs["input_features"],
    )
    assert sample_y.shape == (
        dummy_data_cfgs["forecast_length"],
        dummy_data_cfgs["output_features"],
    )


@pytest.fixture()
def dummy_train_cfgs(dummy_data_cfgs):
    return {
        "training_cfgs": {
            "early_stopping": False,
            "patience": 4,
            "epochs": 12,
            "start_epoch": 1,
            "which_first_tensor": "batch",
            "device": [-1],  # Assuming CPU device
            "train_mode": True,
            "criterion": "RMSE",
            "optimizer": "Adam",
            "optim_params": {"lr": 0.01},
            # "optim_params": {},
            "lr_scheduler": {0: 0.5, 10: 0.1},
            "batch_size": 5,
            "save_epoch": 1,
        },
        "data_cfgs": dummy_data_cfgs,
        "model_cfgs": {
            "model_type": "Normal",
            "model_name": "CpuLSTM",
            "weight_path": None,
            "model_hyperparam": {
                "n_input_features": 10,
                "n_output_features": 1,
                "n_hidden_states": 64,
            },
        },
        "evaluation_cfgs": {
            "model_loader": {"load_way": "specified", "test_epoch": 2},
            "metrics": ["NSE", "RMSE", "KGE"],
            "fill_nan": "no",
        },
    }


@pytest.fixture()
def deep_hydro(dummy_train_cfgs):
    datasets_dict["MockDataset"] = MockDataset
    # Create an instance of DeepHydro with dummy configurations
    return DeepHydro(dummy_train_cfgs)


def test_model_train(deep_hydro):
    # Call the model_train method and check if it runs without errors
    deep_hydro.model_train()

    # Add assertions to check the expected behavior of the method
    assert deep_hydro.model.state_dict() is not None


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


# add a test func for LambdaLR  show me the lr change in each epoch
def test_get_scheduler_lambda_lr_with_epochs_show_lr(deep_hydro, dummy_train_cfgs):
    # NOTE: for scheduler, the start epoch is 0! but scheduler.step is still after each epoch
    dummy_train_cfgs["training_cfgs"]["lr_scheduler"] = {1: 0.5, 10: 0.1}
    opt = Adam(deep_hydro.model.parameters())
    scheduler = deep_hydro._get_scheduler(dummy_train_cfgs["training_cfgs"], opt)
    for epoch in range(1, 15):
        # We start from epoch 1 not 0 to make it easier to understand for human
        # NOTE: the scheduler.step() is called at the end of each epoch
        # so that at the first epoch, the lr is still the initial lr,
        # one has to set initial lr in the optimizer directly for the first epoch
        print(f"epoch:{epoch}, lr:{opt.param_groups[0]['lr']}")
        scheduler.step()
    assert isinstance(scheduler, LambdaLR)


def test_get_sampler_basin_batch_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "BasinBatchSampler"
    sampler = deep_hydro._get_sampler(
        dummy_train_cfgs["data_cfgs"], deep_hydro.traindataset
    )
    assert isinstance(sampler, BasinBatchSampler)


def test_get_sampler_kuai_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "KuaiSampler"
    sampler = deep_hydro._get_sampler(
        dummy_train_cfgs["data_cfgs"], deep_hydro.traindataset
    )
    assert isinstance(sampler, KuaiSampler)


def test_get_sampler_invalid_sampler(deep_hydro, dummy_train_cfgs):
    dummy_train_cfgs["data_cfgs"]["sampler"] = "InvalidSampler"
    with pytest.raises(
        NotImplementedError, match="Sampler InvalidSampler not implemented yet"
    ):
        deep_hydro._get_sampler(dummy_train_cfgs["data_cfgs"], deep_hydro.traindataset)
