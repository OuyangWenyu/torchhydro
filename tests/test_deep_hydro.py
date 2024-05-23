"""
Author: Wenyu Ouyang
Date: 2024-05-01 10:34:15
LastEditTime: 2024-05-04 11:31:52
LastEditors: Wenyu Ouyang
Description: Unit tests for the DeepHydro class
FilePath: \torchhydro\tests\test_deep_hydro.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import os
import torch
from torch.utils.data import Dataset
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.trainers.train_logger import TrainLogger
import torch
import tempfile


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
        return 200

    def __len__(self):
        return self.ngrid * (self.nt - self.data_cfgs["forecast_history"] + 1)

    def __getitem__(self, idx):
        # Use the stored configurations to generate mock data
        # rho = self.data_cfgs["forecast_history"]
        # x = torch.randn(rho, self.data_cfgs["input_features"])
        # y = torch.randn(rho, self.data_cfgs["output_features"])
        mode = self.data_cfgs["model_mode"]
        if mode == "single":
            src1 = torch.rand(
                self.data_cfgs["forecast_history"],
                self.data_cfgs["input_features"] - 1,
            )
            src2 = torch.rand(
                self.data_cfgs["forecast_history"],
                self.data_cfgs["cnn_size"],
            )
        else:
            src1 = torch.rand(
                self.data_cfgs["forecast_history"],
                self.data_cfgs["input_features"],
            )
            src2 = torch.rand(
                self.data_cfgs["forecast_history"],
                self.data_cfgs["input_size_encoder2"]
                )
        src3 = torch.rand(1, self.data_cfgs["output_features"]) # start_token
        out = torch.rand(self.data_cfgs["forecast_length"], self.data_cfgs["output_features"]) 
        return [src1, src2, src3], out


@pytest.fixture()
def dummy_data_cfgs():
    test_path = "results/test_seq2seq_single2/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    return {
        "dataset": "MockDataset",
        "input_features": 4,
        "output_features": 1,
        # "t_range_valid": ["2010-01-01", "2010-12-31"],
        "t_range_valid": None,
        "test_path": test_path,
        "sampler": "KuaiSampler",
        # "sampler": "HydroSampler",
        "batch_size": 5,
        "forecast_history": 5,
        "forecast_length": 2,
        "warmup_length": 10,
        "cnn_size" : 120,
        "input_size_encoder2": 1,
        "model_mode": "single",
    }


def test_using_mock_dataset(dummy_data_cfgs):
    datasets_dict["MockDataset"] = MockDataset
    is_tra_val_te = True
    dataset_name = "MockDataset"

    dataset = datasets_dict[dataset_name](dummy_data_cfgs, is_tra_val_te)

    assert len(dataset) == 330
    sample_x, sample_y= dataset[0]
    print(sample_x[0].shape)
    print(sample_x[1].shape)
    print(sample_x[2].shape)
    print(sample_y.shape)
    # assert sample_x.shape == (dummy_data_cfgs['forecast_history'], dummy_data_cfgs["input_features"])
    # assert sample_y.shape == (dummy_data_cfgs['forecast_history'], dummy_data_cfgs["output_features"])

@pytest.fixture()
def dummy_train_cfgs(dummy_data_cfgs):
    return {
        "training_cfgs": {
            "early_stopping": False,
            "patience": 4,
            "epochs": 2,
            "start_epoch": 1,
            "which_first_tensor": "batch",
            "device": -1,  # Assuming CPU device
            "train_mode": True,
            "criterion": "RMSE",
            "optimizer": "Adam",
            "optim_params": {},
            "lr_scheduler": {"lr": 0.001},
            "batch_size": 5,
            "save_epoch": 1,
        },
        "data_cfgs": dummy_data_cfgs,
        "model_cfgs": {
            "model_type": "Normal",
            # "model_name": "CpuLSTM",
            "model_name": "Seq2Seq",
            "weight_path": None,
            "model_hyperparam": {
                # "n_input_features": 10,
                # "n_output_features": 1,
                # "n_hidden_states": 64,
                "input_size": 4,
                "output_size": 1,
                "hidden_size": 256,
                "forecast_length": 2,
                "cnn_size": 120,
                "model_mode": "single",
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
    model_filepath = dummy_train_cfgs["data_cfgs"]["test_path"]
    train_logger = TrainLogger(model_filepath, dummy_train_cfgs, opt)
    train_logger.plot_model_structure(deep_hydro.model)