"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2024-05-27 15:53:43
LastEditors: Wenyu Ouyang
Description: Test some functions for dataset
FilePath: \torchhydro\tests\test_sampler.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import hydrodataset as hds
from hydrodataset.caravan import Caravan
from torch.utils.data import Dataset

from torchhydro import SETTING
from torchhydro.datasets.sampler import KuaiSampler


class SimpleDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return idx


@pytest.fixture
def dataset():
    return SimpleDataset()


@pytest.fixture
def sampler(dataset):
    return KuaiSampler(
        dataset,
        batch_size=10,
        warmup_length=5,
        rho_horizon=20,
        ngrid=10,
        nt=365,
    )


def test_sampler_length(sampler):
    num_samples = 810
    assert len(sampler) == num_samples, f"Expected {num_samples} but got {len(sampler)}"


def test_sampler_within_range(sampler, dataset):
    for idx in sampler:
        assert 0 <= idx < len(dataset), f"Index {idx} out of bounds"


def test_sampler_randomness(sampler):
    first_epoch_samples = list(sampler)
    second_epoch_samples = list(sampler)

    # a simple check for two random samples
    # the probability of two samples being the same is very small
    assert (
        first_epoch_samples != second_epoch_samples
    ), "Expected different samples for different epochs"


def test_cache_file():
    """
    Test whether the cache file is generated correctly
    """
    camels_dir = os.path.join(
        SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
    )
    camels_us = hds.Camels(camels_dir)
    camels_us.cache_xrdataset()


def test_cache_caravan():
    """
    Test whether the cache file is generated correctly
    """
    caravan = Caravan(
        os.path.join(SETTING["local_data_path"]["datasets-origin"], "caravan")
    )
    caravan.cache_xrdataset()
