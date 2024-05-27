"""
Author: Wenyu Ouyang
Date: 2023-09-25 08:21:27
LastEditTime: 2024-05-27 15:59:09
LastEditors: Wenyu Ouyang
Description: Some sampling class or functions
FilePath: \torchhydro\torchhydro\datasets\sampler.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from collections import defaultdict
import numpy as np
from torch.utils.data import RandomSampler, Sampler
from torchhydro.datasets.data_sets import BaseDataset
from typing import Iterator, Optional, Sized
import torch


class KuaiSampler(RandomSampler):
    def __init__(
        self,
        data_source,
        batch_size,
        warmup_length,
        rho_horizon,
        ngrid,
        nt,
    ):
        """a sampler from Kuai Fang's paper: https://doi.org/10.1002/2017GL075619
           He used a random pick-up that we don't need to iterate all samples.
           Then, we can train model more quickly

        Parameters
        ----------
        data_source : torch.utils.data.Dataset
            just a object of dataset class inherited from torch.utils.data.Dataset
        batch_size : int
            we need batch_size to calculate the number of samples in an epoch
        warmup_length : int
            warmup length, typically for physical hydrological models
        rho_horizon : int
            sequence length of a mini-batch, for encoder-decoder models, rho+horizon, for decoder-only models, horizon
        ngrid : int
            number of basins
        nt : int
            number of all periods
        """
        while batch_size * rho_horizon >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        batch_size = max(batch_size, 1)
        # 99% chance that all periods' data are used in an epoch
        n_iter_ep = int(
            np.ceil(
                np.log(0.01)
                / np.log(1 - batch_size * rho_horizon / ngrid / (nt - warmup_length))
            )
        )
        assert n_iter_ep >= 1
        # __len__ means the number of all samples, then, the number of loops in an epoch is __len__()/batch_size = n_iter_ep
        # hence we return n_iter_ep * batch_size
        num_samples = n_iter_ep * batch_size
        super(KuaiSampler, self).__init__(data_source, num_samples=num_samples)


class HydroSampler(Sampler[int]):
    """
    A custom sampler for hydrological modeling that iterates over a dataset in
    a way tailored for batches of hydrological data. It ensures that each batch
    contains data from a single randomly selected 'basin' out of several basins,
    with batches constructed to respect the specified batch size and the unique
    characteristics of hydrological datasets.

    Parameters:
    - data_source (Sized): The dataset to sample from, expected to have a `data_cfgs` attribute.
    - num_samples (Optional[int], default=None): The total number of samples to draw (optional).
    - generator: A PyTorch Generator object for random number generation (optional).

    The sampler divides the dataset by the number of basins, then iterates through
    each basin's range in shuffled order, ensuring non-overlapping, basin-specific
    batches suitable for models that predict hydrological outcomes.
    """

    data_source: Sized

    def __init__(
        self,
        data_source: Sized,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = self.data_source.data_cfgs["batch_size"]
        basin_number = len(self.data_source.data_cfgs["object_ids"])
        basin_range = len(self.data_source) // basin_number
        if n > basin_range:
            raise ValueError(
                f"batch_size should equal or less than basin_range={basin_range} "
            )

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # basin_list = torch.randperm(basin_number)
        # for select_basin in basin_list:
        #     x = torch.randperm(basin_range)
        #     for i in range(0, basin_range, n):
        #         yield from (x[i : i + n] + basin_range * select_basin.item()).tolist()
        x = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, n):
            yield from (x[i : i + n]).tolist()

    def __len__(self) -> int:
        return self.num_samples


def fl_sample_basin(dataset: BaseDataset):
    """
    Sample one basin data as a client from a dataset for federated learning

    Parameters
    ----------
    dataset
        dataset

    Returns
    -------
        dict of image index
    """
    lookup_table = dataset.lookup_table
    basins = dataset.basins
    # one basin is one user
    num_users = len(basins)
    # set group for basins
    basin_groups = defaultdict(list)
    for idx, (basin, date) in lookup_table.items():
        basin_groups[basin].append(idx)

    # one user is one basin
    user_basins = defaultdict(list)
    for i, basin in enumerate(basins):
        user_id = i % num_users
        user_basins[user_id].append(basin)

    # a lookup_table subset for each user
    user_lookup_tables = {}
    for user_id, basins in user_basins.items():
        user_lookup_table = {}
        for basin in basins:
            for idx in basin_groups[basin]:
                user_lookup_table[idx] = lookup_table[idx]
        user_lookup_tables[user_id] = user_lookup_table

    return user_lookup_tables


def fl_sample_region(dataset: BaseDataset):
    """
    Sample one region data as a client from a dataset for federated learning

    TODO: not finished

    """
    num_users = 10
    num_shards, num_imgs = 200, 250
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users
