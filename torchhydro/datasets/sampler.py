"""
Author: Wenyu Ouyang
Date: 2023-09-25 08:21:27
LastEditTime: 2023-10-05 15:50:27
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
        rho,
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
        rho : int
            sequence length of a mini-batch
        ngrid : int
            number of basins
        nt : int
            number of all periods
        """
        while batch_size * rho >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        batch_size = max(batch_size, 1)
        # 99% chance that all periods' data are used in an epoch
        n_iter_ep = int(
            np.ceil(
                np.log(0.01)
                / np.log(1 - batch_size * rho / ngrid / (nt - warmup_length))
            )
        )
        assert n_iter_ep >= 1
        # __len__ means the number of all samples, then, the number of loops in an epoch is __len__()/batch_size = n_iter_ep
        # hence we return n_iter_ep * batch_size
        num_samples = n_iter_ep * batch_size
        super(KuaiSampler, self).__init__(data_source, num_samples=num_samples)


class GPM_GFS_Sampler(Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        # if self._num_samples is None:
        return len(self.data_source)
        # return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = self.data_source.data_cfgs["batch_size"]
        basin_number = len(self.data_source.data_cfgs["object_ids"])
        basin_range = len(self.data_source) / basin_number
        if n > basin_range:
            raise ValueError(
                "batch_size should equal or less than basin_range={} ".format(
                    basin_range
                )
            )

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:  # 暂不可用
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                select_basin = torch.randperm(
                    basin_number, generator=generator
                ).tolist()[0]
                x = torch.randperm(int(basin_range), generator=generator)
                yield from (x[:n] + int(basin_range) * (select_basin)).tolist()

    def __len__(self) -> int:
        return self.num_samples


class WuSampler(GPM_GFS_Sampler):
    def __init__(
        self,
        data_source,
        batch_size,
        warmup_length,
        rho,
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
        rho : int
            sequence length of a mini-batch
        ngrid : int
            number of basins
        nt : int
            number of all periods
        """
        while batch_size * rho >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        batch_size = max(batch_size, 1)
        # 99% chance that all periods' data are used in an epoch
        n_iter_ep = int(
            np.ceil(
                np.log(0.01)
                / np.log(1 - batch_size * rho / ngrid / (nt - warmup_length))
            )
        )
        assert n_iter_ep >= 1
        # __len__ means the number of all samples, then, the number of loops in an epoch is __len__()/batch_size = n_iter_ep
        # hence we return n_iter_ep * batch_size
        num_samples = n_iter_ep * batch_size
        super(WuSampler, self).__init__(data_source, num_samples=num_samples)


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
