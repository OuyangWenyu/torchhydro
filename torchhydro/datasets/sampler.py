"""
Author: Wenyu Ouyang
Date: 2023-09-25 08:21:27
LastEditTime: 2024-11-04 18:16:08
LastEditors: Wenyu Ouyang
Description: Some sampling class or functions
FilePath: \torchhydro\torchhydro\datasets\sampler.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from collections import defaultdict
import numpy as np
from torch.utils.data import RandomSampler, Sampler
from torchhydro.datasets.data_sets import BaseDataset
from typing import Iterator, Optional
import torch
import random


class KuaiSampler(RandomSampler):
    def __init__(
        self,
        dataset,
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
        dataset : torch.utils.data.Dataset
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
        super(KuaiSampler, self).__init__(dataset, num_samples=num_samples)


class BasinBatchSampler(Sampler[int]):
    """
    A custom sampler for hydrological modeling that iterates over a dataset in
    a way tailored for batches of hydrological data. It ensures that each batch
    contains data from a single randomly selected 'basin' out of several basins,
    with batches constructed to respect the specified batch size and the unique
    characteristics of hydrological datasets.
    TODO: made by Xinzhuo Wu, maybe need to be tested more

    Parameters
    ----------
    dataset : BaseDataset
        The dataset to sample from, expected to have a `data_cfgs` attribute.
    num_samples : Optional[int], default=None
        The total number of samples to draw (optional).
    generator : Optional[torch.Generator]
        A PyTorch Generator object for random number generation (optional).

    The sampler divides the dataset by the number of basins, then iterates through
    each basin's range in shuffled order, ensuring non-overlapping, basin-specific
    batches suitable for models that predict hydrological outcomes.
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.dataset = dataset
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        n = self.dataset.training_cfgs["batch_size"]
        basin_number = len(self.dataset.data_cfgs["object_ids"])
        basin_range = len(self.dataset) // basin_number
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


class WindowLenBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, balance_strategy='equal'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_lengths = list(dataset.lookup_tables_by_length.keys())
        self.indices_by_window_len = {
            window_len: [
                i for i, (w, _) in enumerate(dataset.lookup_table.values()) if w == window_len
            ] for window_len in self.window_lengths
        }
        self.balance_strategy = balance_strategy  # 'equal' 或 'proportional'
    
    def __iter__(self):
        # 修正：返回批次索引列表的迭代器，而不是单个索引的迭代器
        batches = []
        
        # 确定每个窗口长度应该提供多少批次
        if self.balance_strategy == 'equal':
            # 每个窗口长度提供相同数量的批次
            min_batches = min(len(indices) // self.batch_size for indices in self.indices_by_window_len.values())
            batches_per_window = {wl: min_batches for wl in self.window_lengths}
        else:  # 'proportional'
            # 按比例分配批次
            total_samples = sum(len(indices) for indices in self.indices_by_window_len.values())
            total_full_batches = sum(len(indices) // self.batch_size for indices in self.indices_by_window_len.values())
            batches_per_window = {}
            for wl in self.window_lengths:
                samples = len(self.indices_by_window_len[wl])
                batches_per_window[wl] = max(1, int(samples / total_samples * total_full_batches))
        
        # 打乱窗口长度顺序
        import random
        window_lengths = random.sample(self.window_lengths, len(self.window_lengths))
        
        # 为每个窗口长度创建批次
        for window_len in window_lengths:
            window_indices = self.indices_by_window_len[window_len].copy()
            random.shuffle(window_indices)  # 打乱索引
            
            # 限制批次数量以实现平衡
            max_batches = batches_per_window[window_len]
            batch_count = 0
            
            for i in range(0, len(window_indices), self.batch_size):
                if batch_count >= max_batches:
                    break
                    
                batch_indices = window_indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:  # 只保留完整批次
                    batches.append(batch_indices)
                    batch_count += 1
        
        # 最后再打乱所有批次的顺序
        random.shuffle(batches)
        
        # 返回批次列表的迭代器
        return iter(batches)
    
    def __len__(self):
        if self.balance_strategy == 'equal':
            min_batches = min(len(indices) // self.batch_size for indices in self.indices_by_window_len.values())
            return min_batches * len(self.window_lengths)
        else:  # 'proportional'
            return sum(len(indices) // self.batch_size for indices in self.indices_by_window_len.values())

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


data_sampler_dict = {
    "KuaiSampler": KuaiSampler,
    "BasinBatchSampler": BasinBatchSampler,
    # TODO: DistributedSampler need more test
    # TODO: WindowLenBatchSampler need more test
    "WindowLenBatchSampler": WindowLenBatchSampler
}
