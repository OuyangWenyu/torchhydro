"""
Author: Wenyu Ouyang
Date: 2024-05-27 13:33:08
LastEditTime: 2024-11-05 18:20:19
LastEditors: Wenyu Ouyang
Description: Unit test for datasets
FilePath: \torchhydro\tests\test_data_sets.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
import pickle
from sklearn.preprocessing import StandardScaler
from torchhydro.datasets.data_sets import BaseDataset, Seq2SeqDataset
from torchhydro.datasets.data_source import (
    SelfMadeForecastDataset,
    SelfMadeForecastDataset_P,
)
from torchhydro.datasets.data_sources import data_sources_dict


class MockDatasource:
    def __init__(self, source_path, time_unit="1D"):
        self.ngrid = 2
        self.nt = 366
        # self.data_cfgs = source_cfgs
        # self.data_source_dir = source_cfgs.get("source_path", "")
        self.source_path = source_path

    def read_ts_xrdataset(self, basin_id, t_range, var_lst):
        basins = [f"{i:08d}" for i in range(1013500, 1013500 + self.ngrid)]

        # 创建时间序列
        start_date, end_date = t_range
        times = pd.date_range(start=start_date, end=end_date, freq="D")

        # 确保时间序列长度和 nt 一致
        if len(times) != self.nt:
            raise ValueError(
                "The generated time range does not match the expected length."
            )

        # 创建数据变量
        data_vars = {
            var: (["basin", "time"], np.random.rand(self.ngrid, self.nt))
            for var in var_lst
        }

        # 创建 xarray Dataset
        dataset = xr.Dataset(
            data_vars=data_vars, coords={"basin": basins, "time": times}
        )
        if "streamflow" in dataset.variables:
            dataset["streamflow"].attrs["units"] = "mm/day"

        if "prcp" in dataset.variables:
            dataset["prcp"].attrs["units"] = "mm/day"

        return dataset

    def read_attr_xrdataset(self, basin_id, var_lst, all_number=True):
        # 创建 basin ID 列表
        basins = [f"{i:08d}" for i in range(1013500, 1013500 + self.ngrid)]

        # 创建数据变量
        attr_lst = ["geol_1st_class", "geol_2nd_class"]
        data_vars = {attr: (["basin"], np.random.rand(self.ngrid)) for attr in attr_lst}

        # 添加整数类型的虚拟数据变量

        for var in attr_lst:
            data_vars[var] = (["basin"], np.random.randint(0, 10, size=self.ngrid))

        return xr.Dataset(data_vars=data_vars, coords={"basin": basins})

    def read_forecast_xrdataset(
        self,
        basin_ids,
        reference_date,
        variables,
        lead_time_selector=None,
        num_samples=5,
        forecast_mode="all_lead_times",
    ):
        """读取预见期数据的模拟实现

        Parameters
        ----------
        basin_ids : list
            流域ID列表
        reference_date : datetime
            参考日期
        variables : list
            变量列表
        lead_time_selector : callable or list, optional
            选择lead_time的函数或固定值列表
        num_samples : int, optional
            样本数量
        forecast_mode : str, optional
            预见期数据加载模式，可选值为：
            - "all_lead_times": 加载所有预见期的数据（默认）
            - "specific_day_forecasts": 加载最后一天的1-n天前的预报该天的数据
            - "forecast_matrix": 加载预报矩阵，lead_time和time都是独立维度

        Returns
        -------
        xr.Dataset
            预见期数据
        """
        # 创建 basin ID 列表
        basins = [f"{i:08d}" for i in range(1013500, 1013500 + self.ngrid)]

        # 创建lead_time（发布时间）
        if lead_time_selector is None:
            lead_times = [
                reference_date - pd.Timedelta(days=i) for i in range(num_samples)
            ]
        elif callable(lead_time_selector):
            hours = lead_time_selector(np.arange(1, 20), reference_date)
            lead_times = [
                reference_date - pd.Timedelta(hours=int(h)) for h in hours[:num_samples]
            ]
        else:
            lead_times = [
                reference_date - pd.Timedelta(hours=int(lt))
                for lt in lead_time_selector[:num_samples]
            ]

        # 创建时间序列（目标时间）
        if forecast_mode == "specific_day_forecasts":
            # 模式1：加载最后一天的1-n天前的预报该天的数据
            # 所有预见期数据都指向同一个时间点（参考日期）
            times = [reference_date]
        elif forecast_mode == "forecast_matrix":
            # 模式3：加载预报矩阵，lead_time和time都是独立维度
            # 目标时间是未来几天
            times = [
                reference_date + pd.Timedelta(days=i) for i in range(7)
            ]  # 预报未来7天
        else:
            # 模式2：加载1-n天预见期的数据
            # 每个预见期对应不同的时间点
            times = [
                reference_date + pd.Timedelta(hours=int(lt))
                for lt in range(1, num_samples + 1)
            ]

        # 创建数据变量
        data_vars = {}

        if forecast_mode == "forecast_matrix":
            # 预报矩阵模式：数据形状为 [basin, lead_time, time]
            for var in variables:
                # 创建随机数据，形状为 [basin, lead_time, time]
                data = np.random.rand(self.ngrid, len(lead_times), len(times))
                data_vars[var] = (["basin", "lead_time", "time"], data)
        else:
            # 其他模式：数据形状为 [basin, lead_time]
            for var in variables:
                # 创建随机数据，形状为 [basin, lead_time]
                data = np.random.rand(self.ngrid, len(lead_times))
                data_vars[var] = (["basin", "lead_time"], data)

        # 创建 xarray Dataset
        if forecast_mode == "forecast_matrix":
            dataset = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "basin": basins,
                    "lead_time": lead_times,
                    "time": times,
                },
            )
        else:
            dataset = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "basin": basins,
                    "lead_time": lead_times,
                    "time": ("lead_time", times),
                },
            )

        # 添加单位属性
        if "streamflow" in dataset.variables:
            dataset["streamflow"].attrs["units"] = "mm/day"
        if "prcp" in dataset.variables:
            dataset["prcp"].attrs["units"] = "mm/day"

        return dataset

    def read_area(self, basin_ids):
        """读取流域面积"""
        return {basin: 100.0 for basin in basin_ids}


def test_create_lookup_table(tmp_path):
    temp_test_path = tmp_path / "test_datasets"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})
    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01013501",
        ],  # Add this line with the actual object IDs
        "t_range_train": [
            "2001-01-01",
            "2002-01-01",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_test": [
            "2002-01-01",
            "2003-01-01",
        ],  # Add this line with the actual start and end dates for validation.
        "relevant_cols": [
            # List the relevant column names here.
            "prcp",
            "pet",
            # ... other relevant columns ...
        ],
        "target_cols": [
            # List the target column names here.
            "streamflow",
            "surface_sm",
            # ... other target columns ...
        ],
        "constant_cols": [
            # List the constant column names here.
            "geol_1st_class",
            "geol_2nd_class",
            # ... other constant columns ...
        ],
        "forecast_history": 7,
        "warmup_length": 14,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",  # Add the scaler configuration here
        "stat_dict_file": None,  # Added the missing configuration
    }
    is_tra_val_te = "train"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    lookup_table = dataset.lookup_table
    assert isinstance(lookup_table, dict)
    assert len(lookup_table) > 0
    assert all(
        isinstance(key, int) and isinstance(value, tuple)
        for key, value in lookup_table.items()
    )
    is_tra_val_te = "test"
    mock_data = np.random.rand(100, 2)  # Replace with relevant data.
    scaler = StandardScaler()
    scaler.fit(mock_data)
    scaler_file_path = tmp_path / "test_datasets_scaler.pkl"
    data_cfgs["stat_dict_file"] = str(scaler_file_path)
    with open(scaler_file_path, "wb") as file:
        pickle.dump(scaler, file)
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    lookup_table = dataset.lookup_table
    assert isinstance(lookup_table, dict)
    assert len(lookup_table) > 0
    assert all(
        isinstance(key, int) and isinstance(value, tuple)
        for key, value in lookup_table.items()
    )


def test_forecast_data_integration(tmp_path):
    """测试预见期数据集成功能"""
    temp_test_path = tmp_path / "test_forecast_datasets"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})

    # 基本配置
    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01013501",
        ],
        "t_range_train": [
            "2001-01-01",
            "2002-01-01",
        ],
        "t_range_test": [
            "2002-01-01",
            "2003-01-01",
        ],
        "relevant_cols": [
            "prcp",
            "pet",
        ],
        "target_cols": [
            "streamflow",
            "surface_sm",
        ],
        "constant_cols": [
            "geol_1st_class",
            "geol_2nd_class",
        ],
        "forecast_history": 7,
        "warmup_length": 14,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "stat_dict_file": None,
        # 预见期数据配置
        "use_forecast_data": True,
        "forecast_target_available": False,
        "forecast_cfg": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
            "num_samples": 5,  # 添加5个预见期样本
            "lead_time_selector": [1, 2, 3, 4, 5],  # 选择的lead_time值
        },
    }

    # 测试训练集
    is_tra_val_te = "train"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)

    # 验证数据形状
    assert dataset.x.shape[1] > 366  # 时间维度应该增加了预见期样本数

    # 获取一个样本并验证
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # 验证预见期标志是否存在（最后一列）
    if data_cfgs.get("use_forecast_data", False):
        # 检查x的最后一列是否包含预见期标志（0或1）
        forecast_flags = x[:, -1].numpy()
        # 验证是否有部分数据被标记为预见期数据
        assert np.any(forecast_flags == 1.0)
        # 验证是否有部分数据被标记为历史数据
        assert np.any(forecast_flags == 0.0)

    # 测试测试集
    is_tra_val_te = "test"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)

    # 获取一个样本并验证
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # 测试自定义lead_time选择函数
    data_cfgs["forecast_cfg"]["lead_time_selector"] = (
        lambda lead_times, ref_date: lead_times[::2][:3]
    )
    dataset = BaseDataset(data_cfgs, is_tra_val_te)
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_forecast_data_without_integration(tmp_path):
    """测试不使用预见期数据的情况"""
    temp_test_path = tmp_path / "test_no_forecast_datasets"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})

    # 基本配置，不使用预见期数据
    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "01013500",
            "01013501",
        ],
        "t_range_train": [
            "2001-01-01",
            "2002-01-01",
        ],
        "t_range_test": [
            "2002-01-01",
            "2003-01-01",
        ],
        "relevant_cols": [
            "prcp",
            "pet",
        ],
        "target_cols": [
            "streamflow",
            "surface_sm",
        ],
        "constant_cols": [
            "geol_1st_class",
            "geol_2nd_class",
        ],
        "forecast_history": 7,
        "warmup_length": 14,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "stat_dict_file": None,
        # 添加新的配置项
        "horizon": 1,  # 确保有 horizon 参数
        "train_mode": True,  # 添加 train_mode 参数
        # 明确设置不使用预见期数据
        "use_forecast_data": False,
    }

    # 测试训练集
    is_tra_val_te = "train"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)

    # 验证数据形状
    assert dataset.x.shape[1] == 366  # 时间维度应该是原始长度

    # 获取一个样本并验证
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # 测试测试集
    is_tra_val_te = "test"
    dataset = BaseDataset(data_cfgs, is_tra_val_te)

    # 获取一个样本并验证
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_forecast_data_modes(tmp_path):
    """测试不同的预见期数据加载模式"""
    # 使用传入的临时路径而不是硬编码路径
    temp_test_path = tmp_path / "test_forecast_modes"
    os.makedirs(temp_test_path, exist_ok=True)
    data_sources_dict.update({"mockdatasource": MockDatasource})

    # 基本配置
    base_cfg = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "changdian_60513400",
        ],
        "t_range_train": [
            "2017-01-01",
            "2018-01-01",
        ],
        "t_range_test": [
            "2019-01-01",
            "2020-01-01",
        ],
        "relevant_cols": [
            "precipitation",
            # "pet",
        ],
        "target_cols": [
            "streamflow",
            # "surface_sm",
        ],
        "constant_cols": [
            "geol_1st_class",
            "geol_2nd_class",
        ],
        "forecast_history": 5,
        "warmup_length": 14,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "stat_dict_file": None,
        "use_forecast_data": True,
        "forecast_target_available": False,
    }

    # 测试模式1：加载最后一天的1-7天前的预报该天的数据
    data_cfgs1 = base_cfg.copy()
    data_cfgs1["forecast_cfg"] = {
        # "source_cfgs": {
        #     "source_name": "mockdatasource",
        #     "source_path": str(temp_test_path),
        # },
        # "source_name": "mockdatasource",
        # "source_path": str(temp_test_path),
        "num_samples": 5,
        "lead_time_selector": [1, 2, 3, 4, 5],
        "forecast_mode": "specific_day_forecasts",
    }

    # 测试训练集
    is_tra_val_te = "train"
    dataset1 = BaseDataset(data_cfgs1, is_tra_val_te)

    # 验证数据形状
    assert dataset1.x.shape[1] > 366  # 时间维度应该增加了预见期样本数

    # 获取一个样本并验证
    x1, y1 = dataset1[0]
    assert isinstance(x1, torch.Tensor)
    assert isinstance(y1, torch.Tensor)

    # 测试模式2：加载1-7天预见期的数据
    data_cfgs2 = base_cfg.copy()
    data_cfgs2["forecast_cfg"] = {
        "source_name": "mockdatasource",
        "source_path": str(temp_test_path),
        "num_samples": 7,
        "lead_time_selector": [1, 2, 3, 4, 5, 6, 7],
        "forecast_mode": "all_lead_times",  # 默认模式
    }

    # 测试训练集
    dataset2 = BaseDataset(data_cfgs2, is_tra_val_te)

    # 验证数据形状
    assert dataset2.x.shape[1] > 366  # 时间维度应该增加了预见期样本数

    # 获取一个样本并验证
    x2, y2 = dataset2[0]
    assert isinstance(x2, torch.Tensor)
    assert isinstance(y2, torch.Tensor)


def test_read_forecast_xrdataset():
    test_datasource = SelfMadeForecastDataset(
        data_path=r"D:\research\data\tuojiang\forcing",
        time_unit=["1D"],
    )
    data = test_datasource.read_forecast_xrdataset(
        basin_ids=["changdian_60513400"],
        times=["2021-01-01"],
        reference_date="2001-01-01",
        lead_times=[1, 2, 3, 4, 5],
        forecast_mode="specific_day_forecasts",
    )
    assert type(data) is xr.Dataset


if __name__ == "__main__":
    # test_create_lookup_table()
    # test_forecast_data_integration()
    # test_forecast_data_without_integration()

    # 修正：直接调用新函数，不传递参数
    test_forecast_data_modes()

    # 或者，如果你想使用特定的NC文件路径，可以修改test_forecast_data_modes函数内部的代码
    # 将temp_test_path设置为你的NC文件所在目录
