"""
Author: Wenyu Ouyang
Date: 2026-01-09
LastEditTime: 2026-01-09
LastEditors: Wenyu Ouyang
Description: NaN-safe 版本的统计计算函数
FilePath: \torchhydro\torchhydro\datasets\data_scalers_nan_safe.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
from typing import Dict, List


def cal_stat_nan_safe(x: np.ndarray) -> List[float]:
    """
    计算统计量（p10, p90, 均值和标准差），安全处理 NaN 值
    
    Parameters
    ----------
    x : np.ndarray
        输入数据，可能包含 NaN 值
        
    Returns
    -------
    List[float]
        [p10, p90, mean, std] - 使用 nanpercentile, nanmean 和 nanstd 计算
    """
    # 使用 nanpercentile, nanmean 和 nanstd 忽略 NaN 值
    p10 = np.nanpercentile(x, 10)
    p90 = np.nanpercentile(x, 90)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    
    # 如果所有值都是 NaN，返回默认值
    if np.isnan(mean):
        print(f"警告: 数据全部为 NaN，使用默认值 mean=0.0")
        mean = 0.0
    if np.isnan(std) or std == 0:
        print(f"警告: 标准差为 NaN 或 0，使用默认值 std=1.0")
        std = 1.0
    if np.isnan(p10):
        p10 = mean - std
    if np.isnan(p90):
        p90 = mean + std
        
    return [float(p10), float(p90), float(mean), float(std)]


def cal_stat_prcp_norm_nan_safe(
    x: np.ndarray, 
    mean_prep: np.ndarray
) -> List[float]:
    """
    计算降水归一化的统计量，安全处理 NaN 值
    
    Parameters
    ----------
    x : np.ndarray
        streamflow 数据，shape: (basin, time) 或 (basin, time, 1)
    mean_prep : np.ndarray
        平均降水量，shape: (basin, 1)
        
    Returns
    -------
    List[float]
        [p10, p90, mean, std] - 归一化后的统计量
    """
    # 确保维度匹配
    if x.ndim == 3:
        x = x.squeeze(-1)  # (basin, time, 1) -> (basin, time)
    
    # 使用 log(sqrt(x) + 0.1) 变换
    x_transformed = np.log10(np.sqrt(x) + 0.1)
    
    # 除以平均降水量进行归一化
    if mean_prep.ndim == 2 and mean_prep.shape[1] == 1:
        mean_prep = mean_prep.squeeze(-1)  # (basin, 1) -> (basin,)
    
    # 广播除法: (basin, time) / (basin,) -> (basin, time)
    x_normalized = x_transformed / mean_prep[:, np.newaxis]
    
    # 使用 nanpercentile, nanmean 和 nanstd 计算统计量
    p10 = np.nanpercentile(x_normalized, 10)
    p90 = np.nanpercentile(x_normalized, 90)
    mean = np.nanmean(x_normalized)
    std = np.nanstd(x_normalized)
    
    # 处理异常情况
    if np.isnan(mean):
        print(f"警告: streamflow 归一化后均值为 NaN，使用默认值 0.0")
        mean = 0.0
    if np.isnan(std) or std == 0:
        print(f"警告: streamflow 归一化后标准差为 NaN 或 0，使用默认值 1.0")
        std = 1.0
    if np.isnan(p10):
        p10 = mean - std
    if np.isnan(p90):
        p90 = mean + std
        
    return [float(p10), float(p90), float(mean), float(std)]


def cal_stat_gamma_nan_safe(x: np.ndarray) -> List[float]:
    """
    使用 gamma 变换计算统计量，安全处理 NaN 值
    
    Parameters
    ----------
    x : np.ndarray
        输入数据
        
    Returns
    -------
    List[float]
        [p10, p90, mean, std] - 变换后的统计量
    """
    # 使用 log(sqrt(x) + 0.1) 变换
    x_transformed = np.log10(np.sqrt(x) + 0.1)
    
    # 使用 nanpercentile, nanmean 和 nanstd
    p10 = np.nanpercentile(x_transformed, 10)
    p90 = np.nanpercentile(x_transformed, 90)
    mean = np.nanmean(x_transformed)
    std = np.nanstd(x_transformed)
    
    # 处理异常情况
    if np.isnan(mean):
        print(f"警告: gamma 变换后均值为 NaN，使用默认值 0.0")
        mean = 0.0
    if np.isnan(std) or std == 0:
        print(f"警告: gamma 变换后标准差为 NaN 或 0，使用默认值 1.0")
        std = 1.0
    if np.isnan(p10):
        p10 = mean - std
    if np.isnan(p90):
        p90 = mean + std
        
    return [float(p10), float(p90), float(mean), float(std)]


def check_data_quality(vars_data: Dict, var_name: str) -> Dict[str, any]:
    """
    检查数据质量，报告 NaN 值的情况
    
    Parameters
    ----------
    vars_data : Dict
        变量数据字典
    var_name : str
        变量名称
        
    Returns
    -------
    Dict[str, any]
        数据质量报告
    """
    report = {}
    
    for k, v in vars_data.items():
        if v is None:
            continue
            
        if var_name in v.coords["variable"].values:
            data = v.sel(variable=var_name).to_numpy()
            
            total_count = data.size
            nan_count = np.isnan(data).sum()
            nan_ratio = nan_count / total_count if total_count > 0 else 0
            
            report[k] = {
                "total_count": total_count,
                "nan_count": nan_count,
                "nan_ratio": f"{nan_ratio:.2%}",
                "valid_count": total_count - nan_count,
                "min": np.nanmin(data) if nan_count < total_count else np.nan,
                "max": np.nanmax(data) if nan_count < total_count else np.nan,
                "mean": np.nanmean(data) if nan_count < total_count else np.nan,
                "std": np.nanstd(data) if nan_count < total_count else np.nan,
            }
    
    return report
