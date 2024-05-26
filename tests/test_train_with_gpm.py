"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-26 13:54:25
LastEditors: Wenyu Ouyang
Description: 
FilePath: /torchhydro/tests/test_train_with_gpm.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import cProfile
import io
import pstats
from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_train_evaluate(s2s_args, config_data, tmpdir):
    update_cfg(config_data, s2s_args)
    pr = cProfile.Profile()
    pr.enable()  # 开始分析
    train_and_evaluate(config_data)
    pr.disable()  # 停止分析

    # 创建一个 io.StringIO 对象，用于存储分析结果
    s = io.StringIO()
    # 创建一个 Stats 对象，并将分析结果排序后写入 s
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    # 打印分析结果
    print(s.getvalue())
    profile_output = tmpdir.join("profile_output.txt")
    with open(profile_output, "w") as f:
        f.write(s.getvalue())

    print(f"Profile data saved to {profile_output}")
