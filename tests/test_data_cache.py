"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2023-09-18 13:59:47
LastEditors: Wenyu Ouyang
Description: Test the cache file
FilePath: /torchhydro/tests/test_data_cache.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import hydrodataset as hds


def test_cache_file():
    """
    Test whether the cache file is generated correctly
    """
    camels_us = hds.Camels()
    camels_us.cache_xrdataset()
