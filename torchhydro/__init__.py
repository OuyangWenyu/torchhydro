"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2023-12-17 16:12:33
LastEditors: Wenyu Ouyang
Description: Top-level package for torchhydro
FilePath: \torchhydro\torchhydro\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

""""""

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = '0.0.4'

# we use a .hydrodataset dir to save the setting
from pathlib import Path
import json


datasource_setting_dir = Path.home().joinpath(".hydrodataset")
if not datasource_setting_dir.is_dir():
    datasource_setting_dir.mkdir(parents=True)
datasource_cache_dir = datasource_setting_dir.joinpath("cache")
if not datasource_cache_dir.is_dir():
    datasource_cache_dir.mkdir(parents=True)
datasource_setting_file = datasource_setting_dir.joinpath("settings.json")
if not datasource_setting_file.is_file():
    datasource_setting_file.touch(exist_ok=False)
    # default data dir is cache, user should modify it to his/her own
    set_json = {
        "cache": str(datasource_cache_dir),
        "root": str(datasource_cache_dir),
        "datasets-origin": str(
            datasource_cache_dir.joinpath("waterism", "datasets-origin")
        ),
        "datasets-interim": str(
            datasource_cache_dir.joinpath("waterism", "datasets-interim")
        ),
    }
    # Ensure directories exist
    for path in set_json.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    with open(datasource_setting_file, "w") as file:
        json.dump(set_json, file, indent=4)

# read json file
DATASOURCE_SETTINGS = json.loads(datasource_setting_file.read_text())

from .configs import *
from .datasets import *
from .models import *
from .trainers import *
from .explainers import *
