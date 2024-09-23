"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2024-09-10 20:10:25
LastEditors: Wenyu Ouyang
Description: Top-level package for torchhydro
FilePath: \torchhydro\torchhydro\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

""""""

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = '0.0.8'

from pathlib import Path
import yaml
import os

from hydroutils import hydro_file

CACHE_DIR = Path(hydro_file.get_cache_dir())
SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "minio:\n"
        "  server_url: 'http://minio.waterism.com:9090' # Update with your URL\n"
        "  client_endpoint: 'http://minio.waterism.com:9000' # Update with your URL\n"
        "  access_key: 'your minio access key'\n"
        "  secret: 'your minio secret'\n\n"
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin'\n"
        "  datasets-interim: 'D:\\data\\waterism\\datasets-interim'"
        "postgres:\n"
        "  server_url: 'your database server URL'"  # if you don't have a database, you can just write this key and leave the rest empty
        "  port: your database port"
        "  username: 'your database username'"
        "  password: 'your database password'"
        "  database: 'your database name'"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "minio": ["server_url", "client_endpoint", "access_key", "secret"],
        "local_data_path": ["root", "datasets-origin", "datasets-interim"],
        "postgres": ["server_url", "port", "username", "password", "database"],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


try:
    SETTING = read_setting(SETTING_FILE)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")


from .configs import *
from .datasets import *
from .models import *
from .trainers import *
from .explainers import *
