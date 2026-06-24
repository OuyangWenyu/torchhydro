"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2025-11-08 15:06:58
LastEditors: Wenyu Ouyang
Description: Top-level package for torchhydro
FilePath: \torchhydro\torchhydro\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

""""""

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
try:
    from ._version import __version__
except ImportError:
    # This is a fallback for cases where the package is not installed in an editable mode and the _version.py file is not generated.
    # For instance, when the code is just cloned and not installed.
    __version__ = "0.0.0+unknown"

from pathlib import Path
import warnings

from hydrodataset.configs.settings import (
    load_settings,
    get_local_root,
    get_cache_dir,
)

# Load settings from ~/hydro_setting.yml (shared with hydrodataset)
_settings = load_settings()

# Extract resolved paths using the public API
LOCAL_ROOT = get_local_root(_settings)
CACHE_DIR = get_cache_dir(_settings)

# Raw settings dict (exported for code that needs e.g. minio/postgres config)
SETTING = _settings

# Validate the one required setting
if LOCAL_ROOT is None:
    raise RuntimeError(
        "storage.local.root is not configured.\n"
        "Create ~/hydro_setting.yml with:\n"
        "  storage:\n"
        "    local:\n"
        "      root: D:/data/hydrodatasets"
    )


from .configs import *
from .datasets import *
from .models import *
from .trainers import *
from .explainers import *
