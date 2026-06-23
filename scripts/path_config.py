"""Load batch-automation paths from configs/paths.yaml.

Lookup order:
    1. $TORCHHYDRO_PATHS_CFG (absolute or relative to repo root)
    2. <repo_root>/configs/paths.yaml
    3. <repo_root>/configs/paths.example.yaml
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE = {}


def get_paths_cfg_file() -> Path:
    env = os.environ.get("TORCHHYDRO_PATHS_CFG")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (REPO_ROOT / p)
    local = REPO_ROOT / "configs" / "paths.yaml"
    if local.exists():
        return local
    return REPO_ROOT / "configs" / "paths.example.yaml"


def get_paths() -> dict:
    cfg_file = get_paths_cfg_file()
    cached = _CACHE.get(str(cfg_file))
    if cached is not None:
        return cached
    if not cfg_file.exists():
        raise FileNotFoundError(f"paths config not found: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _CACHE[str(cfg_file)] = data
    return data


if __name__ == "__main__":
    import json

    print(f"# {get_paths_cfg_file()}")
    print(json.dumps(get_paths(), indent=2, ensure_ascii=False))
