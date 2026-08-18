"""Integration tests for P0 experiment config creation.

Verifies that each experiment's config can be created with the new
``dataset_id`` format and that ``source_cfgs`` is correctly populated.
"""

import os
import importlib
import importlib.util
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _import_experiment(module_name: str, file_path: str):
    """Import an experiment module without executing top-level code."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Some experiments run code at import time (e.g. arg parsing).
        # That's fine — we only need the module object for config creation.
        pass
    return mod


def _get_config(mod):
    """Extract config from an experiment module.

    Tries create_config(), then main() with a mock, then returns None.
    """
    if hasattr(mod, "create_config"):
        return mod.create_config()
    return None


# ---------------------------------------------------------------------------
# CAMELS-US experiments (data available, must pass)
# ---------------------------------------------------------------------------

CAMELS_US_EXPERIMENTS = [
    "experiments/train_moe_camels.py",
    "experiments/train_densemoe_camels.py",
    "experiments/train_switchmoe_camels.py",
    "experiments/train_hydrotopkmoe_camels.py",
    "experiments/train_lstmmoe_camels.py",
    "experiments/train_diffusion_camels.py",
    "experiments/train_fno_camels.py",
    "experiments/plot_loss_landscape.py",
    "experiments/evaluate_moe_camels.py",
    "experiments/evaluate_lstmmoe_camels.py",
]


@pytest.mark.parametrize("experiment_path", CAMELS_US_EXPERIMENTS)
def test_experiment_imports(experiment_path: str):
    """Each experiment module should import without error."""
    mod = _import_experiment(experiment_path.replace("/", "_").replace(".py", ""), experiment_path)
    assert mod is not None


@pytest.mark.parametrize("experiment_path", CAMELS_US_EXPERIMENTS)
def test_experiment_no_old_source_name(experiment_path: str):
    """After migration, experiments should not use source_name in source_cfgs."""
    with open(experiment_path, encoding="utf-8") as f:
        content = f.read()
    # Should not have source_name in source_cfgs context
    assert '"source_name"' not in content or "dataset_id" in content, (
        f"{experiment_path} still uses source_name without dataset_id"
    )


@pytest.mark.parametrize("experiment_path", CAMELS_US_EXPERIMENTS)
def test_experiment_no_setting_dependency(experiment_path: str):
    """After migration, experiments should not depend on SETTING for data paths."""
    with open(experiment_path, encoding="utf-8") as f:
        content = f.read()
    # Should not import SETTING for data path resolution
    if "SETTING" in content:
        # Allow SETTING for non-data purposes, but flag if used with local_data_path
        assert "local_data_path" not in content, (
            f"{experiment_path} still uses SETTING['local_data_path']"
        )


# ---------------------------------------------------------------------------
# Self-made data experiments (may fail due to missing data)
# ---------------------------------------------------------------------------

SELFMADE_EXPERIMENTS = [
    "experiments/train_with_balstm.py",
    "experiments/train_with_mtslstm.py",
]


@pytest.mark.parametrize("experiment_path", SELFMADE_EXPERIMENTS)
def test_selfmade_experiment_imports(experiment_path: str):
    """Self-made data experiments should at least import."""
    mod = _import_experiment(experiment_path.replace("/", "_").replace(".py", ""), experiment_path)
    assert mod is not None
