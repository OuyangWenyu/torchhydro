# Installation Guide

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA (optional, for GPU acceleration)

## Stable Release

The recommended way to install the latest stable version is using pip:

```bash
pip install torchhydro
```

or with [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv pip install torchhydro
```

## Development Version

For the latest development version, install directly from the repository. We recommend `uv` for environment management:

```bash
git clone https://github.com/OuyangWenyu/torchhydro.git
cd torchhydro
uv sync --all-extras
```

> **Sister packages.** `hydrodataset`, `hydrodatasource`, and `hydroutils` are configured as editable local dependencies in `pyproject.toml`. For local development they must be cloned into sibling directories (`../hydrodataset`, `../hydrodatasource`, `../hydroutils`). When installing from PyPI, released versions are used instead.

## Configure Data Path

Before running any example, create a `hydro_setting.yml` file in your user home directory (`C:\Users\YourUsername` on Windows, `~/` on Linux/macOS) pointing to your data folders:

```yaml
storage:
  default_source: local
  local:
    root: 'D:/data/hydrodatasets'
  cache: 'D:/data/hydrodatasets/cache'
```

Standard dataset paths (e.g., CAMELS-US) are resolved automatically from `storage.local.root` by the unified data resolver — no per-dataset path configuration is needed.

## Verify Installation

After installation, you can verify it by running the following code in your Python environment:

```python
import torchhydro
print(torchhydro.__version__)
```
