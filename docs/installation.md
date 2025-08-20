# Installation Guide

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA (optional, for GPU acceleration)

## Stable Release

The recommended way to install the latest stable version is using pip:

```bash
pip install torchhydro
```

If you don't have [pip](https://pip.pypa.io) installed, you can follow this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/).

## Development Version

For the latest development version, you can install directly from the GitHub repository:

```bash
git clone https://github.com/OuyangWenyu/torchhydro.git
cd torchhydro
pip install -e .
```

## Optional Dependencies

For specific functionality, you might need to install the following optional dependencies:

```bash
# For data visualization
pip install matplotlib seaborn

# For data processing
pip install pandas numpy

# For geographic data processing
pip install geopandas rasterio
```

## Verify Installation

After installation, you can verify it by running the following code in your Python environment:

```python
import torchhydro
print(torchhydro.__version__)
```
