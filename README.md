<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-13 18:29:19
 * @LastEditTime: 2025-11-08 15:14:07
 * @LastEditors: Wenyu Ouyang
 * @Description: English version of the README
 * @FilePath: \torchhydro\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# Torchhydro

[![image](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![image](https://img.shields.io/conda/vn/conda-forge/torchhydro.svg)](https://anaconda.org/conda-forge/torchhydro)

- License: BSD license
- Documentation: https://OuyangWenyu.github.io/torchhydro

`torchhydro` provides datasets and models for applying deep learning to hydrological modeling.

## Installation

### For Users

You can install `torchhydro` using `pip` or `uv` (which is faster).

```shell
pip install torchhydro
```
or
```shell
uv pip install torchhydro
```

### For Developers

If you want to contribute to the project, we recommend using `uv` for environment management.

```shell
# Clone the repository
git clone https://github.com/OuyangWenyu/torchhydro.git
cd torchhydro

# Create a virtual environment and install all dependencies
uv sync --all-extras
```

## Usage

### 1. Configure Data Path

Before running any examples, you need to tell `torchhydro` where your data is located.

Create a file named `hydro_setting.yml` in your user home directory (`C:\Users\YourUsername` on Windows or `~/` on Linux/macOS). Then, add the following content, pointing to your data folders:

```yaml
local_data_path:
  root: 'D:/data/waterism' # Update with your root data directory
  datasets-origin: 'D:/data/waterism/datasets-origin'
  datasets-interim: 'D:/data/waterism/datasets-interim'
  cache: 'D:/data/waterism/cache'
```

The examples use the CAMELS dataset. If you don't have it, `torchhydro` will automatically call [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) to download it.

### 2. Run Examples

We provide standalone scripts in the `examples/` directory to help you get started.

- **`examples/lstm_camels_example.py`**: A basic example of training a standard LSTM model on the CAMELS dataset.
- **`examples/dpl_xaj_example.py`**: An advanced example of training a differentiable model based on the Xinanjiang (XAJ) hydrological model.

To run an example:
```shell
python examples/lstm_camels_example.py
```

Feel free to modify these scripts to experiment with different models, datasets, and parameters.

## Explore More Features

The examples above cover two primary use cases, but `torchhydro` is much more flexible. We support a variety of models, datasets, and data sources out of the box. Explore the full public API to see all available components:

- **[Models API](docs/api/models.md)**: Discover all available model architectures.
- **[Datasets API](docs/api/datasets.md)**: See all dataset classes, data sources, and samplers.
- **[Trainers API](docs/api/trainers.md)**: Understand the core training and evaluation pipeline.

We are continuously working to expand the documentation with more examples.

## Main Modules

The project is organized into several key modules:

- **Trainers**: Manages the end-to-end training and evaluation pipeline. The core `DeepHydro` class handles data loading, model initialization, training loops, and evaluation. It is designed to be extensible for various learning paradigms like transfer learning or multi-task learning.
- **Models**: Contains all available model architectures, including standard neural networks (e.g., LSTM) and differentiable models. A central dictionary allows for easy configuration and selection of models and loss functions.
- **Datasets**: Provides data handling capabilities. It interfaces with data source libraries like [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) (for public datasets like CAMELS) and [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) (for custom data) to create `torch.utils.data.Dataset` objects suitable for training.
- **Configs**: Manages all experiment configurations, including settings for the model, data (time periods, variables), training (epochs, batch size), and evaluation.

## Why Torchhydro?

While mature tools like [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) exist, `torchhydro` was developed with a different architectural philosophy:

1.  **Decoupled Data Sources**: We believe data handling, especially for complex or private datasets, requires a separate abstraction layer. Our approach uses `hydrodataset` and `hydrodatasource` to manage data access first, which then feeds into a PyTorch `Dataset`. This modularity promotes code reuse and allows the data source tools to be used even without a deep learning model.
2.  **Flexible Learning Paradigms**: The framework is explicitly designed to support not just standard supervised learning, but also more complex modes like transfer learning, multi-task learning, and federated learning from the ground up.
3.  **Deep Configuration**: We provide fine-grained control over many aspects of the pipeline, including data traversal, normalization methods, batch sampling strategies, and advanced dropout techniques, allowing for greater flexibility in experimentation.
4.  **Extensibility**: The core design principle is to externalize as much configuration as possible, enabling flexible matching and calling of different data sources and models.

## Additional Information

This package was inspired by:

- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [hydroDL](https://github.com/mhpi/hydroDL)

This package was created using the [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.