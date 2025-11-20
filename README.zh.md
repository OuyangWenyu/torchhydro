<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-13 18:29:19
 * @LastEditTime: 2025-11-08 15:34:19
 * @LastEditors: Wenyu Ouyang
 * @Description: Chinese version of the README
 * @FilePath: \torchhydro\README.zh.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# Torchhydro

[![image](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![image](https://img.shields.io/conda/vn/conda-forge/torchhydro.svg)](https://anaconda.org/conda-forge/torchhydro)

- 许可证: BSD license
- 文档: https://OuyangWenyu.github.io/torchhydro

`torchhydro` 为水文水资源领域的深度学习应用提供数据集和模型。

## 安装

### 普通用户

您可以使用 `pip` 或更快的 `uv` 来安装 `torchhydro`。

```shell
pip install torchhydro
```
或
```shell
uv pip install torchhydro
```

### 开发者

如果您想为本项目做出贡献，我们推荐使用 `uv` 进行环境管理。

```shell
# 克隆仓库
git clone https://github.com/OuyangWenyu/torchhydro.git
cd torchhydro

# 创建虚拟环境并安装所有依赖
uv sync --all-extras
```

## 使用方法

### 1. 配置数据路径

在运行任何示例之前，您需要告诉 `torchhydro` 您的数据存放位置。

在您的用户主目录（Windows下为 `C:\Users\YourUsername`，Linux/macOS下为 `~/`）下创建一个名为 `hydro_setting.yml` 的文件。然后，添加以下内容，并将其指向您的数据文件夹：

```yaml
local_data_path:
  root: 'D:/data/waterism' # 更新为您的根数据目录
  datasets-origin: 'D:/data/waterism/datasets-origin'
  datasets-interim: 'D:/data/waterism/datasets-interim'
  cache: 'D:/data/waterism/cache'
```

这些示例使用了 CAMELS 数据集。如果您没有该数据集，`torchhydro` 会自动调用 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 进行下载。

### 2. 运行示例

我们在 `examples/` 目录中提供了独立的脚本来帮助您入门。

- **`examples/lstm_camels_example.py`**: 在 CAMELS 数据集上训练一个标准 LSTM 模型的基础示例。
- **`examples/dpl_xaj_example.py`**: 一个基于新安江（XAJ）水文模型训练可微分模型的高级示例。

运行一个示例：
```shell
python examples/lstm_camels_example.py
```

您可以随时修改这些脚本来试验不同的模型、数据集和参数。

## 探索更多功能

以上示例涵盖了两个主要用例，但 `torchhydro` 的功能远不止于此，它原生支持多种模型、数据集和数据源。您可以通过我们完整的公共 API 文档来探索所有可用的组件：

- **[模型 API](docs/api/models.md)**: 发现所有可用的模型架构。
- **[数据集 API](docs/api/datasets.md)**: 查看所有的数据集类、数据源和采样器。
- **[训练器 API](docs/api/trainers.md)**: 理解核心的训练和评估流程。

我们正持续努力扩展文档，以包含更多示例。

## 主要模块

项目由几个关键模块组成：

- **Trainers (训练器)**: 管理端到端的训练和评估流程。核心的 `DeepHydro` 类处理数据加载、模型初始化、训练循环和评估。其设计具有可扩展性，以支持如迁移学习或多任务学习等多种学习范式。
- **Models (模型)**: 包含所有可用的模型架构，包括标准神经网络（如 LSTM）和可微分模型。一个中央字典使得通过配置来选择模型和损失函数变得简单。
- **Datasets (数据集)**: 提供数据处理能力。它通过与 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset)（用于 CAMELS 等公共数据集）和 [hydrodatasource](https://github.com/iHeadWater/hydrodatasource)（用于自定义数据）等数据源库的接口，创建适用于训练的 `torch.utils.data.Dataset` 对象。
- **Configs (配置)**: 管理所有实验配置，包括模型、数据（时间周期、变量）、训练（轮次、批量大小）和评估的设置。

## 为什么选择 Torchhydro?

尽管已有像 [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) 这样成熟的工具，但 `torchhydro` 的开发基于一套不同的架构理念：

1.  **解耦的数据源**: 我们认为数据处理，特别是对于复杂或私有数据集，需要一个独立的抽象层。我们的方法是先使用 `hydrodataset` 和 `hydrodatasource` 管理数据访问，然后再将其送入 PyTorch `Dataset`。这种模块化设计促进了代码复用，并使得数据源工具可以在没有深度学习模型的情况下被使用。
2.  **灵活的学习范式**: 该框架从一开始就明确设计为不仅支持标准的监督学习，还支持更复杂的模式，如迁移学习、多任务学习和联邦学习。
3.  **深度配置**: 我们对流程的许多方面提供了细粒度的控制，包括数据遍历、归一化方法、批处理采样策略和高级的 dropout 技术，从而为实验提供了更大的灵活性。
4.  **可扩展性**: 核心设计原则是尽可能地将配置外部化，以实现不同数据源和模型的灵活匹配与调用。

## 其他信息

本软件包的灵感来源于：

- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [hydroDL](https://github.com/mhpi/hydroDL)

本软件包是使用 [Cookiecutter](https://github.com/cookiecutter/cookiecutter) 和 [giswqs/pypackage](https://github.com/giswqs/pypackage) 项目模板创建的。