<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-13 18:29:19
 * @LastEditTime: 2026-08-17
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

`torchhydro` 为水文水资源领域的深度学习应用提供数据集和模型,支持常规、迁移学习、多任务学习与联邦学习等模式。

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

> **本地开发提示**:姊妹包 `hydrodataset`、`hydrodatasource`、`hydroutils` 在 `pyproject.toml` 中配置为 editable 本地依赖,需克隆到并行目录(`../hydrodataset`、`../hydrodatasource`、`../hydroutils`)才能被 `uv sync` 解析。发布到 PyPI 时使用正式版本。

## 使用方法

### 1. 配置数据路径

在运行任何示例之前，您需要告诉 `torchhydro` 您的数据存放位置。

在您的用户主目录(Windows 下为 `C:\Users\YourUsername`,Linux/macOS 下为 `~/`)下创建一个名为 `hydro_setting.yml` 的文件。然后添加以下内容,并将其指向您的数据文件夹:

```yaml
storage:
  default_source: local
  local:
    root: 'D:/data/hydrodatasets' # 更新为您的根数据目录
  cache: 'D:/data/hydrodatasets/cache'
```

每个标准数据集(如 CAMELS-US)的路径由统一数据 resolver 从该 root 自动解析,您**无需**为每个数据集单独配置路径。`storage.default_source` 用于选择存储后端(默认为 `local`,也可选 `s3`/`cloud`)。

### 2. 运行示例

我们在 `examples/` 目录中提供了独立的脚本来帮助您入门。

- **`examples/lstm_camels_example.py`**: 在 CAMELS-US 数据集上训练一个标准 LSTM 模型的基础示例。
- **`examples/dpl_xaj_example.py`**: 一个基于新安江(XAJ)水文模型训练可微分模型的高级示例。

运行一个示例:
```shell
python examples/lstm_camels_example.py
```

您可以随时修改这些脚本来试验不同的模型、数据集和参数。

## 数据源

所有数据访问都通过**统一数据 resolver**(`torchhydro.configs.data_resolver`),它委托给 hydrodataset/hydrodatasource 注册表。在 `source_cfgs` 中使用 `dataset_id` 键选择数据集;路径统一从 `hydro_setting.yml` 解析(为向后兼容仍接受旧的 `source_name`/`source_path` 键,但 `source_path` 会被忽略,以统一解析为准)。

由注册表解析的标准数据集:

| `dataset_id` | 说明 |
|---|---|
| `camels_us` | CAMELS-US(美国 671 个流域) |
| `camelsh` | CAMELS-Hourly(小时尺度) |
| `caravan` / `grdc_caravan` | Caravan / GRDC-Caravan 多数据集集合 |
| `camels_aus` / `camels_br` / `camels_cl` / `camels_gb` / `camels_col` | CAMELS-澳大利亚 / 巴西 / 智利 / 英国 / 哥伦比亚 |

自定义数据读取器(由 `hydrodatasource` 提供):`selfmadehydrodataset`、`selfmadeforecastdataset`、`longtermdataset`、`floodeventdatasource`、`stationhydrodataset`、`tghydrodatasource`。

> GAGES-II、MOPEX 与水库数据源读取器已从 `torchhydro` 移交至 `hydrodatasource`,`torchhydro` 不再复现。

## 基准测试结果

我们提供了在 CAMELS-US 数据集上的基准测试结果，以展示模型的性能表现。下图展示了两种 CAMELS 数据处理方式的 NSE(纳什效率系数)分布对比:

<img src="docs/images/nse_comparison_camels.png" alt="NSE 对比" width="600"/>

**图示**: 在 CAMELS-US 数据集上训练的 LSTM 模型的 NSE 对比。左侧箱线图(红色)展示了使用 `hydrodataset.Camels` 类进行数据处理的结果，右侧箱线图(黄绿色)展示了使用 `hydrodataset.CamelsUS` 类的结果。两种方法都达到了 0.72 的中位数 NSE，证明了在 671 个流域上稳健且一致的性能表现。

详细的训练结果、方法说明和全面的性能分析，请参阅:
- **[LSTM 在 CAMELS-US 上的训练结果](docs/examples/lstm_camels_results.md)**: 完整的实验文档，包括模型配置、训练过程和流域级别的性能统计。

## 探索更多功能

以上示例涵盖了两个主要用例，但 `torchhydro` 的功能远不止于此，它原生支持多种模型、数据集和数据源。您可以通过我们完整的公共 API 文档来探索所有可用的组件:

- **[模型 API](docs/api/models.md)**: 发现所有可用的模型架构。
- **[数据集 API](docs/api/datasets.md)**: 查看所有的数据集类、数据源和采样器。
- **[训练器 API](docs/api/trainers.md)**: 理解核心的训练和评估流程。

值得注意的能力包括:LSTM 家族与 Seq2Seq/Transformer 模型;可微分物理模型(XAJ/HBV/GR4J);图神经网络(GCN/GAT/STGCN/WDNE)与三维洪水场次模型;MoE(专家混合)模型;Diffusion 与 FNO 模型;云端 Zarr 懒加载以支持远程数据联合训练;月度与亚日时间单位。

我们正持续努力扩展文档，以包含更多示例。

## 主要模块

项目由几个关键模块组成:

- **Trainers(训练器)**: 管理端到端的训练和评估流程。核心的 `DeepHydro` 类处理数据加载、模型初始化、训练循环和评估。其设计具有可扩展性，以支持如迁移学习或多任务学习等多种学习范式。
- **Models(模型)**: 包含所有可用的模型架构，包括标准神经网络(如 LSTM)和可微分模型。一个中央字典使得通过配置来选择模型和损失函数变得简单。
- **Datasets(数据集)**: 提供数据处理能力。标准数据集通过由 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 与 [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) 支撑的统一数据 resolver 打开;`torchhydro.datasets.data_sources` 是向数据集与 PyTorch `Dataset` 类暴露注册表的薄层。
- **Configs(配置)**: 管理所有实验配置，包括模型、数据(时间周期、变量)、训练(轮次、批量大小)和评估的设置。`torchhydro.configs.data_resolver` 集中负责数据集路径/URI 解析。

## 为什么选择 Torchhydro?

尽管已有像 [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) 这样成熟的工具，但 `torchhydro` 的开发基于一套不同的架构理念:

1.  **解耦的数据源**: 我们认为数据处理，特别是对于复杂或私有数据集，需要一个独立的抽象层。我们的方法是让所有标准数据集都通过统一 resolver(hydrodataset/hydrodatasource)先管理数据访问，然后再将其送入 PyTorch `Dataset`。这种模块化设计促进了代码复用，并使得数据源工具可以在没有深度学习模型的情况下被使用。
2.  **灵活的学习范式**: 该框架从一开始就明确设计为不仅支持标准的监督学习，还支持更复杂的模式，如迁移学习、多任务学习和联邦学习。
3.  **深度配置**: 我们对流程的许多方面提供了细粒度的控制，包括数据遍历、归一化方法、批处理采样策略和高级的 dropout 技术，从而为实验提供了更大的灵活性。
4.  **可扩展性**: 核心设计原则是尽可能地将配置外部化，以实现不同数据源和模型的灵活匹配与调用。

## 其他信息

本软件包的灵感来源于:

- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [hydroDL](https://github.com/mhpi/hydroDL)

本软件包是使用 [Cookiecutter](https://github.com/cookiecutter/cookiecutter) 和 [giswqs/pypackage](https://github.com/giswqs/pypackage) 项目模板创建的。
