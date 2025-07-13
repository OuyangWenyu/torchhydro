<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-13 18:29:19
 * @LastEditTime: 2025-07-13 10:24:29
 * @LastEditors: Wenyu Ouyang
 * @Description: 中文版本的README
 * @FilePath: /torchhydro/README.zh.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# Torchhydro


[![image](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![image](https://img.shields.io/conda/vn/conda-forge/torchhydro.svg)](https://anaconda.org/conda-forge/torchhydro)

- 开源协议: BSD license
- 文档: https://OuyangWenyu.github.io/torchhydro  

**注意：这个仓库还在开发中**

## 安装

我们提供了一个pip包的安装方式

```Shell
pip install torchhydro
```

如果想以开发者的身份一起参与开发，可以使用以下方式安装环境下载运行代码：

```Shell
# fork this repository to your GitHub account -- xxxx
git clone git@github.com:xxxx/torchhydro.git
cd torchhydro
```

### 方式1：使用uv独立虚拟环境（推荐）

```Shell
# 如果尚未安装uv，先安装, 适用于linux和macos
curl -LsSf https://astral.sh/uv/install.sh | sh
# 适用于windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 6
# 安装uv
pip install uv
# 安装项目依赖
# 创建并激活虚拟环境，安装依赖
uv sync --dev
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac系统
# 或者
.venv\Scripts\activate     # Windows系统
```

### 方式2：使用uv配合conda环境

```Shell
# 创建conda环境（需要Python >=3.9）
conda create -n torchhydro python=3.11
conda activate torchhydro
# 在conda环境中安装uv
pip install uv
# 将项目依赖安装到conda环境
uv pip install -e .[dev]
```

**注意**：
- 方式1会在`.venv/`文件夹中创建独立的虚拟环境
- 方式2会直接在conda环境中安装包
- 两种方式功能相同
- 如果需要安装更多pytorch geometric的包，可以使用以下命令：
```Shell
uv pip install torch-scatter torch-sparse torch-cluster \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## 使用

目前我们提供了一个CAMELS数据集下面训练LSTM的示例，读取CAMELS的函数我们都写在 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 里了，所以先阅读它的 readme 把数据都下载好，并且放置到指定的文件夹路径下。关于文件夹的配置，可以查看自己用户目录下是否有 hydro_setting.yml 文件，如果没有的话，就自己手动创建一个，具体的配置参考[这里](https://github.com/OuyangWenyu/torchhydro/blob/6aec414d99e35f4f1672903eb9e18e8eebeadb09/torchhydro/__init__.py#L34)，保障 local_data_path 路径正确即可。CAMELS数据如果下载不下来，可以直接使用我们在Kaggle上上传的一个版本：[kaggle CAMELS](https://www.kaggle.com/datasets/headwater/camels)

然后你就可以尝试运行 experiments 文件夹下的文件了，比如：

```Shell
cd experiments
python run_camelslstm_experiments.py.py
```

更多使用教程，后续我们会逐渐补充。

## 主要模块

程序主要包括 trainers、models、datasets和configs几个方面，另外还额外增加了一个explainer，负责把模型解释部分。

- trainers：设计来应对多种模式，主体是一个 DeepHydro 类，在 deep_hydro 这个module（就是一个.py文件）里面，这个类的主要作用就是配置好它的数据源，获取它关于模型、数据、训练和测试各方面的配置（详见这里），然后根据这些配置初始化模型（load_model函数）、初始化数据（make_dataset函数）、并执行训练（model_train函数）以及测试(model_evaluate函数)。迁移学习、多任务学习、联邦学习模式都会继承这个类并重写具体的执行代码。
- models：模型主要通过一个 model_dict 来做一个简单的声明，通过一个dict的value来展示哪些模型是可以被使用的，这样方便能够进行配置选择，这里也包括loss的选择，然后剩下的就是各个model的module文件，有lstm的，有耦合物理机制的可微分模型。
- datasets：首先，我们设置了几个datasource的仓库工具，来提供数据源，包括公开数据集的[hydrodataset](https://github.com/OuyangWenyu/hydrodataset)（比如CAMELS）、处理自己数据的[hydrodatasource](https://github.com/iHeadWater/hydrodatasource)（不像CAMELS这样做好的数据集，而是需要自己整理的），这些数据源主要提供的功能就是对数据的访问，然后在torchhydro里面就能写具体的torch dataset了，就是按照和模型对接的数据类型来编写dataset，dataset整体也有一个dict 来记录，然后就是具体的dataset类的module了。datasets里面还有一些归一化的、通用处理数据的工具的module
- configs：这部分主要就是一个总体的配置，是被 DeepHydro 类 初始化的时候加载的配置信息，内容在 config module  里面，主要就是四个部分的配置：模型（目前模式和模型在一起）的、数据的（使用数据的时间范围、建模的对象等）、训练的（训练代数epoch、batch 的size等）、测试的（性能指标等）。

后续我们会补充更详细的文档。

## 为什么要做torchhydro

尽管目前是有像[NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)这样相对较成熟的工具，但是我们没有选择直接用它，这主要出于几个方面的考虑：
1. 我们构建模型的模式不只限于 固定数据集对应固定 Dataset，然后再和模型对接，我们认为数据源尤其是考虑了像中国这类不公开数据的情况后，情况会非常复杂，有必要独立地处理数据源，需要一个专门处理Datasource的模块，然后再来做 torch Dataset，这样加一层抽象会更容易使代码复用，另外，不是每个人都要deep learning，所以单独做一个Datasource的模块能让更多水文相关者使用，我们做了 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 和 [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) 就是出于这个考虑
2. 深度学习的模式不局限于径流单一变量的监督学习，比如常用的就有 迁移学习、多任务学习、联邦学习等，这些模式使用的具体模型可能和常规模式是一样的，但是程序表达上会有较大的差别，需要把这些不同模式考虑到整个程序设计中
3. 有时候对数据的遍历、归一化的方式、批次生成时的数据采样、模型训练时的dropout功能 等都需要额外配置，需要一种更灵活兼容不同具体设置的设计
4. 历史原因，最早时候我们就独立并行地开发了torchhydro，所以一直就保持下来了，主体思想主要就是尽量将配置外延，放到最外层，实现数据模型的灵活匹配和调用

## 其他说明

本软件包参考了

- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/).
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [hydroDL](https://github.com/mhpi/hydroDL)

本软件包是使用 [Cookiecutter](https://github.com/cookiecutter/cookiecutter) 和 [giswqs/pypackage](https://github.com/giswqs/pypackage) 项目模板创建的。
