<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-13 18:29:19
 * @LastEditTime: 2024-04-14 09:12:47
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

📜 [中文文档](README.zh.md)

**Note: This repository is still under development**

## Installation

We provide a pip package for installation:

```Shell
pip install torchhydro
```

If you want to participate in the development as a developer, you can install the environment and download the code using the following method:

```Shell
# fork this repository to your GitHub account -- xxxx
git clone git@github.com:xxxx/torchhydro.git
cd torchhydro
# If you find it slow, you can install with mamba
# conda install mamba -c conda-forge
# mamba env create -f env-dev.yml
conda env create -f env-dev.yml
conda activate torchhydro
```

## Usage

Currently, we provide an example of training an LSTM on the CAMELS dataset. The functions for reading CAMELS are all written in [hydrodataset](https://github.com/OuyangWenyu/hydrodataset), so first read its readme to download the data properly and place it in the specified folder path. Regarding the folder configuration, check if there is a hydro_setting.yml file in your user directory. If not, manually create one, and refer to [here](https://github.com/OuyangWenyu/torchhydro/blob/6aec414d99e35f4f1672903eb9e18e8eebeadb09/torchhydro/__init__.py#L34) to ensure the local_data_path is set correctly. If you can't download the CAMELS data, you can directly use a version we uploaded on Kaggle: [kaggle CAMELS](https://www.kaggle.com/datasets/headwater/camels)

Then you can try running the files under the experiments folder, such as:

```Shell
cd experiments
python run_camelslstm_experiments.py
```

More tutorials will be added gradually.

## Main Modules

The program mainly includes trainers, models, datasets, and configs, with an additional explainer responsible for the model interpretation part.

- **Trainers**: Designed to handle various modes, the main one being a DeepHydro class, found in the deep_hydro module (a .py file). This class configures its data sources, obtains configurations about the model, data, training, and testing (details here), and then initializes the model (load_model function), the data (make_dataset function), and performs training (model_train function) and testing (model_evaluate function). Transfer learning, multitask learning, and federated learning modes will inherit this class and rewrite specific execution code.
- **Models**: Mainly declared through a model_dict, which shows which models are available for configuration. This includes the selection of loss, and then the remaining model modules like lstm or differentiable models with coupled physical mechanisms.
- **Datasets**: First, we set up several datasource repository tools to provide data sources, including the public dataset [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) (like CAMELS) and [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) (which requires organizing data by oneself). These data sources mainly provide data access, and in torchhydro, specific torch datasets can be written to match the model's data type. The dataset also has a dict to record, and then specific dataset class modules.
- **Configs**: This mainly involves overall configurations, which are loaded during the initialization of the DeepHydro class. It's contained in the config module, primarily encompassing four parts: model (currently mode and model together), data (use of data time range, modeling object, etc.), training (training epochs, batch size, etc.), and testing (performance metrics).

## Why Torchhydro?

Although there are relatively mature tools like [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), we chose not to use it directly for several reasons:
1. Our model-building mode is not limited to fixed datasets corresponding to a fixed Dataset and then connecting to the model. We believe that the data source, especially considering non-public data situations like in China, is very complex and requires a separate Datasource module to handle the data sources and then make a torch Dataset. This extra layer of abstraction makes code reuse easier. Moreover, not everyone requires deep learning, so having a separate Datasource module allows more hydrologists to use it. We created [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) and [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) for this reason.
2. Deep learning modes are not limited to single-variable supervised learning of runoff. Commonly used modes include transfer learning, multitask learning, and federated learning. These modes may use the same specific models as conventional ones, but the program expression will differ significantly, requiring these different modes to be considered in the overall program design.
3. Sometimes, extra configuration is needed for data traversal, normalization methods, data sampling during batch generation, and dropout functionality during model training, necessitating a more flexible design compatible with different specific settings.
4. For historical reasons, we developed torchhydro independently and in parallel from the beginning, so it has continued as such. The main idea is to extend configuration outwardly as much as possible to achieve flexible matching and calling of data and models.

## Additional Information

This package was inspired by:

- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/).
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [hydroDL](https://github.com/mhpi/hydroDL)

This package was created using the [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
