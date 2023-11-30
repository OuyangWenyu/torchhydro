<!--
 * @Author: Wenyu Ouyang
 * @Date: 2022-05-28 17:46:32
 * @LastEditTime: 2023-11-27 17:49:09
 * @LastEditors: Wenyu Ouyang
 * @Description: README for torchhydro
 * @FilePath: \torchhydro\README.md
 * Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
-->
# torchhydro


[![image](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![image](https://img.shields.io/conda/vn/conda-forge/torchhydro.svg)](https://anaconda.org/conda-forge/torchhydro)


**datasets, samplers, transforms, and pre-trained models for hydrology and water resources**


-   Free software: BSD license
-   Documentation: https://OuyangWenyu.github.io/torchhydro  

**NOTE: THIS REPOSITORY IS **STILL UNDER **DEVELOPMENT**!!!****  

## Features

-   TODO

## For developers

To install the environment, run the following code in the terminal:

```Shell
conda env create -f env-dev.yml
conda activate torchhydro
```

To use this repository of dev or other branches in your existing environment:

1. you can fork it to your GitHub account. Don't choose "only fork the main branch" when forking in the Github page.
2. run the following code in the terminal:

```Shell
# xxxxxx is your github account; here we choose to use dev branch
pip install git+ssh://git@github.com/xxxxxx/torchhydro.git@dev
```

For the dataset we set a unified data path in settings.txt in the `.hydrodataset` directory which is located in the user directory (for example, `C:\Users\username\.hydrodataset` in Windows). You can change the data path in this file.

Then we have some conventions for the dataset:

1. Public datasets such as CAMELS is put in the `waterism/datasets-origin` directory.
2. The processed datasets are put in the `waterism/datasets-interim` directory.

You can specify by yourself, but some changes are needed. We will optimize this part in the future.

## Credits

This package is inspired by [TorchGeo](https://torchgeo.readthedocs.io/en/stable/).

It was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
