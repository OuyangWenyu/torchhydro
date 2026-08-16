<img alt="image" src="https://github.com/mhpi/dMC-Juniata-hydroDL2/assets/16233925/c93f352c-648d-40cb-bee7-2ee8916d4e89">


# dMC-Juniata-hydroDL2

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)

This repo contains a released version of the differentiable Muskingum-Cunge Method. 

__more documentation will be released upon publication of this paper__

## Installation/Getting Started

1. Create a conda env using the defined `environment.yml` file

```shell
conda env create -f environment.yml
```

2. Download the River graph data from our Zenodo link (Link coming soon)

3. Run an experiment. Your experiments are controlled by config files within the `dMC/conf/configs` dir.

To change the config file, go to `dMC/conf/global_settings.yaml` and make sure to change the experiment `name` as desired
## Experiments

### 01: Single Parameter Experiments
To run these, you should use the following configs:
- `01_generate_single_synth_parameter_data.yaml`
- `01_train_against_single_synthetic.yaml`

### 02: Synthetic Parameter Distribution Recovery

There are many synthetic parameter experiments. Run the following configs to recreate them

#### Synthetic Constants
- `02_generate_mlp_param_list.yaml`
- `02_train_mlp_param_list.yaml`

#### Synthetic Power Law A
- `02_generate_mlp_power_a.yaml`
- `02_train_mlp_power_a.yaml`

#### Synthetic Power Law B
- `02_train_mlp_power_b.yaml`
- `02_generate_mlp_power_b.yaml`

### 03: Train against USGS data:
You can run the following cfgs to train models against USGS data
- `03_train_usgs_period_1a.yaml`
- `03_train_usgs_period_1b.yaml`
- `03_train_usgs_period_2a.yaml`
- `03_train_usgs_period_2b.yaml`
- `03_train_usgs_period_3a.yaml`
- `03_train_usgs_period_3b.yaml`
- `03_train_usgs_period_4a.yaml`
- `03_train_usgs_period_4b.yaml`

##### Jupyter notebooks for generating testing metrics are coming soon. 




