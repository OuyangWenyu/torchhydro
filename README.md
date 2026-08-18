<!--
 * @Author: Wenyu Ouyang
 * @LastEditTime: 2026-08-18
 * @Description: English version of the README
-->
# TorchHydro

[![PyPI version](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![Python](https://img.shields.io/pypi/pyversions/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![License](https://img.shields.io/github/license/OuyangWenyu/torchhydro.svg)](https://github.com/OuyangWenyu/torchhydro/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://OuyangWenyu.github.io/torchhydro)

A PyTorch-based deep learning framework for hydrological modeling. Supports Normal, Transfer Learning, Multi-Task Learning, and Federated Learning modes.

**Documentation**: https://OuyangWenyu.github.io/torchhydro

## Installation

**Python 3.10+** required.

```shell
pip install torchhydro
```

or with [uv](https://docs.astral.sh/uv/) (faster):

```shell
uv pip install torchhydro
```

See the [Installation Guide](https://OuyangWenyu.github.io/torchhydro/installation/) for developer setup.

## Quick Start

### 1. Configure data path

Create `hydro_setting.yml` in your home directory (`~/` or `%USERPROFILE%`):

```yaml
storage:
  default_source: local
  local:
    root: 'D:/data/hydrodatasets'  # your data directory
  cache: 'D:/data/hydrodatasets/cache'
```

Standard dataset paths (e.g., CAMELS-US) are resolved automatically from this root by the unified data resolver.

### 2. Run a model

```python
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# Build config for LSTM on CAMELS-US
args = cmd(
    source_cfgs={"dataset_id": "camels_us"},
    model_name="CpuLSTM",
    model_hyperparam={"n_input_features": 23, "n_output_features": 1, "n_hidden_states": 256},
    gage_id=["01013500", "01022500", "01030500"],
    train_period=["2000-10-01", "2001-10-01"],
    valid_period=["2001-10-01", "2002-10-01"],
    test_period=["2002-10-01", "2003-10-01"],
    var_t=[
        StandardVariable.PRECIPITATION,
        StandardVariable.SOLAR_RADIATION,
        StandardVariable.TEMPERATURE_MAX,
    ],
    var_out=[StandardVariable.STREAMFLOW],
    dataset="StreamflowDataset",
    sampler="KuaiSampler",
    scaler="DapengScaler",
)

config_data = default_config_file()
update_cfg(config_data, args)
train_and_evaluate(config_data)
```

See `examples/` for runnable scripts:
- **`examples/lstm_camels_example.py`** — Standard LSTM on CAMELS-US
- **`examples/dpl_xaj_example.py`** — Differentiable Xinanjiang (XAJ) model

## Features

### Models

| Category | Models |
|---|---|
| **LSTM variants** | SimpleLSTM, CudaLSTM, CNNLSTM, MCLSTM, SPPLSTM, BALSTM, MTSLSTM |
| **Seq2Seq / Encoder-Decoder** | GeneralSeq2Seq, Transformer, DataFusionModel, SeqForecast |
| **Differentiable physical models** | DPL-XAJ, DPL-HBV, DPL-GR4J, DPL-MC-Reservoir |
| **Graph neural networks** | GCN, ResGCN, GCNII, ResGAT, GCGRU, GCLSTM, STGCN |
| **Mixture of Experts** | DenseMoE, SparseMoE, SwitchMoE, HydrologyTopKMoE, GlobalTopKMoE |
| **Generative / spectral** | Diffusion, FNO (Fourier Neural Operator) |
| **Specialized** | WDNE (3D flood), CoupledLSTM, RegulLSTM |

### Datasets

Standard datasets resolved via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) and [hydrodatasource](https://github.com/iHeadWater/hydrodatasource):

| `dataset_id` | Description |
|---|---|
| `camels_us` | CAMELS-US (671 US basins) |
| `camelsh` | CAMELS-Hourly |
| `caravan` / `grdc_caravan` | Caravan / GRDC-Caravan collections |
| `camels_aus` / `camels_br` / `camels_cl` / `camels_gb` / `camels_col` | CAMELS regional datasets |

Custom data sources: `selfmadehydrodataset`, `selfmadeforecastdataset`, `longtermdataset`, `floodeventdatasource`, `stationhydrodataset`, `tghydrodatasource`.

### Training modes

- **Standard supervised learning** — single basin or regional training
- **Transfer learning** — cross-basin model adaptation
- **Multi-task learning** — shared backbone, task-specific heads
- **Federated learning** — decentralized basin-level training

### Key capabilities

- Unified data resolver — all dataset paths from one `hydro_setting.yml` config
- Monthly and sub-daily (hourly) time units
- Cloud-Zarr lazy loading for joint training on remote data
- Lightning Fabric integration for debugging and distributed training
- SHAP-based model interpretability and loss landscape visualization
- Advanced dropout strategies and data augmentation

## Guides & Documentation

### Examples & Results

- [LSTM on CAMELS-US](https://OuyangWenyu.github.io/torchhydro/examples/lstm_camels_results/) — baseline benchmark and basin-level performance
- [Songliao Flood Models](https://OuyangWenyu.github.io/torchhydro/examples/songliao_flood_guide/) — LSTM, MCLSTM, GNN, WDNE flood forecasting
- [Sanxia Multi-Site](https://OuyangWenyu.github.io/torchhydro/examples/sanxia_guide/) — reservoir and multi-site experiments
- [MoE / Diffusion / FNO](https://OuyangWenyu.github.io/torchhydro/examples/moe_diffusion_fno_guide/) — advanced model experiments
- [Seq2Seq / BALSTM / MTSLSTM](https://OuyangWenyu.github.io/torchhydro/examples/seq2seq_balstm_mtslstm_guide/) — encoder-decoder and spatial-temporal models

### Advanced Topics

- [Cloud-Zarr Joint Training](https://OuyangWenyu.github.io/torchhydro/advanced/cloud_zarr_lazy_loading/) — lazy loading for large-scale remote data
- [Lightning Fabric](https://OuyangWenyu.github.io/torchhydro/advanced/fabric_debug_guide/) — debugging and distributed training
- [ERA5-Land & GPM](https://OuyangWenyu.github.io/torchhydro/advanced/era5land_gpm_guide/) — gridded meteorological data experiments
- [Model Interpretability](https://OuyangWenyu.github.io/torchhydro/advanced/model_interpretability/) — SHAP and loss landscape analysis

### API Reference

[Models](https://OuyangWenyu.github.io/torchhydro/api/models/) · [Datasets](https://OuyangWenyu.github.io/torchhydro/api/datasets/) · [Trainers](https://OuyangWenyu.github.io/torchhydro/api/trainers/) · [Configs](https://OuyangWenyu.github.io/torchhydro/api/configs/) · [Explainers](https://OuyangWenyu.github.io/torchhydro/api/explainers/)

## Architecture

```
configs/          config.py, model_config.py, data_resolver.py (unified data layer)
  │ drives
  ├── datasets/   data_sets.py, data_sources.py, sampler.py, scalers.py
  ├── models/     20+ architectures registered in model_dict_function.py
  ├── trainers/   deep_hydro.py (orchestrator), trainer.py, fabric_wrapper.py
  ├── explainers/ shap.py, loss_landscape/, uncertainty_analysis.py
  └── utils/      dist_utils.py (MPI distributed computing)
```

## Why TorchHydro?

- **Decoupled data layer** — a unified resolver supports 10+ datasets, custom sources, and cloud storage; data tooling is usable even without a deep-learning model
- **Flexible learning paradigms** — transfer, multi-task, and federated learning are first-class, not afterthoughts
- **Deep configuration** — fine-grained control over data traversal, normalization, batch sampling, and advanced dropout
- **Extensible** — configuration is externalized so new data sources and models plug in without modifying core code

## Contributing

See the [Contributing Guide](https://OuyangWenyu.github.io/torchhydro/contributing/) for development setup, code standards, and pull request workflow.

## License

BSD License. See [LICENSE](LICENSE).

## Acknowledgments

- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- [hydroDL](https://github.com/mhpi/hydroDL)
