<!--
 * @Author: Wenyu Ouyang
 * @LastEditTime: 2026-08-18
 * @Description: Chinese version of the README
-->
# TorchHydro

[![PyPI version](https://img.shields.io/pypi/v/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![Python](https://img.shields.io/pypi/pyversions/torchhydro.svg)](https://pypi.python.org/pypi/torchhydro)
[![License](https://img.shields.io/github/license/OuyangWenyu/torchhydro.svg)](https://github.com/OuyangWenyu/torchhydro/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://OuyangWenyu.github.io/torchhydro)

一个基于 PyTorch 的水文建模深度学习框架，支持常规、迁移学习、多任务学习与联邦学习等模式。

**文档**: https://OuyangWenyu.github.io/torchhydro

## 安装

需要 **Python 3.10+**。

```shell
pip install torchhydro
```

或使用更快的 [uv](https://docs.astral.sh/uv/):

```shell
uv pip install torchhydro
```

开发者环境配置请参阅 [安装指南](https://OuyangWenyu.github.io/torchhydro/installation/)。

## 快速开始

### 1. 配置数据路径

在用户主目录(`~/` 或 `%USERPROFILE%`)下创建 `hydro_setting.yml`:

```yaml
storage:
  default_source: local
  local:
    root: 'D:/data/hydrodatasets'  # 您的数据目录
  cache: 'D:/data/hydrodatasets/cache'
```

标准数据集(如 CAMELS-US)的路径由统一数据 resolver 从该 root 自动解析。

### 2. 运行模型

```python
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 构建 LSTM on CAMELS-US 的配置
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

`examples/` 目录提供了可直接运行的脚本:
- **`examples/lstm_camels_example.py`** — 在 CAMELS-US 上训练标准 LSTM
- **`examples/dpl_xaj_example.py`** — 可微分新安江(XAJ)水文模型

## 功能特性

### 模型

| 类别 | 模型 |
|---|---|
| **LSTM 变体** | SimpleLSTM, CudaLSTM, CNNLSTM, MCLSTM, SPPLSTM, BALSTM, MTSLSTM |
| **Seq2Seq / 编码器-解码器** | GeneralSeq2Seq, Transformer, DataFusionModel, SeqForecast |
| **可微分物理模型** | DPL-XAJ, DPL-HBV, DPL-GR4J, DPL-MC-水库 |
| **图神经网络** | GCN, ResGCN, GCNII, ResGAT, GCGRU, GCLSTM, STGCN |
| **专家混合 (MoE)** | DenseMoE, SparseMoE, SwitchMoE, HydrologyTopKMoE, GlobalTopKMoE |
| **生成式 / 谱方法** | Diffusion, FNO (Fourier Neural Operator) |
| **专用模型** | WDNE (三维洪水), CoupledLSTM, RegulLSTM |

### 数据集

由 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 和 [hydrodatasource](https://github.com/iHeadWater/hydrodatasource) 解析的标准数据集:

| `dataset_id` | 说明 |
|---|---|
| `camels_us` | CAMELS-US(美国 671 个流域) |
| `camelsh` | CAMELS-Hourly(小时尺度) |
| `caravan` / `grdc_caravan` | Caravan / GRDC-Caravan 数据集集合 |
| `camels_aus` / `camels_br` / `camels_cl` / `camels_gb` / `camels_col` | CAMELS 区域数据集 |

自定义数据源: `selfmadehydrodataset`、`selfmadeforecastdataset`、`longtermdataset`、`floodeventdatasource`、`stationhydrodataset`、`tghydrodatasource`。

### 训练模式

- **常规监督学习** — 单流域或区域训练
- **迁移学习** — 跨流域模型适配
- **多任务学习** — 共享骨干网络,任务专属输出头
- **联邦学习** — 去中心化流域级训练

### 关键能力

- 统一数据 resolver — 所有数据集路径由一个 `hydro_setting.yml` 配置
- 支持月度与亚日(小时)时间单位
- Cloud-Zarr 懒加载,支持远程数据联合训练
- Lightning Fabric 集成,支持调试与分布式训练
- SHAP 模型可解释性与损失面可视化
- 高级 dropout 策略与数据增强

## 指南与文档

### 示例与结果

- [CAMELS-US 上的 LSTM](https://OuyangWenyu.github.io/torchhydro/examples/lstm_camels_results/) — 基准测试与流域级性能
- [松辽流域洪水模型](https://OuyangWenyu.github.io/torchhydro/examples/songliao_flood_guide/) — LSTM, MCLSTM, GNN, WDNE 洪水预报
- [三峡多站点](https://OuyangWenyu.github.io/torchhydro/examples/sanxia_guide/) — 水库与多站点实验
- [MoE / Diffusion / FNO](https://OuyangWenyu.github.io/torchhydro/examples/moe_diffusion_fno_guide/) — 高级模型实验
- [Seq2Seq / BALSTM / MTSLSTM](https://OuyangWenyu.github.io/torchhydro/examples/seq2seq_balstm_mtslstm_guide/) — 编码器-解码器与时空模型

### 高级主题

- [Cloud-Zarr 联合训练](https://OuyangWenyu.github.io/torchhydro/advanced/cloud_zarr_lazy_loading/) — 大规模远程数据懒加载
- [Lightning Fabric](https://OuyangWenyu.github.io/torchhydro/advanced/fabric_debug_guide/) — 调试与分布式训练
- [ERA5-Land 与 GPM](https://OuyangWenyu.github.io/torchhydro/advanced/era5land_gpm_guide/) — 格网气象数据实验
- [模型可解释性](https://OuyangWenyu.github.io/torchhydro/advanced/model_interpretability/) — SHAP 与损失面分析

### API 参考

[模型](https://OuyangWenyu.github.io/torchhydro/api/models/) · [数据集](https://OuyangWenyu.github.io/torchhydro/api/datasets/) · [训练器](https://OuyangWenyu.github.io/torchhydro/api/trainers/) · [配置](https://OuyangWenyu.github.io/torchhydro/api/configs/) · [可解释性](https://OuyangWenyu.github.io/torchhydro/api/explainers/)

## 架构

```
configs/          config.py, model_config.py, data_resolver.py(统一数据层)
  │ 驱动
  ├── datasets/   data_sets.py, data_sources.py, sampler.py, scalers.py
  ├── models/     20+ 种架构, 注册于 model_dict_function.py
  ├── trainers/   deep_hydro.py(编排器), trainer.py, fabric_wrapper.py
  ├── explainers/ shap.py, loss_landscape/, uncertainty_analysis.py
  └── utils/      dist_utils.py(MPI 分布式计算)
```

## 为什么选择 TorchHydro?

- **解耦的数据层** — 统一 resolver 支持 10+ 数据集、自定义数据源与云存储;数据处理工具甚至可以不依赖深度学习模型单独使用
- **灵活的学习范式** — 迁移学习、多任务学习、联邦学习是一等公民,而非事后添加
- **深度配置** — 对数据遍历、归一化、批采样与高级 dropout 提供细粒度控制
- **可扩展** — 配置高度外部化,新数据源与新模型无需修改核心代码即可接入

## 参与贡献

请参阅 [贡献指南](https://OuyangWenyu.github.io/torchhydro/contributing/) 了解开发环境配置、代码规范与 Pull Request 流程。

## 许可证

BSD License。详见 [LICENSE](LICENSE)。

## 致谢

- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [TorchGeo](https://torchgeo.readthedocs.io/en/stable/)
- [hydroDL](https://github.com/mhpi/hydroDL)
