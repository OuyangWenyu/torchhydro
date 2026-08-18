# ERA5-Land 与 GPM 数据实验

## 概述

使用 ERA5-Land 再分析数据和 GPM（GPM IMERG）卫星降水数据进行水文预报的实验。两个数据源均通过 `selfmadehydrodataset` 读取，属于自定义数据集，不包含在仓库中，需用户自行准备数据并配置路径。

核心共性：

- 数据源：`selfmadehydrodataset`（需在 `source_cfgs` 中通过 `dataset_id` 或 `source_name` 指定）
- 时间分辨率：小时级（`min_time_unit: "h"`）
- 标准化：`DapengScaler`
- 多输出损失：`MultiOutLoss`（`RMSESum`）
- 模型类型：多任务学习（`model_type: "MTL"`）

## ERA5-Land 实验

ERA5-Land 是 ECMWF 提供的全球陆面再分析数据集，空间分辨率 0.1 度，逐小时输出。可用变量包括逐小时降水、2m 温度、露点温度、地表太阳净辐射、表层土壤含水量等。

### 训练（Seq2Seq）

**脚本：** `experiments/train_with_era5land.py`

使用 Seq2Seq 编码器-解码器模型，基于 ERA5-Land 时序数据和流域静态属性进行径流预报。

关键配置：

| 参数 | 值 |
|------|-----|
| 模型 | `Seq2Seq` |
| `en_input_size` | 16（动态变量 + 静态属性） |
| `de_input_size` | 17 |
| `hidden_size` | 256 |
| `forecast_length` | 56（小时） |
| `hindcast_length` | 240（小时） |
| `batch_size` | 256 |
| `min_time_interval` | 1 |
| `rolling` | 56 |
| `time_unit` | `["1h"]` |
| `model_type` | `MTL` |

时序变量（`var_t`）：

- `total_precipitation_hourly` — 逐小时总降水量

静态属性（`var_c`）：15 个流域特征，包括面积、海拔、坡度、森林覆盖率、土地覆盖类型、径流、淹没范围、气候湿度指数、蒸散发、雪盖、土壤含水量、地下水位深度、黏土含量、调节程度等。

输出目标（`var_out`）：`streamflow`

运行方式：

```bash
uv run python experiments/train_with_era5land.py
```

### 评估

**脚本：** `experiments/evaluate_with_era5land.py`

加载已训练模型进行评估，输出预测结果。评估脚本通过 `weight_path` 和 `stat_dict_file` 指向训练产出的模型权重和标准化统计文件。

关键配置：

| 参数 | 值 |
|------|-----|
| `input_size` | 20 |
| `output_size` | 2 |
| `min_time_interval` | 3 |
| `batch_size` | 1024 |
| `rolling` | 56 |

时序变量（`var_t`）扩展为 5 个：`total_precipitation_hourly`、`temperature_2m`、`dewpoint_temperature_2m`、`surface_net_solar_radiation`、`sm_surface`

输出目标（`var_out`）：`["streamflow", "sm_surface"]` — 同时预报径流和表层土壤含水量

评估期：`["2015-06-01-01", "2016-05-31-01"]`

运行方式：

```bash
uv run python experiments/evaluate_with_era5land.py
```

### GNN + DDP 分布式训练

**脚本：** `experiments/train_with_era5land_gnn_ddp.py`

使用图神经网络（GCN）建模流域拓扑关系，结合站点数据进行分布式训练。

关键配置：

| 参数 | 值 |
|------|-----|
| 模型 | `GCN` |
| `in_channels` | 5 |
| `hidden_channels` | 64 |
| `num_hidden` | 2 |
| `param_sharing` | True |
| `edge_orientation` | `"downstream"` |
| 数据集 | `GNNDataset` |
| `batch_size` | 等于站点数 |
| `hindcast_length` | 168（7 天 x 24 小时） |
| `forecast_length` | 24（1 天） |

站点数据配置（`station_cfgs`）：

```python
station_cfgs = {
    "station_cols": ["temperature", "humidity", "pressure", "wind_speed"],
    "station_rm_nan": True,
    "use_adjacency": True,
    "station_time_units": ["1h"],
    "station_scaler_type": "DapengScaler",
}
```

站点 ID 通过 `gage_id` 参数指定（示例使用 5 个松辽流域站点）。使用时需根据实际数据修改站点列表。

时序变量（`var_t`）：5 个 ERA5-Land 变量（降水、温度、露点、辐射、土壤含水量）

静态属性（`var_c`）：5 个流域特征（面积、海拔、坡度、森林覆盖率、径流）

输出目标（`var_out`）：`streamflow`

损失函数：`RMSESum`

运行方式：

```bash
uv run python experiments/train_with_era5land_gnn_ddp.py
```

### Transformer 迁移学习

**脚本：** `experiments/train_with_era5land_trans.py`

使用 Transformer 模型进行迁移学习实验，利用 ERA5-Land 多变量特征和空间拓扑信息（shp 文件）。

关键配置：

| 参数 | 值 |
|------|-----|
| 模型 | `Transformer` |
| `n_encoder_inputs` | 20 |
| `n_decoder_inputs` | 19 |
| `n_decoder_output` | 2 |
| `channels` | 256 |
| `num_embeddings` | 512 |
| `nhead` | 8 |
| `num_layers` | 4 |
| `dropout` | 0.3 |
| 数据集 | `TransformerDataset` |
| 采样器 | `HydroSampler` |
| `batch_size` | 128 |
| `forecast_length` | 56 |
| `which_first_tensor` | `"sequence"` |
| `train_epoch` | 100 |
| `model_type` | `MTL` |

时序变量（`var_t`）：5 个 ERA5-Land 变量（降水、温度、露点、辐射、土壤含水量）

输出目标（`var_out`）：`["streamflow", "sm_surface"]`

损失函数配置：

```python
loss_param = {
    "loss_funcs": "RMSESum",
    "data_gap": [0, 0],
    "device": [1],
    "item_weight": [0.8, 0.2],  # 径流权重 0.8，土壤含水量权重 0.2
}
```

空间拓扑配置：

- `network_shp`：河网 shapefile 路径
- `node_shp`：节点 shapefile 路径

站点列表通过动态筛选获取：从已有时序文件中提取站点 ID，交集 CAMELS-Hourly USGS 站点，再排除指定的移除站点列表。

运行方式：

```bash
uv run python experiments/train_with_era5land_trans.py
```

## GPM 实验

GPM（Global Precipitation Measurement）IMERG 提供全球卫星降水估计数据，是水文预报中常用的降水输入源。

### 训练

**脚本：** `experiments/train_with_gpm_streamflow.py`

使用 Seq2Seq 模型，基于 GPM 降水数据、表层土壤含水量和上游站点流量进行多步预报。

关键配置：

| 参数 | 值 |
|------|-----|
| 模型 | `Seq2Seq` |
| `input_size` | 18 |
| `output_size` | 2 |
| `hidden_size` | 256 |
| `forecast_length` | 56（小时） |
| `hindcast_length` | 240（小时） |
| `batch_size` | 1024 |
| `min_time_interval` | 3 |
| `time_unit` | `["3h"]` |
| `model_type` | `MTL` |
| `train_epoch` | 100 |
| `sampler` | `BasinBatchSampler` |

时序变量（`var_t`）：

- `gpm_tp` — GPM 总降水量
- `sm_surface` — 表层土壤含水量
- `streamflow` — 上游站点流量

静态属性（`var_c`）：15 个流域特征（与 ERA5-Land 实验相同）

输出目标（`var_out`）：`["streamflow", "sm_surface"]`

损失函数配置：

```python
loss_param = {
    "loss_funcs": "RMSESum",
    "data_gap": [0, 0],
    "device": [1],
    "item_weight": [0.8, 0.2],  # 径流权重 0.8，土壤含水量权重 0.2
}
```

运行方式：

```bash
uv run python experiments/train_with_gpm_streamflow.py
```

### 评估

有两个评估脚本：

**`experiments/evaluate_with_gpm.py`** — 使用 GPM 降水和土壤含水量（不含上游流量）进行评估

关键配置：

| 参数 | 值 |
|------|-----|
| `en_input_size` | 17 |
| `de_input_size` | 18 |
| `output_size` | 2 |
| `min_time_interval` | 3 |

时序变量（`var_t`）：`gpm_tp`、`sm_surface`

评估期：`["2015-06-01-01", "2016-06-01-01"]`

**`experiments/evaluate_with_gpm_streamflow.py`** — 使用 GPM 降水、土壤含水量和上游流量进行评估

关键配置：

| 参数 | 值 |
|------|-----|
| `input_size` | 18 |
| `output_size` | 2 |
| `min_time_interval` | 3 |

时序变量（`var_t`）：`gpm_tp`、`sm_surface`、`streamflow`

评估期：`["2015-06-01-01", "2016-05-31-01"]`

两个脚本均通过 `weight_path` 和 `stat_dict_file` 指向训练产出的文件。

运行方式：

```bash
uv run python experiments/evaluate_with_gpm.py
uv run python experiments/evaluate_with_gpm_streamflow.py
```

## 数据要求

ERA5-Land 和 GPM 数据均为自定义数据集（`selfmadehydrodataset`），不包含在本仓库中。用户需自行准备数据并配置路径。

### 数据源配置

在实验脚本的 `source_cfgs` 中指定：

```python
source_cfgs={
    "dataset_id": "selfmadehydrodataset",
    "source_path": SETTING["local_data_path"]["datasets-interim"],  # 可选，显式指定路径
    "other_settings": {
        "time_unit": ["1h"],  # 或 ["3h"]
    },
}
```

或使用新式 `dataset_id` 格式：

```python
source_cfgs={
    "dataset_id": "selfmadehydrodataset",
    "other_settings": {
        "time_unit": ["3h"],
    },
}
```

数据路径需在 `~/hydro_setting.yml` 的 `storage` 配置中正确设置（参见项目 README）。

### 变量对照表

| 变量名 | 数据源 | 说明 |
|--------|--------|------|
| `total_precipitation_hourly` | ERA5-Land | 逐小时总降水量 |
| `temperature_2m` | ERA5-Land | 2m 高度气温 |
| `dewpoint_temperature_2m` | ERA5-Land | 2m 高度露点温度 |
| `surface_net_solar_radiation` | ERA5-Land | 地表太阳净辐射 |
| `sm_surface` | ERA5-Land | 表层土壤含水量 |
| `gpm_tp` | GPM IMERG | GPM 总降水量 |
| `streamflow` | 自定义 | 上游站点流量（作为输入时） |

### 站点数据

站点列表通过 CSV 文件或动态筛选获取：

- `data/basin_id(46+1).csv` — 47 个松辽流域站点
- `data/basin_id(498+24).csv` — 522 个站点
- `data/basin_id(498+41).csv` — 539 个站点
- 动态筛选：从已有数据文件中自动提取站点 ID，与 CAMELS-Hourly 等站点列表取交集

## 运行注意事项

1. **GPU 内存**：`batch_size` 较大（256/1024），需确保 GPU 显存充足。可适当减小 `batch_size`。
2. **训练时长**：GPM 实验默认 100 个 epoch，ERA5-Land 基础实验默认 3 个 epoch（用于快速验证）。
3. **评估路径**：评估脚本中的 `weight_path` 和 `stat_dict_file` 需指向实际训练产出的文件路径。
4. **站点筛选**：`train_with_era5land_trans.py` 中的站点列表是动态生成的，运行时需确保数据目录存在对应的 CSV 文件。
