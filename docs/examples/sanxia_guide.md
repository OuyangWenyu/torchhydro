# 三峡流域多站点实验案例

> 基于三峡流域的 LSTM / MTSLSTM 多站点洪水预报实验

---

## 1. 实验总览

| 脚本 | 模型 | 数据集 | 站点数 | 时间尺度 | 特点 |
|------|------|--------|:------:|----------|------|
| `train_lstm_with_sanxia.py` | SimpleLSTM | StreamflowDataset | 5 | 日尺度 | 基线模型，5站点 |
| `train_lstm_with_sanxia_copy.py` | SimpleLSTM | StreamflowDataset | 5 | 日尺度 | 基线副本，不同数据路径 |
| `train_mtslstm_with_sanxia.py` | MTSLSTM | StreamflowDataset | 11 | 日+小时 | 多时间尺度，预训练迁移 |
| `train_mtslstm_with_sanxia_all.py` | MTSLSTM | StreamflowDataset | 51 | 日+小时 | 全站点超参搜索 |
| `train_mtslstm_with_sanxia_all_floodEvent.py` | MTSLSTM | FloodEventDataset | 223 | 日+小时 | 洪水场次，FloodLoss |
| `train_mtslstm_with_sanxia_all_floodEvent_nosc.py` | MTSLSTM | FloodEventDataset | 38 | 日+小时 | 洪水场次，无预计算Scaler |
| `train_mtslstm_with_sanxia_all_longseq.py` | MTSLSTM | StreamflowDataset | 223 | 日+小时 | 长序列推理，仅评估 |
| `train_mtslstm_with_sanxia_all_scaler.py` | MTSLSTM | StreamflowDataset | 51 | 日+小时 | DapengScaler 对比实验 |

---

## 2. SimpleLSTM 三峡基线

**脚本**: `experiments/train_lstm_with_sanxia.py`、`train_lstm_with_sanxia_copy.py`

这是三峡流域的最简单基线实验，使用标准 LSTM 在日尺度上进行流量预报。

### 模型配置

```python
model_name = "SimpleLSTM"
model_hyperparam = {
    "input_size": 39,      # 4 动态 + 35 静态
    "output_size": 1,      # 流量
    "hidden_size": 128,
    "dr": 0.4,             # Dropout
}
```

### 输入特征（39 维）

**动态变量（4 个）**:

| 变量 | 说明 |
|------|------|
| `precipitation_obs` | 降水观测 |
| `temperature_2m` | 2米气温 |
| `surface_net_solar_radiation_hourly` | 地表净太阳辐射 |
| `snowfall_hourly` | 降雪量 |

**静态变量（35 个）**: 包括流域面积（`area`）、年均PET、年均AET、平均高程、湖泊覆盖率、森林覆盖率、坡度、流域体积、侵蚀系数、年均温、河道长度、GDP、土壤有机碳、喀斯特比例、人口密度、夜光指数、永久耕地、永久草地、沙地比例、人口、土壤含水、河道面积、粉土比例、粘土比例、作物比例、不透水面、气候指数、积雪比例、干旱指数、灌溉比例、可再生水资源、地下水补给比例、城市化比例、永久冰川等。

### 训练设置

```python
train_period = ["2021-01-01", "2023-12-31"]
valid_period = ["2022-01-01", "2022-12-31"]
test_period  = ["2024-01-01", "2024-12-31"]
batch_size   = 256
train_epoch  = 20
forecast_length = 336   # 14天（小时）
min_time_unit = "h"
scaler = "DapengScaler"
loss_func = "RMSESum"
opt = "Adam"
opt_param = {"lr": 0.0001}
lr_scheduler = {"lr_factor": 0.95}
metrics = ["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"]
early_stopping = True
patience = 2
```

### 站点列表（5 站）

```
sanxia_60406350, sanxia_60406500, sanxia_60406700, sanxia_60407100, sanxia_60407200
```

### 两个脚本的区别

| 项目 | `train_lstm_with_sanxia.py` | `train_lstm_with_sanxia_copy.py` |
|------|----------------------------|----------------------------------|
| 数据路径 | `/Volumes/Untitled/data/sanxia` | `/home/mahaoran/data/` |
| 数据集名称 | 未指定 | `era5hengjiangqujian_test` |
| `time_unit` | `["1h"]` | `["1D"]` |

### 运行

```bash
python experiments/train_lstm_with_sanxia.py
```

---

## 3. MTSLSTM 三峡基线

**脚本**: `experiments/train_mtslstm_with_sanxia.py`

使用多时间尺度 LSTM（MTSLSTM），支持日尺度和小时尺度双分支输入。需要预训练的日尺度 SimpleLSTM 权重。

### 模型配置

```python
model_name = "MTSLSTM"
model_hyperparam = {
    "pretrained_flag": True,
    "linear1_size": 41,
    "linear2_size": 128,
    "pretrained_day_path": ".../best_model.pth",
    "pretrained_lstm_prefix": "lstm.",
    "pretrained_head_prefix": "linearOut.",
    "hidden_sizes": [128, 128],
    "output_size": 1,
    "shared_mtslstm": False,
    "transfer": "linear",
    "dropout": 0.4,
    "return_all": True,
    "feature_buckets": feature_buckets,
    "per_feature_aggs_map": per_feature_aggs_map,
    "frequency_factors": [24],
    "seq_lengths": [14, 14 * 24],
    "slice_transfer": True,
    "slice_use_ceil": True,
}
```

### 多时间尺度设计

MTSLSTM 的核心是将不同频率的特征分配到不同分支：

| 分支 | 频率 | 序列长度 | 包含变量 |
|------|------|----------|----------|
| 日尺度（低频） | 1天 | 14 步 | `surface_net_solar_radiation_hourly`、`snowfall_hourly` |
| 小时尺度（高频） | 1小时 | 336 步 | `precipitation_obs`、`temperature_2m` |
| 静态变量 | - | - | 全部 35 个（`feature_buckets_sta = [1] * 35`，放在小时层） |

`frequency_factors = [24]` 表示日-小时之间的频率倍率为 24。

### 站点列表（11 站）

```
sanxia_60406350, sanxia_60406500, sanxia_60406700, sanxia_60407100, sanxia_60407200,
sanxia_60701101, sanxia_60711600, sanxia_60713400, sanxia_60713630, sanxia_60713800,
sanxia_60717050
```

### 预训练迁移

此脚本加载日尺度 SimpleLSTM 的预训练权重（`pretrained_day_path`），通过线性映射（`transfer: "linear"`）迁移到 MTSLSTM 的日尺度分支，小时尺度分支从头训练。

### 运行

```bash
python experiments/train_mtslstm_with_sanxia.py
```

---

## 4. MTSLSTM 全站点超参搜索

**脚本**: `experiments/train_mtslstm_with_sanxia_all.py`

在 51 个站点上进行超参数搜索，遍历学习率、批次大小、训练轮数和学习率衰减因子的笛卡尔积。

### 超参搜索空间

```python
FIXED_HIDDEN_SIZES = [128, 128]
LRS = [2e-3, 5e-4]
BATCH_SIZES = [256, 128, 64]
EPOCHS = [20]
LR_FACTORS = [0.9, 0.95, 0.98]
# 总实验数：2 x 3 x 1 x 3 = 18 组
```

### 站点来源

站点从 CSV 文件 `data/basin_51/new_51.csv` 读取（51 个站点）。

### 训练周期

```python
train_period = ["2014-01-01", "2022-12-31"]
valid_period = ["2023-01-01", "2024-12-31"]
test_period  = ["2023-01-01", "2024-12-31"]
```

### 运行

```bash
python experiments/train_mtslstm_with_sanxia_all.py
```

脚本会依次执行 18 组实验，每组输出实验编号和对应超参数。

---

## 5. MTSLSTM 洪水场次实验

**脚本**: `experiments/train_mtslstm_with_sanxia_all_floodEvent.py`、`train_mtslstm_with_sanxia_all_floodEvent_nosc.py`

使用 `FloodEventDataset` 和 `FloodLoss` 进行洪水场次级别的训练和评估。

### 与普通训练的差异

| 项目 | 普通 MTSLSTM | 洪水场次 MTSLSTM |
|------|-------------|-----------------|
| 数据集 | `StreamflowDataset` | `FloodEventDataset` |
| 损失函数 | `RMSESum` | `FloodLoss` |
| 评估方式 | 默认滑窗 | `evaluator={"eval_way": "floodevent"}` |
| 输出目标 | `["streamflow"]` | `["streamflow", "flood_event"]` |
| 序列结构 | `forecast_length` | `hindcast_length` + `forecast_length` + `warmup_length` |
| `return_all` | `True` | `False` |

### 序列结构（洪水场次）

```
|<-- warmup_length -->|<-- hindcast_length -->|<-- forecast_length -->|
|       720 步         |        72 步           |       24 步            |
|      (模型预热)      |     (历史观测)         |     (预报期)           |
```

### 特征分配

与 MTSLSTM 基线不同，洪水场次实验将静态变量放在**日尺度分支**（`feature_buckets_sta = [0] * 35`），聚合方式也有所调整：

```python
agg_map_dyn = {
    "precipitation_obs": "sum",       # 降水用 sum 聚合
    "temperature_2m": "mean",
    "surface_net_solar_radiation_hourly": "mean",
    "snowfall_hourly": "sum",         # 降雪量用 sum 聚合
}
```

### 两个脚本的区别

| 项目 | `_floodEvent.py` | `_floodEvent_nosc.py` |
|------|-------------------|----------------------|
| 站点数 | 223（从 `76_floodevents.csv` 读取） | 38（从 `4_38.csv` 读取） |
| 预训练权重 | `best_model2.pth` | `best_model3.pth` |
| `stat_dict_file` | 指定预计算的 `dapengscaler_stat.json` | 不指定（自动计算） |
| 数据集名称 | `223_FloodEvents_76_Interpolation` | `223_FloodEvents_38_repick` |
| hindcast_length | 72 | 30 |
| forecast_length | 24 | 72 |
| `use_nan_safe_stats` | 未设置 | `True` |
| `dropout` | 0.1 | 0.1 |
| 超参搜索 | lr=[1e-3], batch=[128] | lr=[1e-3], batch=[36] |

### 运行

```bash
python experiments/train_mtslstm_with_sanxia_all_floodEvent.py
python experiments/train_mtslstm_with_sanxia_all_floodEvent_nosc.py
```

---

## 6. MTSLSTM 长序列推理

**脚本**: `experiments/train_mtslstm_with_sanxia_all_longseq.py`

加载已训练的 MTSLSTM 模型，仅在测试集上执行推理评估，不进行训练。

### 关键配置

```python
train_mode = False              # 不训练，仅推理
model_loader = {"load_way": "pth", "pth_path": PRETRAIN_DAY_PTH}
evaluator = {"eval_way": "once", "stride": 0}
warmup_length = 0               # 无预热期
```

### 数据集

使用 `223_FloodEvents_105_final` 数据集，站点列表从 `data/223_floodevents.csv` 读取。

### 序列结构

```
|<-- hindcast_length -->|<-- forecast_length -->|
|        72 步           |       24 步            |
```

### 运行

```bash
python experiments/train_mtslstm_with_sanxia_all_longseq.py
```

环境变量可覆盖默认路径：

```bash
PRETRAIN_DAY_PTH=/path/to/model.pth \
STAT_DICT_FILE=/path/to/stat.json \
SANXIA_DATA_DIR=/path/to/data/ \
BATCH_SIZE=1 \
python experiments/train_mtslstm_with_sanxia_all_longseq.py
```

---

## 7. MTSLSTM Scaler 对比实验

**脚本**: `experiments/train_mtslstm_with_sanxia_all_scaler.py`

与全站点超参搜索结构相同，但使用预计算的 DapengScaler 统计量文件，用于对比不同 Scaler 策略的影响。

### 与 `_all.py` 的区别

| 项目 | `_all.py` | `_all_scaler.py` |
|------|-----------|-------------------|
| Scaler 统计量 | 自动计算 | 指定 `pth_file/dapengscaler_stat.json` |
| 输出目录 | `newbasin_51_smoothed_5_non_fix/` | `newbasin_51_smoothed_5_sc_new/` |

其余超参搜索空间、站点列表、模型配置完全一致。

### 运行

```bash
python experiments/train_mtslstm_with_sanxia_all_scaler.py
```

---

## 8. 快速开始

### 前置条件

1. 安装 torchhydro 及依赖：

```bash
cd torchhydro
uv sync --all-extras
```

2. 确保 `hydro_setting.yml` 配置正确（见项目 README）。

3. 准备三峡流域数据（见下节数据要求）。

4. 下载预训练权重（MTSLSTM 实验需要）。

### 最小运行示例

运行 5 站点 SimpleLSTM 基线：

```bash
python experiments/train_lstm_with_sanxia.py
```

### 修改站点

编辑脚本中的 `gage_id` 列表，或修改 CSV 文件路径以加载不同的站点集合。

### 修改数据路径

每个脚本顶部有 `sanxia_dir` 变量，修改为实际的数据存放路径即可。

---

## 9. 数据要求

所有三峡实验均使用 `selfmadehydrodataset` 数据源，数据不包含在仓库中。

### 数据来源

- **数据源类型**: `selfmadehydrodataset`（通过 `hydrodatasource` 的 `SelfMadeHydroDataset` 读取）
- **数据格式**: NetCDF 或 CSV，按站点组织
- **时间分辨率**: 小时级（`1h`）

### 站点命名规范

所有站点 ID 以 `sanxia_` 为前缀，后接 8 位数字编码，例如：

```
sanxia_60406350
sanxia_60717050
```

### 数据目录结构

各脚本中的 `sanxia_dir` 指向数据根目录，实际路径因运行环境而异：

| 脚本 | 默认数据路径 |
|------|-------------|
| `train_lstm_with_sanxia.py` | `/Volumes/Untitled/data/sanxia` |
| `train_lstm_with_sanxia_copy.py` | `/home/mahaoran/data/` |
| `train_mtslstm_with_sanxia.py` | `/Volumes/Untitled/data` |
| `train_mtslstm_with_sanxia_all.py` | `/home/litianli/Dataset/newbasin_223_smoothed/` |
| `train_mtslstm_with_sanxia_all_floodEvent.py` | `/home/litianli/Dataset/` |
| `train_mtslstm_with_sanxia_all_floodEvent_nosc.py` | `/home/litianli/Dataset/` |
| `train_mtslstm_with_sanxia_all_longseq.py` | `/home/litianli/Dataset/`（可通过 `SANXIA_DATA_DIR` 环境变量覆盖） |
| `train_mtslstm_with_sanxia_all_scaler.py` | `/home/litianli/Dataset/newbasin_223_smoothed/` |

### 预训练权重

MTSLSTM 实验依赖预训练的 SimpleLSTM 日尺度权重（`.pth` 文件）。权重通过脚本中的 `PRETRAIN_DAY_PTH` 变量指定路径。

### 配套 CSV 文件

部分脚本从 CSV 文件读取站点列表：

| CSV 路径 | 使用脚本 | 站点数 |
|----------|----------|:------:|
| `data/basin_51/new_51.csv` | `_all.py`、`_all_scaler.py` | 51 |
| `data/76_floodevents.csv` | `_floodEvent.py` | 223 |
| `data/4_38.csv` | `_floodEvent_nosc.py` | 38 |
| `data/223_floodevents.csv` | `_longseq.py` | 223 |
