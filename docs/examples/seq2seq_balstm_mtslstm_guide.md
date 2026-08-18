# Seq2Seq / BALSTM / MTSLSTM 实验案例

> 洪水场次预测与多站点时间序列模型

---

## 1. 模型对比总览

| 模型 | 实验脚本 | 输入 | 输出 | 数据要求 | 适用场景 |
|------|----------|------|------|----------|----------|
| **Seq2Seq** | `train_seq2seq_floodevent.py` | 编码器: 降雨 (rain)<br>解码器: 降雨 + 上一步输出 | 径流 (inflow) | floodeventdatasource<br>3h 时段, 松辽流域 | 洪水场次预测, 编码器-解码器结构 |
| **SimpleLSTM** | `train_simplelstm_floodevent.py` | 降雨 (rain) | 径流 (inflow) | floodeventdatasource<br>3h 时段, 松辽流域 | 洪水场次基线, 对比 Seq2Seq |
| **SimpleBALSTM** | `train_with_balstm.py` | 动态 + 静态 + 全球变量 | 径流 (streamflow) | longtermdataset<br>GRDC 全球月尺度 | 多源输入, 月尺度径流模拟 |
| **MTSLSTM** | `train_with_mtslstm.py` | 多时间尺度动态 + 静态 | 径流 (qobs_mm_per_hour) | camelsh (CAMELS-Hourly)<br>小时级全球数据 | 多时间尺度建模, 日/时/周融合 |

---

## 2. Seq2Seq 洪水场次预测

### Standard Seq2Seq (带预测期降雨输入)

```
降雨 ──→ Encoder (后验期 7天) ──→ hidden state ──→ Decoder (预测期 2天) ──→ 径流
                                ↑
                          预测期降雨作为解码器输入
```

- **脚本**: `experiments/train_seq2seq_floodevent.py`
- **模型**: `"Seq2Seq"` (定义在 `torchhydro/models/seq2seq.py`)
- **数据源**: `floodeventdatasource` (legacy `source_name` 格式)
- **数据集**: `FloodEventSeq2SeqDataset`
- **站点**: `songliao_21401550`
- **输入变量**: `var_t=["rain"]`, `var_c=["None"]`
- **输出变量**: `var_out=["inflow", "flood_event"]`
- **评估方式**: `evaluator={"eval_way": "floodevent"}` (只评估洪水场次)

**关键配置**:

```python
source_cfgs={
    "source_name": "floodeventdatasource",  # 旧格式，尚未迁移到 dataset_id
    "source_path": "E:/data/ClassC/songliaorrevent",  # 实际数据路径
    "other_settings": {
        "time_unit": ["3h"],
        "dataset_name": "songliaorrevent",
        "net_rain_key": "net_rain",
        "obs_flow_key": "inflow",
        "delta_t_hours": 3.0,
    },
},
model_name="Seq2Seq",
model_hyperparam={
    "en_input_size": 1,                    # 编码器输入特征数 (降雨)
    "de_input_size": 2,                    # 解码器输入维度 (降雨 1 + 前一步输出 1)
    "output_size": 1,                      # 输出特征数 (径流)
    "hidden_size": 16,                     # LSTM 隐藏层大小
    "forecast_length": 16,                 # 预测步长 (16 x 3h = 48h = 2天)
    "hindcast_output_window": 56,          # 后验期输出窗口 (56 x 3h = 168h = 7天)
    "teacher_forcing_ratio": 0.5,          # 训练时使用真实值的概率
},
loss_func="FloodLoss",                     # 洪水区间加权 loss
loss_param={
    "loss_func": "MSELoss",
    "flood_weight": 2.0,
    "flood_strategy": "weight",
},
```

**特色功能**:

- **洪峰对齐采样**: `peak_aligned_sampling={"enabled": True, "slide_step": 1, "peak_selection": "first"}` — 从洪峰点开始向后滑动窗口
- **变长序列**: `variable_length_cfgs={"use_variable_length": True}` — 支持不同长度的洪水事件
- **Teacher Forcing**: 训解码器时以 50% 概率使用真实值, 加速收敛
- **FloodLoss**: 对洪水期间的预测误差给予更大权重

### SimpleLSTM 洪水场次 (基线模型)

```
降雨 ──→ LinearIn ──→ LSTM ──→ Dropout ──→ LinearOut ──→ 径流
```

- **脚本**: `experiments/train_simplelstm_floodevent.py`
- **模型**: `"SimpleLSTM"` (定义在 `torchhydro/models/simple_lstm.py`)
- **数据源**: 与 Seq2Seq 相同 (floodeventdatasource)
- **数据集**: `AugmentedFloodEventDataset`

**关键配置**:

```python
source_cfgs={
    "source_name": "floodeventdatasource",  # 旧格式，尚未迁移到 dataset_id
    "source_path": "E:/data/ClassC/songliaorrevent",  # 实际数据路径
    "other_settings": {
        "time_unit": ["3h"],
        "dataset_name": "songliaorrevent",
        "net_rain_key": "net_rain",
        "obs_flow_key": "inflow",
        "delta_t_hours": 3.0,
    },
},
model_name="SimpleLSTM",
model_hyperparam={
    "input_size": 1,       # 输入特征数 (降雨)
    "output_size": 1,      # 输出特征数 (径流)
    "hidden_size": 16,     # LSTM 隐藏层大小
    "dr": 0.3,             # Dropout 比率
},
```

**与 Seq2Seq 的区别**:

| 特性 | Seq2Seq | SimpleLSTM |
|------|---------|------------|
| 结构 | 编码器-解码器, 分阶段处理 | 单向 LSTM, 一次性输出 |
| 预测期降雨 | 作为解码器输入 | 不使用 |
| Teacher Forcing | 支持 (0.5) | 不支持 |
| 洪峰对齐采样 | 启用 | 未启用 |
| 用途 | 高性能预报 | 基线对比 |

**运行方式**:

```bash
# 训练 Seq2Seq
python experiments/train_seq2seq_floodevent.py

# 训练 SimpleLSTM 基线
python experiments/train_simplelstm_floodevent.py
```

---

## 3. BALSTM (Attention-based LSTM)

```
全球变量 (var_g) ──→ W_xg ──→ g_t ──┐
                                     ├──→ LSTM Cell ──→ 输出
静态属性 (var_c) ──→ W_xs1, W_xs2 ──→ i1, i2 (门控)
                                     │
动态变量 (var_t) ──→ W_xt, W_f, W_o ─→ 输入门/遗忘门/输出门
```

- **脚本**: `experiments/train_with_balstm.py`
- **模型**: `"SimpleBALSTM"` (定义在 `torchhydro/models/balstm_model.py`)
- **数据源**: `dataset_id="longtermdataset"` (GRDC 全球月尺度数据)
- **数据集**: `BALSTMDataset`
- **时间尺度**: 月 (`min_time_unit="ME"`)

**关键配置**:

```python
source_cfgs={"dataset_id": "longtermdataset"},
model_name="SimpleBALSTM",
model_hyperparam={
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "input_size_dyn": len(var_t),    # 动态变量数 (从数据自动读取)
    "input_size_glo": len(var_g),    # 全球变量数 (从数据自动读取)
    "input_size_sta": len(var_c),    # 静态属性数 (从数据自动读取)
    "hindcast_output_window": 0,
},
dataset="BALSTMDataset",
min_time_unit="ME",
min_time_interval=1,
which_first_tensor="batch",
```

**三类输入**:

| 输入类型 | 变量名 | 说明 |
|----------|--------|------|
| 动态变量 (`var_t`) | 从 GRDC 时序数据自动读取 | 月尺度气象/水文时序 |
| 静态变量 (`var_c`) | 从 GRDC 属性数据自动读取 | 流域属性 (最多 195 个) |
| 全球变量 (`var_g`) | 从 global_data.csv 读取 | 全球气候指标 |

**模型结构要点**:

- 多层 LSTM, 每层包含 6 个权重矩阵: `W_xs1`, `W_xs2` (静态属性门控), `W_xg` (全球变量融合), `W_xt` (动态输入), `W_f` (遗忘门), `W_o` (输出门)
- 静态属性通过 `i1 = sigmoid(W_xs1(xs))` 和 `i2 = sigmoid(W_xs2(xs))` 作为门控信号, 调节隐藏状态更新
- 全球变量与隐藏状态拼接后通过 `W_xg` 融合

**训练配置**:

```python
batch_size=512,
train_epoch=100,
save_epoch=10,
train_period=["1951-01-01", "2020-12-31"],
test_period=["1982-01-01", "1992-12-31"],
loss_func="NSELoss",
opt="Adam",
lr_scheduler={"lr": 0.0001},
metrics=["NSE", "RMSE", "R2"],
```

**运行方式**:

```bash
python experiments/train_with_balstm.py
```

---

## 4. MTSLSTM (Multi-Timescale LSTM)

```
小时级输入 (23个动态变量 + 27个静态属性)
  │
  ├─→ 周级分支 (weekly):    聚合 7 天 → T_week=2 步
  ├─→ 日级分支 (daily):     聚合 24 小时 → T_day=14 步
  └─→ 小时级分支 (hourly):  保持原始 → T_hour=336 步
                               │
                          频率分支间状态传递 (linear transfer)
                               │
                               ↓
                          合并输出 → 径流预测
```

- **脚本**: `experiments/train_with_mtslstm.py`
- **模型**: `"MTSLSTM"` (定义在 `torchhydro/models/mtslstm.py`)
- **数据源**: `dataset_id="camelsh"` (CAMELS-Hourly)
- **数据集**: `CamelsHourlyDataset`
- **时间尺度**: 小时级 (`min_time_unit="h"`)

**关键配置**:

```python
source_cfgs={"dataset_id": "camelsh"},
model_name="MTSLSTM",
model_hyperparam={
    "hidden_sizes": [128, 128, 128],      # 三个频率分支的隐藏维度
    "output_size": 1,
    "shared_mtslstm": False,               # 各频率分支独立权重
    "transfer": "linear",                  # 频率分支间状态传递方式
    "dropout": 0.1,
    "return_all": True,
    "feature_buckets": feature_buckets,    # 每个特征对应的频率桶
    "per_feature_aggs_map": per_feature_aggs_map,  # 每个特征的聚合方式
    "frequency_factors": [7, 24],          # 频率因子: 周→日 x7, 日→时 x24
    "seq_lengths": [2, 14, 336],           # 各频率序列长度: 周2步, 日14步, 时336步
    "slice_transfer": True,                # NeuralHydrology 风格切片传递
    "slice_use_ceil": True,
    # 可选: 预训练日级分支权重
    "pretrained_day_path": "best_model.pth",
    "pretrained_lstm_prefix": "lstm.",
    "pretrained_head_prefix": "linearOut.",
},
dataset="CamelsHourlyDataset",
```

**23 个动态变量**:

| 变量 | 说明 | 频率桶 | 聚合方式 |
|------|------|:------:|----------|
| convective_fraction | 对流降水比例 | 小时 | mean |
| longwave_radiation | 长波辐射 | 日 | mean |
| potential_energy | 位能 | 周 | mean |
| potential_evaporation | 潜在蒸散发 | 日 | mean |
| pressure | 气压 | 日 | mean |
| shortwave_radiation | 短波辐射 | 日 | mean |
| specific_humidity | 比湿 | 小时 | mean |
| temperature | 温度 | 小时 | mean |
| total_precipitation | 总降水量 | 小时 | sum |
| wind_u | U 风分量 | 小时 | mean |
| wind_v | V 风分量 | 小时 | mean |

**27 个静态属性**: elev_mean, slope_mean, area_gages2, frac_forest, lai_max, lai_diff, dom_land_cover_frac, dom_land_cover, root_depth_50, soil_depth_statsgo, soil_porosity, soil_conductivity, max_water_content, geol_1st_class, geol_2nd_class, geol_porostiy, geol_permeability 等 (均在小时级分支, 不做下采样)。

**训练配置**:

```python
batch_size=256,
train_epoch=20,
save_epoch=1,
rolling=0,                              # 滑动窗口模式
forecast_length=336,                    # 预测 14 天 (336 小时)
loss_func="RMSESum",
opt="Adam",
opt_param={"lr": 1e-4},
lr_scheduler={"lr_factor": 0.95},
metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
early_stopping=True,
patience=2,
```

**运行方式**:

```bash
python experiments/train_with_mtslstm.py
```

---

## 5. 快速开始

### Seq2Seq / SimpleLSTM 洪水场次预测

```bash
# 1. 确保数据存在
# floodeventdatasource 数据位于: E:/data/ClassC/songliaorrevent

# 2. 训练 Seq2Seq
python experiments/train_seq2seq_floodevent.py

# 3. 训练 SimpleLSTM 基线
python experiments/train_simplelstm_floodevent.py

# 4. 查看结果
# 结果保存在 results/flood_event_Seq2Seq/ 和 results/flood_event_SimpleLSTM/
```

### BALSTM 全球月尺度径流模拟

```bash
# 1. 准备 GRDC 数据
# longtermdataset 需要数据预处理 (属性/时序/全球变量 CSV)

# 2. 训练
python experiments/train_with_balstm.py

# 3. 查看结果
# 结果保存在 results/train_with_LongTerm780/
```

### MTSLSTM 多时间尺度预测

```bash
# 1. 确保 CAMELS-Hourly 数据可访问
# camelsh 通过 dataset_id="camelsh" 自动解析路径

# 2. 训练
python experiments/train_with_mtslstm.py

# 3. 查看结果
# 结果保存在 results/camels/mtslstm_DapengScaler_h-unified_3freq_14dwin/
```

---

## 6. 数据要求

| 实验 | 数据源 | 数据位置 | 说明 |
|------|--------|----------|------|
| Seq2Seq / SimpleLSTM | `floodeventdatasource` | `E:/data/ClassC/songliaorrevent` | 松辽流域 3h 洪水场次数据, 使用 legacy `source_name` 格式 |
| BALSTM | `longtermdataset` (dataset_id) | GRDC 全球月尺度数据, 含属性/时序/全球变量 CSV | 需要预处理为 `attributes/`, `timeseries/1MS/`, `global_data.csv` 格式 |
| MTSLSTM | `camelsh` (dataset_id) | CAMELS-Hourly | 小时级全球流域数据, 通过 hydrodataset 统一路径解析 |

**注意**: Seq2Seq 和 SimpleLSTM 的洪水场次实验仍使用 legacy `source_name` 格式 (非 `dataset_id`), 因为 `floodeventdatasource` 是 torchhydro 特有的数据源, 尚未迁移到 hydrodatasource 统一注册表。其余模型 (BALSTM, MTSLSTM) 均使用 `dataset_id` 格式。

---

## 7. MCLSTM（多时间尺度卷积 LSTM）

MCLSTM 通过多尺度卷积核捕获不同时间尺度的径流响应模式。

### 7.1 CAMELS-US 训练

```bash
uv run python experiments/train_with_mclstm.py
```

- 模型：`MCLSTM`
- 数据：CAMELS-US（`dataset_id: "camels_us"`）
- 输入：2 个动态变量
- 输出：1 个目标变量（streamflow）
- 隐藏层：128

### 7.2 长序列训练

```bash
uv run python experiments/train_mclstm_longseq.py
```

- 模型：`MCLSTM`
- 数据：自建数据（`dataset_id: "selfmadehydrodataset"`）
- 特点：支持更长的输入序列

### 7.3 MTSLSTM 三峡评估

```bash
uv run python experiments/evaluate_with_mtslstm.py
```

- 模型：`MTSLSTM`
- 数据：自建数据（`dataset_id: "selfmadehydrodataset"`）
- 用途：评估 MTSLSTM 在三峡流域的表现

---

## 8. MCLSTM Encoder-Decoder 变体

```bash
uv run python experiments/train_with_Enc_Dec_BALSTM.py
```

- 模型：`SimpleBALSTM_EncDec`（或 `Attn_SimpleBALSTM_EncDec`）
- 数据：longtermdataset（GRDC 全球月尺度）
- 特点：编码器-解码器架构的 BALSTM，支持注意力机制
