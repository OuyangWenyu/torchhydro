# 松辽流域洪水预报模型使用指南

> 基于 `experiments/train_with_wdne.py` 配置文件

---

## 1. 四模型对比总览

| 模型 | 文件 | 图结构 | 站点数据 | 适用场景 |
|------|------|:---:|:---:|------|
| **Basin LSTM** | `train_basin_lstm.py` | 无 | 不需要 | 只有流域平均降雨，无站点数据 |
| **LSTM-only** | `train_with_wdne.py` → `Baseline_LSTM_only` | 无 | 需要 | 有站点但图结构不明确 |
| **LSTM+GNN** | `train_with_wdne.py` → `Baseline_LSTM_GNN` | 静态GCN | 需要 | 密集时空交互，稳定首选 |
| **WDNE** | `train_with_wdne.py` → `WDNEAdaptiveRoutingNet` | 自适应路由 | 需要 | 有静态属性+复杂河网 |

---

## 2. 模型架构详解

### 2.1 Basin LSTM — 最简单基线

```
流域平均降雨 → LSTM → 输出
```

- **输入**: 流域级 `rain` (CSV文件中的basin平均降雨)
- **不使用站点数据**, 不使用图结构
- **数据集**: `FloodEventDataset`，`source_name="selfmadehydrodataset"`
- **模型**: `SimpleLSTM`
- **运行**: `python experiments/train_basin_lstm.py`
- **优点**: 最简单，1.5s/epoch，不需要图和站点
- **缺点**: 无法利用空间信息

### 2.2 LSTM-only — 站点级无图

```
站点DRP [N,84,1] → LSTM → [N,84,hidden] → mean_pool → [B,12,1]
```

- **输入**: 站点级 DRP (每个站点独立)
- **无图结构**, 所有站点独立 LSTM 编码后平均池化
- **模型**: `SpatioTemporalFloodNet3D`, `use_gnn=False, use_lstm=True`
- **优点**: 利用了站点数据，比 Basin LSTM 信息更丰富

### 2.3 LSTM+GNN — 密集时空交互 (当前最优基线)

```
站点DRP [N,84,1]
  → LSTM → [N,84,H]
  → 每个时间步独立GCN (84次空间交互)
  → mean_pool → [B,12,1]
```

- **图结构**: 静态 GCNConv，每个时间步做一次（84次/序列）
- **边方向**: `downstream` = 上游→下游（原始 `ID→NEXTDOWNID`）
- **模型**: `SpatioTemporalFloodNet3D`, `use_gnn=True, use_lstm=True`
- **关键参数**: `num_hidden=2` (GCN层数), `gnn_type="gcn2"`, `aggregation="mean"`
- **优点**: 密集时空交互，最稳定（avg NSE 0.58）

### 2.4 WDNE — 自适应路由图网络

```
站点DRP [N,84,1]
  → 拓扑特征计算 (入度/出度/上游深度/到出口距离)
  → 结构嵌入 [N,16]
  → LSTM逐步 → base_prediction [N,12,1]
  → 自适应图传播 (route_steps=4次) → graph_correction
  → output = base_prediction + correction
  → root_output: 取出度=0的节点(出口)预测
```

- **核心创新**: 
  1. 拓扑编码: 每个节点的河流角色被编码为向量
  2. 自适应路由: 边权重不是固定的，由当前气象条件+节点状态动态计算
  3. 门控更新: sigmoid gate 控制新旧信息比例
  4. 结构分离: LSTM做基础预测, 图传播做修正
- **模型**: `WDNEAdaptiveRoutingNet` (继承自 `WDNEInspiredFloodNet`)
- **优点**: 加静态属性后 avg NSE 0.64, 21401550达0.89
- **注意**: 需要 route_steps≥图深度才能让上游信息传到出口

---

## 3. 配置文件说明

### 3.1 当前配置 (`train_with_wdne.py`)

#### 必须根据流域修改的参数:

```python
# ===== 数据 =====
gage_id = ["songliao_20800900", ...]       # 流域ID列表
train_period = ["1980-01-01-02", "2014-12-31-23"]  # 训练时段
valid_period = ["2015-01-01-02", "2018-12-31-23"]  # 验证时段
test_period  = ["2019-01-01-02", "2024-12-31-23"]  # 测试时段

# ===== 特征 =====
var_t = ["rain"]                            # 时序输入变量
var_c = ["None"]                            # 静态属性(实验2用)
var_out = ["inflow", "flood_event"]         # 输出变量
station_cols = ["DRP"]                      # 站点级变量

# ===== 图结构 =====
edge_orientation = "downstream"             # 边方向: downstream=上→下
adjacency_src_col = "ID"                    # 邻接矩阵源列
adjacency_dst_col = "NEXTDOWNID"            # 邻接矩阵目标列
adjacency_weight_col = None                 # 边权重列(None=等权重)

# ===== 时间窗口 =====
warmup_length = 72      # 预热步数 (72×3h=216h=9天)
hindcast_length = 0     # 回望步数
forecast_length = 12    # 预测步数 (12×3h=36h)
```

#### 模型超参数:

```python
# Baseline (LSTM-only / LSTM+GNN)
hidden_channels = 32   # LSTM隐藏维度
num_hidden = 2         # GNN层数
gnn_type = "gcn2"      # GNN类型: gcn/gat/gcn2
aggregation = "mean"   # 节点聚合: mean/attention/multiscale

# WDNE 专属
route_steps = 4        # 图传播步数 (应 ≥ 流域最大深度)
route_decay = 0.5      # 远距离衰减系数 (0-1, 越小越激进)
route_residual_scale = 1.0  # 图修正权重
structure_embedding_dim = 16  # 结构嵌入维度
topology_encoder_type = "mlp"  # 拓扑编码器: mlp/gcn
graph_output_mode = "residual"  # 输出模式: residual/blend_residual/main
root_output_mode = "root"       # 提取模式: root(出口)/mean(平均)
local_lstm_layers = 1   # 局部LSTM层数
dropout = 0.3           # Dropout比例
```

### 3.2 根据流域实际情况修改指南

#### 衰减系数 `route_decay` 的设置

`route_decay` 控制信息每跳衰减多少：`decay^hops` = 剩余信号比例。

| 河网最大深度 | 建议 decay | 最深节点衰减 | 说明 |
|:---:|:---:|:---:|------|
| 2-3跳 | 0.5 | 0.5²=25% | 浅河网,低衰减足够 |
| 4-5跳 | 0.6-0.7 | 0.7⁴=24% | 中等深度 |
| 6-7跳 | 0.7-0.8 | 0.8⁶=26% | 深河网,需高衰减系数 |
| 8跳以上 | 0.8-0.85 | 0.85⁸=27% | 非常深的河网 |

**原则**: 最上游节点到出口的信号剩余应≥20-30%，否则上游信息传不到。

#### 其他关键参数

| 流域特征 | 修改参数 | 建议值 | 原因 |
|------|------|:---:|------|
| **河网深度大** | `route_steps` | ≥深度 | 保证上游信息传到底 |
| **河网深度小** | `route_steps` | 2-3, decay可低 | 避免过拟合 |
| **站点数多** (>50) | `hidden_channels` | 64-128 | 更大容量 |
| **站点数少** (<10) | `hidden_channels` | 16-32 | 防过拟合 |
| **有静态属性** | `var_c` + `use_basin_features` | 设为True | WDNE可利用 |
| **无站点数据** | 使用Basin LSTM | — | 不需要图结构 |
| **边方向不确定** | `edge_orientation` | `downstream` | 上→下, 当前默认 |
| **训练数据少** | `dropout` + `patience` | 0.3-0.5, 15 | 强正则化 |
| **预报时长不同** | `forecast_length` | 自定义 | 输出长度对应 |

---

## 3.3 四种模型分别如何配置

### Basin LSTM — 不需要图和站点

```python
# 直接运行
python experiments/train_basin_lstm.py

# 核心配置 (train_basin_lstm.py 中)
model_name = "SimpleLSTM"
dataset = "FloodEventDataset"
var_t = ["rain"]        # 流域平均降雨
var_c = ["None"]        # 不加静态属性
source_name = "selfmadehydrodataset"  # 数据源
batch_size = 8
model_hyperparam = {
    "input_size": 1,     # 降雨特征数
    "output_size": 1,    # 输出径流
    "hidden_size": 64,   # LSTM隐藏维度
}
```

**何时选择**: 没有站点数据，只有流域平均降雨。最快，1.5s/epoch。

### LSTM-only — 有站点，不要图

```python
python experiments/train_with_wdne.py Baseline_LSTM_only normal

# 配置 (train_with_wdne.py 中)
"Baseline_LSTM_only": {
    "model_name": "SpatioTemporalFloodNet3D",
    "model_type": "baseline",
    "hidden_channels": 32,   # LSTM隐藏维度
    "use_gnn": False,        # 关闭GNN ← 关键!
    "use_lstm": True,
    "aggregation": "mean",   # 所有站点平均池化
}
```

**何时选择**: 有站点数据但河网结构不清晰，或者简单流域不需要空间交互。

### LSTM+GNN — 密集时空交互(当前最优)

```python
python experiments/train_with_wdne.py Baseline_LSTM_GNN normal

# 配置
"Baseline_LSTM_GNN": {
    "model_name": "SpatioTemporalFloodNet3D",
    "model_type": "baseline",
    "hidden_channels": 32,
    "use_gnn": True,         # 开启GNN ← 关键!
    "use_lstm": True,
    "num_hidden": 2,         # GCN层数
    "gnn_type": "gcn2",      # GCN/GAT/GCN2
    "aggregation": "mean",   # 聚合方式
    "edge_orientation": "downstream",  # 上→下
}
```

**何时选择**: 有站点+河网，需要空间交互但不需要复杂路由。**默认首选**。

### WDNE — 自适应路由

```python
python experiments/train_with_wdne.py WDNEAdaptiveRoutingNet normal

# 配置
"WDNEAdaptiveRoutingNet": {
    "model_name": "WDNEAdaptiveRoutingNet",
    "model_type": "wdne_adaptive",
    "hidden_channels": 32,
    "structure_embedding_dim": 16,   # 结构嵌入维度
    "route_steps": 4,                # 图传播步数 ← 根据流域深度!
    "route_decay": 0.5,              # 衰减系数 ← 根据流域深度!
    "graph_output_mode": "residual", # residual/blend_residual/main
    "root_output_mode": "root",      # root(出口)/mean(平均)
    "topology_encoder_type": "mlp",  # mlp/gcn
    "local_lstm_layers": 1,
    "dropout": 0.3,
}
```

**何时选择**: 有静态属性+复杂河网+多流域，自适应路由能利用流域特征差异。

**route_steps + route_decay 联动设置**:
```python
# 浅河网 (2-3跳)
route_steps = 3
route_decay = 0.5

# 中等河网 (4-5跳)
route_steps = 5
route_decay = 0.7

# 深河网 (6-8跳)  
route_steps = 8
route_decay = 0.8
```

**加静态属性的WDNE**:
```python
python experiments/train_with_wdne.py WDNE_static normal
# 自动使用5个流域 + 5个静态属性字段
# attribute_encoder 会把静态属性编码到结构嵌入中
```

```python
# 在 train_with_wdne.py 中
use_static = True
var_c = ["ele_mt_sav", "slp_dg_sav", "for_pc_sse", "pre_mm_syr", "run_mm_syr"]
use_basin_features = True
num_static_features = len(var_c)  # = 5
in_channels = num_features + len(var_t)  # DRP + rain
# num_static_features 会自动传给 WDNE 的 attribute_encoder
```

可用静态属性（从 `attributes/attributes.csv`）:
- `area` — 面积 (km²)
- `ele_mt_sav` — 平均海拔 (m)
- `slp_dg_sav` — 平均坡度 (°)
- `for_pc_sse` — 森林覆盖率 (%)
- `pre_mm_syr` — 年均降水 (mm)
- `run_mm_syr` — 年均径流 (mm)
- `pet_mm_syr` — 年均潜在蒸散发 (mm)
- `soc_th_sav` — 土壤有机碳
- `cly_pc_sav` — 黏土含量

### 3.4 单流域训练

```python
# 方法1: 命令行参数
python experiments/train_with_wdne.py WDNEAdaptiveRoutingNet normal --gage_id songliao_21401550

# 方法2: 修改配置
example_gage_ids = ["songliao_21401550"]  # 只保留一个
```

### 3.5 空间敏感性测试

```python
# 1. 定义流域拓扑 (data_sets.py WATERSHED_TOPOLOGY)
'21401550': {
    'upstream_stations': ['21422650'],    # 上游站点
    'midstream_stations': ['21401500'],    # 中游站点
    'downstream_stations': ['21401550'],   # 下游站点(出口)
}

# 2. 运行测试
python experiments/eval_spatial.py
# 生成 normal/sim1/sim2/sim3 四种情景的预测结果
```

---

## 4. 快速开始

```bash
# 训练所有模型
python experiments/train_with_wdne.py Baseline_LSTM_only normal
python experiments/train_with_wdne.py Baseline_LSTM_GNN normal
python experiments/train_with_wdne.py WDNEAdaptiveRoutingNet normal

# 加静态属性
python experiments/train_with_wdne.py LSTM_only_static normal
python experiments/train_with_wdne.py LSTM_GNN_static normal
python experiments/train_with_wdne.py WDNE_static normal

# 查看结果
cat results/{模型名}/songliao_3h_normal/metric_inflow.csv

# 绘图
python plot_all_final.py
```

## 5. 当前已知限制

1. **站点DRP高度相关**: 同流域所有站点降雨几乎一致，限制自适应路由发挥
2. **6个流域偏少**: WDNE需要更多样化的河网来泛化
3. **21100150深度7**: route_steps=4不够，需针对性调参
4. **无历史径流输入**: 纯降雨驱动，可考虑加入lag inflow
5. **Basin LSTM未随edge fix更新**: 无图结构，不受影响
