# WDNE 自适应图网络模型使用指南

## 概述

本项目集成了来自 LamaH-CE 的 WDNE (Water Dynamics Network Encoding) 系列图网络模型，并与现有的 SpatioTemporalFloodNet3D (Baseline) 进行对比。

### 可用模型

| 模型名 | 配置文件 key | 说明 |
|--------|-------------|------|
| **Baseline LSTM-GNN** | `Baseline_LSTM_GNN` | SpatioTemporalFloodNet3D, LSTM+GCN逐时间步交互 |
| **WDNEInspiredFloodNet** | `WDNEInspiredFloodNet` | 结构感知+局部LSTM+静态GCN图传播 |
| **WDNEAdaptiveRoutingNet** | `WDNEAdaptiveRoutingNet` | 在Inspired基础上增加自适应边权重路由 |
| **WDNE_NoStruct_Mean** | `WDNE_NoStruct_Mean` | (实验) 无结构嵌入+mean pool |
| **WDNE_Mean** | `WDNE_Mean` | (实验) 有结构嵌入+mean pool |

---

## 快速开始

### 1. 训练

```bash
# 训练 WDNE 自适应路由模型（推荐）
python experiments/train_with_wdne.py WDNEAdaptiveRoutingNet normal

# 训练 Baseline 对比模型
python experiments/train_with_wdne.py Baseline_LSTM_GNN normal

# 训练 WDNE 静态GCN版本
python experiments/train_with_wdne.py WDNEInspiredFloodNet normal

# 训练全部模型
python experiments/train_with_wdne.py all normal
```

### 2. 绘制对比图

```bash
# 逐洪水事件对比图（降雨+观测+双模型预测，m³/s）
python plot_flood_baseline_vs_wdne.py

# 指定单个流域
python plot_flood_baseline_vs_wdne.py songliao_21401550
```

输出位置：`results/comparison_plots/flood_events/<basin_id>/`

---

## 模型架构对比

### Baseline LSTM-GNN (SpatioTemporalFloodNet3D)

```
输入: [N, 84, 1]  每个站点84步降雨
  │
  ├─ LSTM → [N, 84, hidden]  保留全部时间步
  │
  ├─ 逐时间步 GCN:  t=0..83 各做一次空间卷积
  │   84次空间交互
  │
  ├─ Mean Pool → [B, 84, hidden]
  ├─ 时间映射 84→12
  └─ 输出: [B, 12, 1]
```

特点：密集时空交互（84次GCN），简单平均池化，训练稳定。

### WDNE (Adaptive Routing / Inspired)

```
输入: [N, 84, 1]
  │
  ├─ 拓扑特征计算 (in/out_degree, upstream_span, 等)
  ├─ 结构嵌入: 拓扑+属性 → structure_emb [N, 16]
  │
  ├─ 局部LSTM → local_seq [N, 84, hidden]
  │   └─ 取最后12步 → local_head逐步预测 → base_pred [N, 12, 1]
  │
  ├─ 图传播 (Inspired: GCN, Adaptive: 动态边权重)
  │   └─ graph_state → 传播修正 → graph_correction [N, 12, 1]
  │
  └─ 输出: base_pred + correction → [B, 12, 1]
```

特点：结构感知（知道谁在上游谁在下游），自适应路由（暴雨时加大上游权重），取出口节点。

---

## 关键超参数

```python
model_hyperparam = {
    # === 通用 ===
    "in_channels": 1,            # 动态特征数 (站点DRP=1)
    "output_size": 1,            # 输出变量数 (inflow=1)
    "seq_len": 84,               # 输入序列长度 (warmup+forecast)
    "output_time": 12,           # 预测步数
    "hidden_channels": 32,       # 隐藏层维度
    "dropout": 0.2,              # Dropout比例

    # === WDNE 专属 ===
    "model_type": "wdne_adaptive",  # "wdne" 或 "wdne_adaptive"
    "num_static_features": 0,       # x末尾静态特征数 (var_c)
    "structure_embedding_dim": 16,  # 结构嵌入维度
    "route_steps": 2,               # 图传播层数 (信息跳数)
    "route_decay": 0.7,             # 传播衰减系数
    "graph_output_mode": "residual", # 输出模式: residual/blend_residual/main
    "root_output_mode": "root",     # 输出提取: root(出口)/mean(平均)
    "structure_fusion": "gated",    # 结构融合: gated/mean
    "local_lstm_layers": 1,         # 局部LSTM层数
    "topology_encoder_type": "mlp", # 拓扑编码器: mlp/gcn
}
```

---

## 数据说明

### 当前数据集

- 松辽流域 6 个 basin，每个 basin 4~20 个站点
- 时间分辨率：3小时
- 输入特征：站点级降雨 (DRP)，仅洪水事件期间有值
- 目标：流域出口 inflow (mm/3h)
- 时间划分：Train 1980-2014, Valid 2015-2018, Test 2019-2024

### 当前是模拟（hindcast），不是预测

`var_t=["rain"]` 在整个时间窗口内可见（包括 forecast horizon），模型在做"已知未来降雨→推演径流"的模拟，而非真正盲预测。若要改为预测，需遮掉未来降雨或替换为气象预报值。

### 已知限制

1. 站点 DRP 数据稀疏（仅洪水事件期间有值，其他时刻 NaN→0）
2. 无站点级独立径流观测（只有流域出口 inflow）
3. 无流域静态属性（area、slope 等在 DapengScaler 归一化后不可用）
4. 6 个流域数据量较小

---

## 输出结果解读

### 评估指标

训练完成后在 `results/<experiment_dir>/songliao_3h_normal/` 下：
- `metric_inflow.csv` — 每个 basin 的 NSE/RMSE/KGE
- `metric_flood_event.csv` — flood_event 列的指标
- `epochbest_model.pthflow_obs.nc` — 观测流量 (去归一化, mm/3h)
- `epochbest_model.pthflow_pred.nc` — 预测流量 (去归一化, mm/3h)

### 洪水事件图

每张图包含：
- 上轴：降雨柱状图 (mm/3h)
- 下轴：观测流量 (m³/s, 黑线) + Baseline预测 (蓝虚线) + WDNE预测 (橙点划线)
- 红色半透明区域：洪水事件段
- 图例中显示 NSE 指标

---

## 添加新模型

在 `experiments/train_with_wdne.py` 中：

```python
model_configs = {
    "YourNewModel": {
        "model_name": "YourModelClassName",  # pytorch_model_dict中的key
        "experiment_dir": "your_experiment_dir",
        "description": "模型描述",
        "model_type": "wdne_adaptive",  # 告诉_build_model_hyperparam用哪组参数
        # 超参数...
    },
}
```

然后在 `valid_models` 列表中加上 `"YourNewModel"`。

---

## 文件结构

```
torchhydro/
├── models/
│   ├── adaptive_gnn.py              # WDNE适配器 (tensor→Data对象)
│   ├── wdne_inspired_flood_net.py   # WDNE基础模型 (静态GCN)
│   ├── wdne_adaptive_routing_net.py # WDNE自适应路由 (动态边权重)
│   └── gated_gat.py                 # 门控GAT层
├── experiments/
│   └── train_with_wdne.py           # 训练脚本 (含Baseline对比)
├── plot_flood_baseline_vs_wdne.py   # 洪水事件绘图 (m³/s)
└── docs/
    └── WDNE_MODEL_USAGE.md          # 本文档
```

---

## 常见问题

**Q: WDNE 为什么不如 Baseline？**

A: 当前数据（6流域、仅降雨、站点数据稀疏）不足以支撑WDNE的复杂架构。
Baseline的密集时空交互(84次GCN)在有限数据下更有效。WDNE在更大、特征更丰富的数据集上
预期会更好。

**Q: 如何切换预测/模拟模式？**

A: 当前是模拟模式（有未来降雨）。若要真正预测：
- 方案A: 在 `var_t` 中去掉 `rain`，只用历史降雨 + 气象预报
- 方案B: 在数据集层将 forecast horizon 内的 rain 置零

**Q: 如何添加流域静态属性？**

A: 将 `var_c=["None"]` 改为实际属性名（如 `var_c=["area","elev_mean"]`），
并设 `use_basin_features=True`，`num_static_features` 相应增加。

---

*最后更新: 2026-06-15*
