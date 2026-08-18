# MoE / Diffusion / FNO 实验案例

> 基于 CAMELS-US 数据集的进阶模型实验

---

## 1. 模型对比总览

| 模型 | 实验脚本 | 类型 | 专家数 / 特殊结构 | 适用场景 |
|------|----------|------|:---:|------|
| **MoE** | `train_moe_camels.py` | 标准混合专家 | 2 专家 + SimpleAnn 门控 | 快速验证 MoE 框架 |
| **DenseMoE** | `train_densemoe_camels.py` | 密集连接 MoE | 8 专家，专家间全连接 | 需要专家间信息融合 |
| **GlobalTopKMoE** | `train_switchmoe_camels.py` | Top-K 稀疏 MoE | 8 专家，全局 Top-K 门控 | 大规模稀疏激活，节省计算 |
| **HydrologyTopKMoE** | `train_hydrotopkmoe_camels.py` | 领域感知 Top-K MoE | 8 专家，流域属性驱动门控 | 利用 17 个静态属性做条件路由 |
| **LSTMMoE** | `train_lstmmoe_camels.py` | LSTM + MoE 混合 | 16 专家 (SimpleAnn)，序列级 MoE | 序列级专家选择，支持 FSDP 分布式 |
| **ConditionalDiffusionModel** | `train_diffusion_camels.py` | 条件扩散模型 | 1000 步去噪扩散 | 生成式建模，不确定性估计 |
| **FNO** | `train_fno_camels.py` | 傅里叶神经算子 | 4 层频谱层，16 频率模式 | 频域学习，长序列建模 |

---

## 2. MoE 家族

### 2.1 MoE（标准混合专家）

```
输入 [B, T, 23]
  → Expert 1: CpuLSTM → [B, T, 1]
  → Expert 2: CpuLSTM → [B, T, 1]
  → Gating: SimpleAnn(最后一时间步) → 权重 [B, 2]
  → 加权融合 → [B, T, 1]
```

- **模型注册名**: `"MoE"`
- **实验脚本**: `experiments/train_moe_camels.py`
- **关键参数**:
  - `n_experts=2` — 2 个 CpuLSTM 专家
  - `gating_mode="last"` — 使用最后一个时间步驱动门控
  - 专家: `CpuLSTM`，每个 `n_hidden_states=256`
  - 门控: `SimpleAnn`，输入维度 23，隐藏层 `[64, 16]`
- **数据集**: `StreamflowDataset`
- **训练设置**: `batch_size=8, train_epoch=2`（演示用，正式训练请调大）
- **特点**: 最简单的 MoE 基线，适合理解框架用法

```python
# 核心配置
model_name = "MoE"
model_hyperparam = {
    "n_experts": 2,
    "gating_mode": "last",
    "experts_cfgs": [
        {
            "model_name": "CpuLSTM",
            "model_hyperparam": {
                "n_input_features": 23,
                "n_output_features": 1,
                "n_hidden_states": 256,
            },
        }
        for _ in range(2)
    ],
    "gating_cfgs": {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": 23, "hidden_size": [64, 16]},
    },
}
```

### 2.2 DenseMoE

```
输入 [B, T, 23]
  → Expert 1..8: KuaiLSTM × 8
  → 密集连接: 每个专家接收前序专家的输出
  → Gating: SimpleAnn(静态属性 17 维) → 权重
  → 加权融合 → [B, T, 1]
```

- **模型注册名**: `"DenseMoE"`
- **实验脚本**: `experiments/train_densemoe_camels.py`
- **关键参数**:
  - `n_experts=8` — 8 个 KuaiLSTM 专家
  - `static_features=17` — 17 个静态属性作为门控输入
  - 专家: `KuaiLSTM`，每个 `n_hidden_states=256`
  - 门控: `SimpleAnn`，输入维度 17（静态属性），隐藏层 `[64, 16]`
- **数据集**: `FNODataset`
- **训练设置**: `batch_size=20, train_epoch=20`

```python
model_name = "DenseMoE"
model_hyperparam = {
    "n_experts": 8,
    "static_features": 17,
    "experts_cfgs": [
        {
            "model_name": "KuaiLSTM",
            "model_hyperparam": {
                "n_input_features": 23,
                "n_output_features": 1,
                "n_hidden_states": 256,
            },
        }
        for _ in range(8)
    ],
    "gating_cfgs": {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": 17, "hidden_size": [64, 16]},
    },
}
```

### 2.3 GlobalTopKMoE（Switch MoE）

```
输入 [B, T, 23]
  → 8 × CpuLSTM 专家
  → 全局 Top-K 门控 (capacity_factor=1.25)
  → 负载均衡损失 (load_balance_loss_coef=0.02)
  → 稀疏激活: 每个 token 只激活部分专家
```

- **模型注册名**: `"GlobalTopKMoE"`
- **实验脚本**: `experiments/train_switchmoe_camels.py`
- **关键参数**:
  - `n_experts=8` — 8 个 CpuLSTM 专家
  - `static_features=17` — 17 个静态属性
  - `capacity_factor=1.25` — 容量因子，控制缓冲区大小
  - `load_balance_loss_coef=0.02` — 负载均衡损失系数
  - `jitter_noise=0.0` — 无噪声注入
- **数据集**: `StreamflowDataset`
- **训练设置**: `batch_size=256, train_epoch=20`
- **特点**: 全局 Top-K 稀疏门控，每个 token 只选择最相关的专家

```python
model_name = "GlobalTopKMoE"
model_hyperparam = {
    "n_experts": 8,
    "static_features": 17,
    "capacity_factor": 1.25,
    "load_balance_loss_coef": 2e-2,
    "jitter_noise": 0.0,
    "experts_cfgs": [
        {
            "model_name": "CpuLSTM",
            "model_hyperparam": {
                "n_input_features": 23,
                "n_output_features": 1,
                "n_hidden_states": 256,
            },
        }
        for _ in range(8)
    ],
    "gating_cfgs": {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": 17, "hidden_size": [64, 16]},
    },
}
```

### 2.4 HydrologyTopKMoE

```
输入 [B, T, 23] + 静态属性 [B, 17]
  → 8 × CpuLSTM 专家
  → 领域感知 Top-K 门控 (top_k=2, capacity_factor=1.25)
  → 负载均衡损失 (load_balance_loss_coef=0.01)
  → 流域属性驱动的条件路由
```

- **模型注册名**: `"HydrologyTopKMoE"`
- **实验脚本**: `experiments/train_hydrotopkmoe_camels.py`
- **关键参数**:
  - `n_experts=8` — 8 个 CpuLSTM 专家
  - `static_features=17` — 17 个静态属性（气候、地形、土壤等）
  - `top_k=2` — 每个 token 选择 2 个专家
  - `capacity_factor=1.25`
  - `load_balance_loss_coef=1e-2`
- **数据集**: `StreamflowDataset`
- **训练设置**: `batch_size=256, train_epoch=2`
- **特点**: 利用流域静态属性进行条件门控，不同流域可能激活不同专家组合

```python
model_name = "HydrologyTopKMoE"
model_hyperparam = {
    "n_experts": 8,
    "static_features": 17,
    "top_k": 2,
    "capacity_factor": 1.25,
    "load_balance_loss_coef": 1e-2,
    "experts_cfgs": [
        {
            "model_name": "CpuLSTM",
            "model_hyperparam": {
                "n_input_features": 23,
                "n_output_features": 1,
                "n_hidden_states": 256,
            },
        }
        for _ in range(8)
    ],
    "gating_cfgs": {
        "model_name": "SimpleAnn",
        "model_hyperparam": {"nx": 17, "hidden_size": [64, 16]},
    },
}
```

### 2.5 LSTMMoE

```
输入 [B, T, 23]
  → LSTM backbone → 隐藏状态 [B, T, 128]
  → 序列级 MoE (moe_level="sequence")
    → 16 × SimpleAnn 专家
    → Top-2 门控，GELU 激活，RMSNorm
  → 输出 → [B, T, 1]
```

- **模型注册名**: `"LSTMMoE"`
- **实验脚本**: `experiments/train_lstmmoe_camels.py`
- **关键参数**:
  - `n_hidden_states=128` — LSTM 隐藏维度
  - `moe_level="sequence"` — 序列级 MoE（非 token 级）
  - `moe_cfgs.n_experts=16` — 16 个 SimpleAnn 专家
  - `moe_cfgs.top_k=2` — 每次选择 2 个专家
  - `moe_cfgs.gating_mode="last"` — 最后时间步门控
  - `moe_cfgs.capacity_factor=1.0`
  - `moe_cfgs.load_balance_loss_coef=0.01`
  - 专家: `SimpleAnn`，`nx=128, ny=128, hidden_size=[512], dr=0.1, activation="gelu"`
  - 门控: `SimpleAnn`，`nx=128, hidden_size=[128], normalization="RMSNorm"`
  - 支持 FSDP 分布式训练 (`use_ddp=True, use_fsdp=True`)
- **数据集**: `StreamflowDataset`
- **训练设置**: `batch_size=256, train_epoch=20`

```python
model_name = "LSTMMoE"
model_hyperparam = {
    "n_input_features": 23,
    "n_output_features": 1,
    "n_hidden_states": 128,
    "moe_level": "sequence",
    "activation": "gelu",
    "normalization": "rmsnorm",
    "dr": 0.4,
    "moe_cfgs": {
        "n_experts": 16,
        "top_k": 2,
        "gating_mode": "last",
        "capacity_factor": 1.0,
        "load_balance_loss_coef": 0.01,
        "noisy_gating": False,
        "experts_cfgs": [
            {
                "model_name": "SimpleAnn",
                "model_hyperparam": {
                    "nx": 128,
                    "ny": 128,
                    "hidden_size": [512],
                    "dr": 0.1,
                    "activation": "gelu",
                    "normalization": None,
                },
            }
            for _ in range(16)
        ],
        "gating_cfgs": {
            "model_name": "SimpleAnn",
            "model_hyperparam": {
                "nx": 128,
                "hidden_size": [128],
                "dr": 0.0,
                "activation": "gelu",
                "normalization": "RMSNorm",
            },
        },
    },
}
```

---

## 3. Diffusion Model

### ConditionalDiffusionModel

```
条件输入 [B, T, 23]
  → 时间嵌入 (timestep → 32 维)
  → 去噪网络: LSTM (hidden_size=256, 4 层)
  → 预测噪声 → 逐步去噪 (1000 步)
  → 输出 [B, T, 1]
```

- **模型注册名**: `"ConditionalDiffusionModel"`
- **实验脚本**: `experiments/train_diffusion_camels.py`
- **关键参数**:
  - `diffusion.num_timesteps=1000` — 扩散步数
  - `diffusion.data_dim=1` — 输出维度（径流）
  - `diffusion.condition_dim=23` — 条件维度（6 动态 + 17 静态）
  - `diffusion.time_emb_dim=32` — 时间嵌入维度
  - `diffusion.beta_schedule="linear"` — 线性噪声调度
  - `denoise_net.hidden_size=256` — 去噪网络隐藏维度
  - `denoise_net.num_layers=4` — 去噪网络层数
- **数据集**: `StreamflowDataset`，采样器 `KuaiSampler`
- **损失函数**: `GenerativeLoss`（评估用 `RMSESum`）
- **张量布局**: `which_first_tensor="batch"`（注意：扩散模型默认 batch 第一）
- **训练设置**: `batch_size=4, train_epoch=5`（扩散模型显存占用大）
- **特点**: 生成式模型，可输出预测分布而非点估计，适合不确定性量化

```python
model_name = "ConditionalDiffusionModel"
model_hyperparam = {
    "diffusion": {
        "num_timesteps": 1000,
        "data_dim": 1,
        "condition_dim": 23,
        "time_emb_dim": 32,
        "beta_schedule": "linear",
    },
    "denoise_net": {
        "hidden_size": 256,
        "num_layers": 4,
    },
}
```

---

## 4. FNO (Fourier Neural Operator)

### FNO

```
输入 [seq, B, 24]
  → Lifting 层: 24 → 64×2 (升维)
  → 4 层频谱层 (每层 16 频率模式)
    → Fourier 层: 频域卷积 + MLP
    → 跳跃连接 (linear + soft-gating)
  → Projection 层: 64×2 → 1 (降维)
  → 输出 [seq, B, 1]
```

- **模型注册名**: `"FNO"`
- **实验脚本**: `experiments/train_fno_camels.py`
- **关键参数**:
  - `n_modes=(16,)` — 保留 16 个频率模式
  - `in_channels=24` — 输入通道数（6 气象变量 + 时间编码）
  - `out_channels=1` — 输出通道数（径流）
  - `hidden_channels=64` — 隐藏层通道数
  - `n_layers=4` — 频谱层数
  - `lifting_channel_ratio=2` — 升维比例
  - `projection_channel_ratio=2` — 降维比例
  - `fno_skip="linear"` — 频谱层跳跃连接
  - `channel_mlp_skip="soft-gating"` — MLP 层跳跃连接
  - `norm="batch_norm"` — 归一化方式
  - `use_channel_mlp=True` — 启用通道 MLP
  - `channel_mlp_expansion=0.5` — MLP 扩展比例
- **数据集**: `FNODataset`
- **损失函数**: `NSELoss`
- **张量布局**: `which_first_tensor="sequence"`
- **训练设置**: `batch_size=64, train_epoch=50, patience=10`
- **特点**: 在频域学习序列模式，对长程依赖有天然优势

```python
model_name = "FNO"
model_hyperparam = {
    "n_modes": (16,),
    "in_channels": 24,
    "out_channels": 1,
    "hidden_channels": 64,
    "n_layers": 4,
    "lifting_channel_ratio": 2,
    "projection_channel_ratio": 2,
    "fno_skip": "linear",
    "channel_mlp_skip": "soft-gating",
    "norm": "batch_norm",
    "use_channel_mlp": True,
    "channel_mlp_expansion": 0.5,
    "preactivation": False,
}
```

---

## 5. 快速开始

### 前提条件

1. 安装 torchhydro 及依赖: `uv sync --all-extras`
2. 配置 `~/hydro_setting.yml`，设置 CAMELS-US 数据路径:

```yaml
storage:
  default_source: local
  local:
    root: 'D:/data/hydrodatasets'
```

### 训练命令

```bash
# MoE (标准混合专家，2 专家)
python experiments/train_moe_camels.py

# DenseMoE (密集连接，8 专家)
python experiments/train_densemoe_camels.py

# GlobalTopKMoE (Switch MoE，稀疏激活)
python experiments/train_switchmoe_camels.py

# HydrologyTopKMoE (流域属性驱动门控)
python experiments/train_hydrotopkmoe_camels.py

# LSTMMoE (序列级 MoE，16 专家)
python experiments/train_lstmmoe_camels.py

# ConditionalDiffusionModel (扩散模型)
python experiments/train_diffusion_camels.py

# FNO (傅里叶神经算子)
python experiments/train_fno_camels.py
```

### 查看结果

```bash
# 训练结果保存在 results/ 目录下
ls results/moe_camels/
ls results/diffusion_camels/
ls results/fno_camels/
ls results/sparseMoE/

# 查看评估指标
cat results/*/metric_streamflow.csv
```

---

## 6. 评估

```bash
# MoE 模型评估
python experiments/evaluate_moe_camels.py

# LSTMMoE 模型评估
python experiments/evaluate_lstmmoe_camels.py
```

评估脚本会加载训练好的模型权重，在测试集上计算 NSE、RMSE、KGE 等水文指标，并生成预测过程线。

---

## 7. 数据要求

所有实验均使用 **CAMELS-US** 数据集（`dataset_id: "camels_us"`），通过统一数据接口从 `hydro_setting.yml` 解析路径。

### 输入变量 (`var_t`)

| 序号 | StandardVariable | 说明 |
|:---:|------|------|
| 1 | `PRECIPITATION` | 降水量（mm，强制变量必须排第一） |
| 2 | `DAYLIGHT_DURATION` | 日照时长（h） |
| 3 | `SOLAR_RADIATION` | 太阳辐射（W/m2） |
| 4 | `TEMPERATURE_MAX` | 最高温度（℃） |
| 5 | `TEMPERATURE_MIN` | 最低温度（℃） |
| 6 | `VAPOR_PRESSURE` | 水汽压（kPa） |

### 输出变量 (`var_out`)

| 序号 | StandardVariable | 说明 |
|:---:|------|------|
| 1 | `STREAMFLOW` | 径流量（mm，必须排第一） |

### 静态属性（MoE 门控网络使用）

DenseMoE、GlobalTopKMoE、HydrologyTopKMoE 使用 17 个静态属性作为门控输入，包括气候、地形、土壤等流域特征。

### 公共训练设置

| 参数 | 值 | 说明 |
|------|------|------|
| `source_cfgs` | `{"dataset_id": "camels_us"}` | 统一数据接口 |
| `scaler` | `"DapengScaler"` | 水文专用标准化 |
| `loss_func` | `"RMSESum"` | RMSE 损失（FNO 用 `"NSELoss"`） |
| `opt` | `"Adam"` | 优化器 |
| `min_time_unit` | `"D"` | 日尺度 |
| `which_first_tensor` | `"sequence"` | 张量布局（扩散模型除外） |

---

## 8. MoE 变体对比

| 特性 | MoE | DenseMoE | GlobalTopKMoE | HydrologyTopKMoE | LSTMMoE |
|------|:---:|:---:|:---:|:---:|:---:|
| 专家数 | 2 | 8 | 8 | 8 | 16 |
| 专家类型 | CpuLSTM | KuaiLSTM | CpuLSTM | CpuLSTM | SimpleAnn |
| 门控输入 | 动态 (23) | 静态 (17) | 静态 (17) | 静态 (17) | 隐藏状态 (128) |
| 门控模式 | 最后时间步 | 密集连接 | Top-K | Top-K (k=2) | 最后时间步 |
| 负载均衡 | 无 | 无 | 有 (0.02) | 有 (0.01) | 有 (0.01) |
| 分布式支持 | 否 | 否 | 否 | 否 | FSDP |
| 序列级 MoE | 否 | 否 | 否 | 否 | 是 |

---

## 8. AEF（注意力增强）变体

AEF（Attention Enhancement Framework）为标准模型添加注意力机制，提升对关键时间步的捕获能力。

### 8.1 LSTMMoE + AEF

```bash
# 训练
uv run python experiments/train_lstmmoe_aef.py --ctx 0,1,2

# 评估
uv run python experiments/evaluate_lstmmoe_aef.py
```

- 模型：`LSTMMoE`（8 个 SimpleAnn 专家 + AEF 注意力门控）
- 数据：CAMELS-Hourly（`dataset_id: "camelsh"`）
- 输入：23 个动态变量 + 静态属性
- 特点：注意力机制增强门控网络对关键时间步的识别

### 8.2 SimpleLSTM + AEF（CAMELS-US）

```bash
# 训练
uv run python experiments/train_with_us_aef.py --ctx 0

# 评估
uv run python experiments/evaluate_with_us_aef.py
```

- 模型：`SimpleLSTM` + AEF 注意力层
- 数据：CAMELS-US（`dataset_id: "camels_us"`）
- 输入：71 个特征（23 动态 + 48 静态属性）
- 隐藏层：128
