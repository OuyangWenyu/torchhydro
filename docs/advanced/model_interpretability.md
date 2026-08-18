# 模型可解释性工具

## 概述

torchhydro 提供两种模型可解释性工具：SHAP 特征归因分析和 Loss Landscape 可视化。两种工具均针对已训练的模型，用于分析模型行为和特征重要性。

## SHAP 特征归因分析

### 原理

基于 SHAP (SHapley Additive exPlanations) 计算每个输入特征对模型预测的贡献。torchhydro 内置 `shap.DeepExplainer`，可直接处理 PyTorch 模型，无需手动构造 background dataset。

主要功能：

- **特征重要性排序**：按平均绝对 SHAP 值排列各输入特征的贡献度
- **时序归因分析**：沿时间步聚合 SHAP 值，观察各特征在不同时段的影响变化
- **Beeswarm 散点图**：展示每个样本中特征值与 SHAP 值的关系
- **热力图**：逐样本展示特征-SHAP 值分布

### 使用方法

#### 独立运行

脚本路径：`experiments/run_shap_standalone.py`

```bash
# 1. 先训练模型（以 MTSLSTM 为例）
# 2. 运行 SHAP 分析
python experiments/run_shap_standalone.py
```

需要修改的参数：

- `case_dir`：已训练模型的输出目录（包含 `config.json` 和 `.pth` 权重文件）
- `basin_ids`：要分析的流域列表；设为 `None` 则自动从 `metric_streamflow.csv` 读取

脚本会自动加载目录中的实验配置，关闭训练模式，加载最佳权重进行推理和 SHAP 分析。

#### 代码集成

在训练脚本中通过 `evaluation_cfgs.shap_cfgs` 启用：

```python
cfg["evaluation_cfgs"]["shap_cfgs"] = {
    "enable": True,
    "save_dir": "./shap_results",
    "max_sample": 1000,           # SHAP 采样数
    "background_size": 100,       # background dataset 大小
    "sequence_enable": True,      # 启用时序归因
    "sequence_max_sample": 64,    # 时序分析采样数
    "time_agg": "abs_sum",        # 时间聚合方式
    "plot_mode": "split_only",    # 绘图模式
}
```

核心配置项：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable` | 是否启用 SHAP 分析 | `False` |
| `max_sample` | SHAP 分析最大样本数 | 3000 |
| `background_size` | background dataset 大小 | 100 |
| `sequence_enable` | 启用时序 SHAP 归因 | `False` |
| `time_agg` | 时间聚合方式 (`abs_sum` / `mean`) | `abs_sum` |
| `plot_mode` | 绘图模式 (`all` / `split_only`) | `all` |

### 输出

- 特征重要性排序柱状图
- 时序堆叠柱状图（各时间步的特征贡献）
- Beeswarm 散点图
- 热力图（逐样本 SHAP 值分布）
- 输出保存在 `save_dir` 指定的目录下

## Loss Landscape 可视化

### 原理

通过在参数空间中沿随机方向扰动模型权重，计算每个扰动点的 loss 值，绘制 loss surface。该方法基于 [Li et al., 2018](https://arxiv.org/abs/1712.09913) 的 loss landscape 可视化方法。

支持三种方向类型：

- `weights`：沿权重参数空间方向扰动
- `states`：沿隐藏状态空间方向扰动

支持两种归一化方式：

- `filter`：按 filter 归一化扰动方向
- `layer`：按 layer 归一化扰动方向

### 使用方法

脚本路径：`experiments/plot_loss_landscape.py`

```bash
# 单机运行（1D loss curve）
python experiments/plot_loss_landscape.py --x "-1:1:51"

# 2D contour plot
python experiments/plot_loss_landscape.py --x "-1:1:51" --y "-1:1:51"

# MPI 并行
mpirun -n 4 python experiments/plot_loss_landscape.py --mpi --x "-1:1:51"
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--x` | X 轴范围 `xmin:xmax:xnum` | `-1:1:51` |
| `--y` | Y 轴范围 `ymin:ymax:ynum`（2D 绘图） | 无（仅 1D） |
| `--dir_type` | 扰动方向类型 (`weights` / `states`) | `weights` |
| `--xnorm` / `--ynorm` | 归一化方式 (`filter` / `layer` / `weight`) | `filter` |
| `--output_dir` | 输出目录 | `./loss_landscape_output` |
| `--mpi` | 启用 MPI 并行计算 | 否 |
| `--cuda` | 使用 CUDA 加速 | 否 |
| `--vmin` / `--vmax` | 绘图的 loss 值范围 | `0.1` / `10` |
| `--show` | 显示绘图窗口 | 否 |

### 输出

- 1D loss curve：沿单一扰动方向的 loss 变化曲线
- 2D contour plot：两个扰动方向组合的 loss 等高线图
- 3D surface plot：loss surface 的三维曲面图
- 输出保存在 `output_dir` 指定的目录下

## 数据要求

- **SHAP**：需要已训练的模型，输出目录中包含 `config.json`（实验配置）和 `.pth`（模型权重文件）
- **Loss Landscape**：需要已训练的模型权重文件和训练数据集（用于计算 loss）

## 相关模块

- `torchhydro/explainers/shap.py` — SHAP 分析核心（特征归因、可视化函数）
- `torchhydro/explainers/loss_landscape/` — Loss Landscape 计算与可视化
- `torchhydro/explainers/uncertainty_analysis.py` — 不确定性分析
- `torchhydro/explainers/weight_anlysis.py` — 权重分析
- `torchhydro/trainers/deep_hydro.py` — SHAP 集成入口（`run_shap_analysis` 方法）
