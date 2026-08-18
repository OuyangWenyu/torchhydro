# Cloud-Zarr 联合训练指南

## 概述

`CloudZarrLazyDataset` 支持从云端 Zarr store 懒加载数据进行联合训练，无需将完整数据集下载到本地。适用于多数据集联合训练场景（如 CAMELS-US + CAMELS-Hourly + Caravan 等），数据量可达数百 GiB。

核心组件：
- `CloudZarrLazyDataset`（`torchhydro/datasets/cloud_zarr_lazy_dataset.py`）— map-style Dataset，按 batch 从 Zarr chunk 读取
- `CloudZarrChunkBatchSampler`（`torchhydro/datasets/cloud_zarr_sampler.py`）— 按 Zarr chunk 分组采样，减少远程读取次数

## 快速开始

### 1. 配置 `~/hydro_setting.yml`

```yaml
storage:
  default_source: cloud
  local:
    root: /path/to/local/cache
  s3:
    bucket: hydrodataset
    prefix: zarr
```

### 2. 单数据集训练

```bash
PYTHONPATH=. python experiments/train_cloud_zarr_lazy_joint.py \
  --dataset-id camels_us \
  --source cloud \
  --batch-size 128 \
  --hidden-size 256 \
  --num-workers 4 \
  --train-epochs 10
```

### 3. 多数据集联合训练

```bash
PYTHONPATH=. python experiments/train_cloud_zarr_lazy_joint.py \
  --dataset-config configs/cloud_zarr_joint.example.yml \
  --datasets all \
  --source cloud \
  --sampling weighted_by_dataset \
  --normalization standard \
  --normalization-scope global \
  --missing-policy fill_and_mask \
  --target-nan-policy drop_any_nan \
  --batch-size 512 \
  --hidden-size 256 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --worker-cache-size-mb 128 \
  --cuda-prefetch \
  --cuda-prefetch-batches 2 \
  --dataloader-timeout 300
```

## 配置说明

### 数据集配置文件

多数据集训练通过 YAML 配置文件管理每个数据集的参数。参考 `configs/cloud_zarr_joint.example.yml`：

```yaml
datasets:
  camels_us:
    basins: "all"
    t_range_train: ["1980-10-01", "2010-09-30"]
    t_range_test: ["2010-10-01", "2019-09-30"]
    variables:
      relevant:
        precipitation: pcp_mm
        temperature_max: airtemp_C_max
        temperature_min: airtemp_C_min
        solar_radiation: solrad_wm2
        daylight_duration: dayl(s)
        vapor_pressure: vp_hpa
      target:
        streamflow: q_cms_obs
    weight: 1.0
```

每个数据集条目包含：
- `basins`：流域 ID 列表或 `"all"`
- `t_range_train`/`t_range_test`：训练/测试时间范围
- `variables.relevant`：输入变量映射（标准名 → Zarr 中的物理名）
- `variables.target`：目标变量映射
- `weight`：采样权重（用于 `weighted_by_dataset` 模式）

### 采样模式

| 模式 | 说明 |
|---|---|
| `weighted_by_sample` | 按各数据集的自然样本量训练 |
| `balanced_by_dataset` | 每个数据集贡献相同 batch 数 |
| `weighted_by_dataset` | 按配置文件中的 `weight` 分配 batch |

使用后两种模式时，建议设置 `max_batches_per_epoch` 明确每个 epoch 的长度。

### 变量映射

云端 Zarr store 中保存的是各数据集自己的原始变量名。联合训练通过 `variables` 配置将它们映射到 hydrodataset 标准变量名：

- 模型侧使用标准变量名（如 `precipitation`、`streamflow`）
- Zarr store 中使用物理变量名（如 `pcp_mm`、`q_cms_obs`）

每个变量可独立配置 `physical`、`scale`、`offset`、`unit`、`required` 和 `fill_value`。变换发生在统计量计算与 batch 归一化之前。

`fill_and_mask` 会同时为缺失的动态变量和静态变量追加 mask；未配置的可选静态变量不会触发 attributes store 读取。当相同标准变量在不同数据集声明了不同单位时，默认给出警告；正式训练可把 `unit_mismatch_policy` 设为 `error`。

### 归一化与统计量

CloudZarr 使用预计算的统计量进行 batch 级归一化，不扫描完整数据集：

- **训练阶段**：首次运行时按 chunk 计算统计量，缓存到 `lazy_zarr_stat_cache`
- **验证/测试阶段**：直接加载训练阶段保存的统计量

`--normalization standard` 使用 z-score 归一化；`--normalization-scope global` 计算跨数据集的全局统计量。

### NaN 处理

- `--target-nan-policy drop_any_nan`：分块扫描 target，丢弃含 NaN 的窗口
- `--missing-policy fill_and_mask`：对缺失变量追加 mask 标记
- 统计量和有效性索引首次构建较慢，后续运行使用缓存

## 性能调优

### DataLoader 配置

| 参数 | 建议值 | 说明 |
|---|---|---|
| `--num-workers` | 4-8 | 过多可能降低对象存储吞吐 |
| `--prefetch-factor` | 2-4 | 每个 worker 预取的 batch 数 |
| `--worker-cache-size-mb` | 64-256 | worker 内 LRU 缓存大小 |
| `--dataloader-timeout` | 300 | 超时秒数，防止 worker 挂起 |

### CUDA Prefetch

启用 `--cuda-prefetch` 可在 GPU 计算时异步预取下一个 batch 到 GPU 内存：

```bash
--cuda-prefetch --cuda-prefetch-batches 2
```

### 调优判断

如果 `avg_dataloader_wait_seconds` 明显高于 `avg_compute_seconds`：
1. 先从 `num-workers=2/4`、`prefetch_factor=2/4` 做矩阵测试
2. 对象存储连接数过多也可能降低吞吐
3. `dataloader_in_order=false` 是云端训练默认值，某个 worker 网络抖动时不会阻塞其他 batch

### Chunk 对齐

不要默认传入 `time_chunk_size` 或 `basin_chunk_size`。代码会读取每个 Zarr 变量的真实 chunk 元数据并自动对齐；只有确认存储编码错误或做对照实验时才手工覆盖。

## Benchmark

先用同一组参数各跑一次普通 GPU 拷贝和 CUDA prefetch，对比性能指标：

```bash
# 无 CUDA prefetch
python experiments/benchmark_cloud_zarr_lazy_dataset.py \
  --dataset-id camels_us \
  --source cloud \
  --max-basins 64 \
  --steps 100 \
  --batch-size 128 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --json-out /tmp/camels_us_lazy_no_cuda_prefetch.json

# 有 CUDA prefetch
python experiments/benchmark_cloud_zarr_lazy_dataset.py \
  --dataset-id camels_us \
  --source cloud \
  --max-basins 64 \
  --steps 100 \
  --batch-size 128 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --cuda-prefetch \
  --json-out /tmp/camels_us_lazy_cuda_prefetch.json
```

Benchmark 默认使用 `--normalization none`，不做过滤，测的是 DataLoader 和 Zarr 读取速度。要测完整初始化成本，加 `--normalization standard --valid-target-filter`。

关注指标：`samples_per_second`、`avg_dataloader_wait_seconds`、`avg_gpu_util_percent`。

## 设计背景

TorchHydro 的 `BaseDataset` 是 eager 加载——`__init__` 时一次性把所有数据读进内存。对于本地小数据集这没问题，但云端联合训练的数据量（~280 GiB）远超单机内存，eager 路径实际需要 559-839 GiB（含归一化和 scaler 临时数组）。

`CloudZarrLazyDataset` 采用 lazy 加载：内存里只存元数据（站点 ID、时间坐标、统计量），实际数据在 `__getitems__` 时按 Zarr chunk 读取。仍使用 map-style Dataset（非 IterableDataset），保证稳定的样本数、shuffle、验证/测试结果恢复。

## 当前支持与后续计划

| 能力 | 状态 |
|---|---|
| 规则日/小时尺度时间序列 | ✅ 已支持 |
| 固定长度窗口 | ✅ 已支持 |
| 动态时间序列变量 + 静态属性 | ✅ 已支持 |
| 单 target 变量（streamflow） | ✅ 已支持 |
| batch-level `__getitems__` | ✅ 已支持 |
| 预计算 scaler 统计量 | ✅ 已支持 |
| 多数据集联合训练 | ✅ 已支持 |
| CUDA prefetch | ✅ 已支持 |
| 变长训练 | 后续计划 |
| 图/站点/洪水事件数据集 | 后续计划 |
| forecast 4D 数组 | 后续计划 |
| 多频率模型 | 后续计划 |
