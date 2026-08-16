# 云端 Zarr 联合训练说明

## 决策

第一版联合训练不应该基于 TorchHydro 当前的 eager `BaseDataset` 路径实现。

当前 hydrodataset 云端 bucket 在 `s3://hydrodataset/zarr` 下有 53 个 Zarr store。只读取 Zarr 元数据后估算，原始 xarray 数组体积约为 279.64 GiB。TorchHydro 当前 eager 路径会保留原始数组、归一化数组，以及 scaler 和 NaN 处理阶段的临时数组。因此，在不计算模型、优化器和 DataLoader 额外开销的情况下，实际训练内存更接近 559.27-838.91 GiB。

本次探测机器总内存为 16 GiB，探测时可用内存约为 3.58 GiB。因此，全量 eager 加载所有云端 Zarr 数据集不可行。

## 探测脚本读取了什么

`experiments/probe_hydrodataset_cloud_zarr_memory.py` 使用 `xr.open_zarr` 打开每个云端 Zarr store，并且只读取元数据：

- 维度
- 变量名
- 根据 dtype 和 shape 计算出的 `nbytes`
- 坐标数组字节数
- xarray 汇总的 `nbytes`

除非显式传入 `--try-load-if-fits`，并且元数据估算结果低于安全内存阈值，否则脚本不会调用 `.load()`。

## Lazy Loading 要求

联合训练数据集应该继续使用 map-style PyTorch Dataset，而不是纯 `IterableDataset`。原因是 TorchHydro 训练需要稳定的样本数量、shuffle、sampler，以及验证和测试阶段的结果恢复。

推荐使用 batch-level lazy 设计：

1. 内存里只保留元数据、站点 id、时间坐标、静态属性和紧凑 lookup table。
2. 实现 `__getitems__(indices)`，并把它作为 DataLoader 主路径。
3. 把样本索引解析成 `(dataset_id, basin_index, start_time, length)`。
4. 从 Zarr 编码元数据读取真实 `(basin, time)` chunk 形状。
5. 按 `(dataset_id, basin_chunk, time_chunk)` 对 batch 内窗口分组。
6. 一次读取整个二维物理块，并在 worker 内使用有界 LRU 缓存复用。
7. 只把当前 batch 物化成 NumPy/Torch tensor。
8. 使用预先计算好的统计量对 batch 做归一化。

不要对每个样本单独发起远程读取。那会产生大量对象存储请求，吞吐会很差。

## Scaler 要求

当前 `DapengScaler` 和 `SklearnScaler` 都会对完整 xarray 对象调用 `.to_numpy()`。lazy 联合训练需要拆成两个组件：

- `StatComputer`：按 chunk 计算训练期统计量，并写出 stat JSON 文件。
- `BatchNormalizer`：读取保存好的统计量，只对 batch 级 NumPy 数组做归一化。

验证集和测试集应该直接加载训练阶段保存的统计量，不应该再次扫描完整云端数据集。

## NaN 处理

在 Dataset 构造阶段做全序列插值不适合 lazy loading。建议按下面优先级处理：

1. 尽量使用已经清洗过的 Zarr store。
2. 通过分块扫描 target 变量生成有效样本 lookup table。
3. 对输入变量使用 batch-local fill 规则，或依赖预处理后的 store。
4. 如果必须做完整时间维插值，应放到离线预处理任务中，而不是训练时动态执行。

## 变量统一

云端 Zarr store 中保存的是各数据集自己的原始变量名。联合训练不应该直接依赖这些原始名字，而应该优先使用 hydrodataset 提供的标准变量映射，例如：

- `streamflow`
- `precipitation`
- `temperature_mean`
- `temperature_min`
- `temperature_max`
- `potential_evapotranspiration`

lazy reader 需要同时保存两类名字：

- 模型侧使用的标准变量名
- Zarr store 中实际存在的变量名

每个数据集可以独立配置 `physical`、`scale`、`offset`、`unit`、
`required` 和 `fill_value`。变换发生在统计量计算与 batch 归一化之前。
`fill_and_mask` 会同时为缺失的动态变量和静态变量追加 mask；未配置的可选静态
变量不会触发 attributes store 读取。
当相同标准变量在不同数据集声明了不同单位时，默认给出警告；正式训练可把
`unit_mismatch_policy` 设为 `error`，避免把物理量不一致的数据静默混合。

联合训练还要求时间步长一致。日尺度和小时尺度数据默认不能进入同一训练任务；
应先离线重采样，而不是仅设置 `allow_mixed_time_steps: true`。

## 多数据集配置

推荐通过 `configs/cloud_zarr_joint.example.yml` 管理每个数据集自己的流域、
时间范围、变量映射、URI 和采样权重。使用 hydrodataset 云端 reader 时可以省略
URI；直接访问指定 Zarr 时，在每个数据集条目中设置 `timeseries_uri` 和
`attributes_uri`。

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

`weighted_by_sample` 按各数据集的自然样本量训练；`balanced_by_dataset` 让每个
数据集贡献相同 batch 数；`weighted_by_dataset` 使用清单中的 `weight` 分配
batch。若使用后两种模式，建议设置 `max_batches_per_epoch` 明确每个 epoch 的长度。

不要默认传入 `time_chunk_size` 或 `basin_chunk_size`。代码会读取每个 Zarr 变量的
真实 chunk 元数据并自动对齐；只有确认存储编码错误或做对照实验时才手工覆盖。

## 第一版实现范围

第一版 lazy 联合训练应尽量收窄范围：

- 只支持规则日尺度或小时尺度时间序列
- 只支持固定长度窗口
- 支持动态时间序列变量和少量静态属性
- 先支持一个 target 变量，通常是 `streamflow`
- 使用 batch-level `__getitems__`
- 使用预先计算好的 scaler 统计量

下面这些能力应等基础路径稳定后再做：

- 变长训练
- 图、站点、洪水事件数据集
- forecast 4D 数组
- 完整时间维插值
- 多频率模型
- 训练时动态展开多个变量来源

## Benchmark 建议

先用同一组参数各跑一次普通 GPU 拷贝和 CUDA prefetch，对比
`avg_dataloader_wait_seconds`、`samples_per_second` 和
`avg_gpu_util_percent`：

```bash
python experiments/benchmark_cloud_zarr_lazy_dataset.py \
  --dataset-id camels_us \
  --source cloud \
  --max-basins 64 \
  --steps 100 \
  --batch-size 128 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --json-out /tmp/camels_us_lazy_no_cuda_prefetch.json

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

benchmark 默认使用 `--normalization none`，并且不做 target 全 NaN 窗口过滤，
目的是先测 DataLoader、Zarr window 读取和 GPU copy/compute 的速度。如果要测
更接近正式训练的完整初始化成本，再显式加：

```bash
--normalization standard --valid-target-filter
```

这会在 benchmark 计时前扫描云端变量计算统计量，并扫描 target 构造有效窗口，
首次运行可能明显变慢。

如果 `avg_dataloader_wait_seconds` 仍然明显高于 `avg_compute_seconds`，
先从 `num_workers=2/4`、`prefetch_factor=2/4` 做矩阵测试。对象存储连接数过多
也可能降低吞吐，因此不要直接把 worker 数开到 CPU 逻辑核心数。输出中的
`effective_chunk_shapes` 应与 Zarr 编码一致。`dataloader_in_order=false` 是云端
训练默认值，某个 worker 网络抖动时不会阻塞其他已经完成的 batch。

正式训练使用 `drop_any_nan`，会分块扫描 target 并缓存压缩后的有效性索引，避免
把部分缺测 target 静默填成 0。统计量和有效性索引首次构建较慢，后续运行直接
使用 `lazy_zarr_stat_cache` 与 `lazy_zarr_index_cache`。

## 本地性能回归

在 Mac 本地使用 64 个 basin、1096 天、物理 chunk `(64, 365)` 的合成 Zarr，
读取 100 个 batch、每批 128 个样本：

- 自动物理 chunk + 64 MiB LRU：约 370,456 samples/s，平均等待 0.332 ms。
- `(1, 128)` + 无缓存：约 51,373 samples/s，平均等待 2.465 ms。

本地磁盘数值不能替代服务器 OSS/S3 基准，但说明连续 chunk 调度和缓存可避免
同一对象的重复读取。服务器仍应以 `samples_per_second`、batch 等待时间、GPU
利用率和 RSS 为最终判断依据。
