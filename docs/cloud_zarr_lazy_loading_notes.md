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
4. 按 `(dataset_id, zarr_store, variable_set, basin)` 对 batch 内窗口分组。
5. 合并同组中重叠或相邻的时间片段。
6. 从 Zarr 中读取最小必要 xarray slice。
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
