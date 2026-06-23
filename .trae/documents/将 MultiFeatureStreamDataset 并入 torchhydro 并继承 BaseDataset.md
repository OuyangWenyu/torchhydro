## 目标

* 在 `torchhydro/datasets/data_sets.py` 中实现一个继承 `BaseDataset` 的 `MultiFeatureStreamDataset`，将脚本 `train_gcgru_jinshajiang_32_daily_HF_multifeature.py:532` 的数据集构建逻辑（多特征、未来信息错位、可选节点掩码、可选 discharge 覆盖）融入框架统一的读取、归一化与取样机制。

* 与现有 `BaseDataset` 管线保持一致：使用 `_read_xyc()` 读取并对齐 X/Y/C，交由 `_normalize()` 统一归一化，保持 `__getitem__` 返回格式一致，同时实现多特征时序堆叠与未来信息错位。

## 现状梳理

* `BaseDataset` 管线与接口：`torchhydro/datasets/data_sets.py:199` 定义，`_load_data()` 在 `torchhydro/datasets/data_sets.py:454` 处调用 `_read_xyc()` → `_normalize()` → `_trans2nparr()` → `_create_lookup_table()`；`__getitem__` 在 `torchhydro/datasets/data_sets.py:443` 返回 `(x_window, y_window)`。

* 目标插入位置：当前在 `torchhydro/datasets/data_sets.py:3542` 存在 `AutoregressiveTGDataset(BaseDataset)`；计划在该位置新增/替换为 `MultiFeatureStreamDataset(BaseDataset)`，保留 TG 特定配置与上游关系可选支持。

* 脚本版数据集逻辑来源：`/home/mahaoran/research/torchhydro/scripts/train_gcgru_jinshajiang_32_daily_HF_multifeature.py:532` 起的 `MultiFeatureStreamDataset(TensorDataset)`，核心特征：

  * 输入包含 3 个时间序列通道：降水、LSTM 流量、实测/水库出库流量；静态属性按时间扩展并拼接。

  * 降水与 LSTM 在历史窗口内向后错位 +1（引入“未来1天”信息），目标为观测流量的未来窗口。

  * 可选 `use_confluence_nodes_only` 时返回/使用节点掩码；可选 `discharge_data` 用非 NaN 值覆盖历史观测。

## 修改方案

1. 新增类 `MultiFeatureStreamDataset(BaseDataset)`（文件位置：约 `torchhydro/datasets/data_sets.py:3542` 附近），结构与接口：

   * `__init__(cfgs, is_tra_val_te, upstream_indices: dict | None = None)`: 读取 `hindcast_length(rho)`、`forecast_length(horizon)`、`warmup_length`；从 `data_cfgs.tg_cfgs` 中读取 `use_confluence_nodes_only`、`basin_names`、`basin_id`、`node_attributes` 等；可选 `upstream_indices`（用来分类源头/汇流节点）。

   * `_read_xyc(self)`: 统一读取并对齐三类时间序列与静态属性：

     * 通过 `self.data_source`（`torchhydro/datasets/data_sets.py:269`）读取 `relevant_cols` 中的 `precipitation`、`lstm_streamflow`、`observed_streamflow` 三个变量（以及 `target_cols` 的 `observed_streamflow`）；`constant_cols` 通过 `read_attr_xrdataset` 读取；

     * 按 `t_s_dict["t_final_range"]` 构造完整时间索引并 `reindex`（参考 `torchhydro/datasets/data_sets.py:3502-3513`）；

     * 将三条序列按 `variable` 维拼为一个 `DataArray`，形状 `(basin, time, variable=3)`；目标 `y` 为观测流量 `(basin, time, variable=1)`；静态 `c` 为 `(basin, variable_attr)`。

     * 返回字典键严格沿用框架：`{"relevant_cols": x, "target_cols": y, "constant_cols": c}`。

   * `interpolate_missing_data(self, data, method="linear")`: 复用脚本插值策略，返回插值后数据与掩码（参考已存在实现 `torchhydro/datasets/data_sets.py:3586`）。在 `_read_xyc` 阶段仅对 X 的合适变量（如降水、LSTM）可选插值，Y 保留 NaN 以便损失掩码。

   * `__getitem__(self, item)`: 基于 `BaseDataset` 的 `lookup_table`，构造窗口：

     * 历史窗口长度 `M = rho`，未来窗口长度 `F = horizon`；`start_idx = idx + M`。

     * 从框架保存的 `self.x` 拆出三个通道：`rain`, `lstm`, `obs`；在时间维对 `rain`、`lstm` 执行向后错位 `+1`（取 `[idx+1 : start_idx+1]`），`obs` 取 `[idx : start_idx]`；全部 `clamp(min=0)`；

     * 若存在 `discharge`（可通过在 `_read_xyc` 中作为额外可选数据或 `data_source` 提供），在历史窗口中用非 NaN 覆盖 `obs`；

     * 将三通道堆叠为 `[num_nodes, M, 3]` 再转置为 `[M, num_nodes, 3]`；如有 `c`，按时间扩展并在特征维拼接，得到 `[M, num_nodes, 3+attr_dim]`；

     * 目标 `y` 返回 `[start_idx : start_idx+F]` 的观测流量，形状 `[F, num_nodes]`（与脚本版一致）；

     * 保持返回二元组 `(input_features.float(), future_obs.float())`，节点掩码作为数据集属性供上层使用。

   * 节点掩码：

     * 若 `use_confluence_nodes_only` 为真，基于 `upstream_indices` 分类：入度为 0 → 源头；入度 ≥ 2 → 汇流；生成 `self.node_mask`（`[num_nodes]`）；不改变返回格式，以属性方式供模型/训练器读取。

2. 与数据源协同

   * 依赖你在 `hydrodatasource/reader/data_source.py:1713` 已更新的逻辑：保证数据源能返回 `lstm_streamflow` 与 `observed_streamflow` 两类时间序列，并与 `precipitation` 在相同时间轴对齐；

   * 若 `discharge` 需要参与覆盖，提供 `read_ts_xrdataset` 或专用读取方法返回 `[basin, time]` 的 `DataArray`，并在 `_read_xyc` 中以可选项读入。

3. 配置约定（示例）

* `data_cfgs`：

  * `relevant_cols`: `["precipitation", "lstm_streamflow", "observed_streamflow"]`

  * `target_cols`: `["observed_streamflow"]`

  * `constant_cols`: `[...]`（静态属性名列表）

  * `min_time_unit/min_time_interval`: 与现有保持一致

  * `source_cfgs`: `{source_name, source_path, other_settings: {...}}` 保证数据源可读上述变量

  * `tg_cfgs`: `{use_confluence_nodes_only: bool, basin_names: [...], basin_id: str/int}`

* `training_cfgs`: `{hindcast_length, forecast_length, warmup_length, valid_batch_mode}` 与框架保持一致。

1. 兼容与归一化

* 归一化与反归一化沿用 `ScalerHub`（`torchhydro/datasets/data_sets.py:519`），无需在数据集类中重复处理；

* `BaseDataset.__getitem__` 的拼接逻辑由子类覆盖，自行实现三通道+静态属性拼接，保持最终返回张量与脚本版一致；

* 不更改框架的 `DataLoader` 使用方式与训练入口；如需节点掩码，上层从数据集属性读取。

1. 验证计划

* 构造一个小样本运行数据集：检查 `len(dataset)` 与 `__getitem__(0)` 的输出形状是否分别为 `[hindcast_length, num_nodes, 3+attr_dim]` 与 `[forecast_length, num_nodes]`；

* 检查时间对齐与 +1 错位是否生效（随机点位对比原始 `x` 三通道索引）；

* 若提供 `discharge`，验证覆盖逻辑仅在非 NaN 位置生效；

* 验证 `node_mask` 与 `upstream_indices` 的一致性（入度统计）。

1. 代码定位参考

* `BaseDataset` 接口与管线：`torchhydro/datasets/data_sets.py:199`、`torchhydro/datasets/data_sets.py:454`

* 计划插入/替换位置：`torchhydro/datasets/data_sets.py:3542` 附近现有 TG 数据集类

* 脚本版数据集来源：`/home/mahaoran/research/torchhydro/scripts/train_gcgru_jinshajiang_32_daily_HF_multifeature.py:532`

## 交付内容

* 在 `torchhydro/datasets/data_sets.py` 中新增 `MultiFeatureStreamDataset(BaseDataset)` 完整实现（含 `_read_xyc`、`__getitem__`、插值与节点掩码支持），并与更新后的 `hydrodatasource` 对接；

* 不修改其他数据集类行为；

* 提供一个最小验证用例片段（仅用于本地自检，与你的训练脚本保持接口一致）。

