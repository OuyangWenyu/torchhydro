# Lightning Fabric 分布式训练指南

## 概述

TorchHydro 通过 `FabricWrapper` 集成了 [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)，支持在单机调试和多 GPU 分布式训练之间切换。Fabric 处理分布式通信、混合精度、设备管理等细节，训练代码无需修改。

## 配置

在 `training_cfgs` 中设置以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `fabric_strategy` | str/null | None | 分布式策略。设为 `"ddp"`/`"fsdp"`/`"auto"` 时启用 Fabric；`None` 时禁用 |
| `precision` | str | `"32-true"` | 训练精度（`"32-true"`、`"16-mixed"`、`"bf16-mixed"` 等） |
| `accelerator` | str | `"auto"` | 加速器类型（`"auto"`、`"gpu"`、`"cpu"` 等） |

### 禁用 Fabric（默认）

不设置 `fabric_strategy` 或设为 `None`，使用普通 PyTorch：

```python
config_data["training_cfgs"]["fabric_strategy"] = None
```

### 启用 DDP 分布式训练

```python
config_data["training_cfgs"]["fabric_strategy"] = "ddp"
config_data["training_cfgs"]["precision"] = "16-mixed"  # 可选
```

### 启用 FSDP（全分片数据并行）

```python
config_data["training_cfgs"]["fabric_strategy"] = "fsdp"
```

## 自动行为

`create_fabric_wrapper`（`torchhydro/trainers/fabric_wrapper.py`）在初始化时：

1. 检查 `fabric_strategy` 是否为 None → 是则禁用 Fabric
2. 检测 GPU 数量 → 单 GPU 时自动禁用 Fabric（无需手动关闭）
3. 将 `precision` 和 `accelerator` 传递给 Fabric

因此，同一份配置在单 GPU 和多 GPU 环境下都能正常工作，无需修改代码。

## 使用示例

### 通过配置文件

```python
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

args = cmd(
    sub="distributed_example",
    ctx=[0, 1],  # 使用 2 个 GPU
    model_name="CpuLSTM",
    # ... 其他参数
)
config_data = default_config_file()
update_cfg(config_data, args)

# 启用 DDP
config_data["training_cfgs"]["fabric_strategy"] = "ddp"

train_and_evaluate(config_data)
```

### 通过环境变量

```bash
# 单 GPU 调试（Fabric 自动禁用）
CUDA_VISIBLE_DEVICES=0 python train_script.py

# 多 GPU 分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_script.py
# 在脚本中设置 fabric_strategy="ddp"
```

## 工作流程

### 开发阶段

使用默认配置（不设 `fabric_strategy`），普通 PyTorch 训练：
- 支持断点调试
- 错误信息简洁
- 小数据量 + 少量 epoch 快速验证

### 生产阶段

设置 `fabric_strategy="ddp"`，启用分布式训练：
- 多 GPU 并行
- 自动处理分布式通信
- 混合精度训练（配合 `precision="16-mixed"`）

## 故障排除

1. **Fabric 未启用**：检查 `fabric_strategy` 是否设为非 None 值
2. **分布式训练启动失败**：检查 `CUDA_VISIBLE_DEVICES` 和 GPU 数量
3. **单 GPU 时 Fabric 被禁用**：这是预期行为——单 GPU 无需 Fabric 开销
4. **模型在不同模式下表现不一致**：检查 `batch_size` 和 `learning_rate` 是否需要调整

## 日志示例

禁用 Fabric 时：
```
[OK] Normal PyTorch initialized, using device: cuda:0
```

启用 Fabric 时：
```
✅ Lightning Fabric initialized successfully
🚀 Distributed training: devices=[0, 1], strategy=ddp
```
