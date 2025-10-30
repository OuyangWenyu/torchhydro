# Lightning Fabric 调试与分布式训练指南

## 问题描述

在使用 Lightning Fabric 进行深度学习训练时，您可能会遇到以下问题：

1. **调试困难**：分布式训练会增加调试的复杂性
2. **开发效率低**：每次修改代码都需要启动分布式训练
3. **灵活性不足**：无法轻松地在单机调试和分布式训练之间切换

## 解决方案

我们提供了一个灵活的 `FabricWrapper` 类，让您可以在调试模式和分布式训练模式之间无缝切换。

## 使用方法

### 1. 配置调试模式

```python
# 在您的训练配置中添加以下参数
config_data["training_cfgs"]["debug_mode"] = True
config_data["training_cfgs"]["use_fabric"] = False
```

### 2. 配置分布式训练模式

```python
# 在您的训练配置中添加以下参数
config_data["training_cfgs"]["debug_mode"] = False
config_data["training_cfgs"]["use_fabric"] = True
config_data["training_cfgs"]["force_fabric"] = True
```

### 3. 自动模式

系统会根据以下条件自动选择模式：

- 如果只有一个 GPU，建议使用调试模式
- 如果有多个 GPU，建议使用分布式训练模式
- 如果设置了 `DEBUG_MODE=true` 环境变量，强制使用调试模式

## 主要特性

### 调试模式 (debug_mode=True)
- ✅ 使用普通的 PyTorch 操作
- ✅ 支持断点调试
- ✅ 简单的错误信息
- ✅ 快速启动
- ✅ 单GPU训练

### 分布式训练模式 (use_fabric=True)
- ✅ 使用 Lightning Fabric
- ✅ 多GPU并行训练
- ✅ 自动处理分布式细节
- ✅ 更好的性能
- ✅ 支持多节点训练

## 使用示例

### 方式1：通过配置文件

```python
from torchhydro.configs.config import default_config_file, update_cfg, cmd
from torchhydro.trainers.trainer import train_and_evaluate

# 创建调试配置
config_data = default_config_file()
args = cmd(
    sub="debug_example",
    ctx=[0],  # 单GPU
    model_name="LSTM",
    # ... 其他参数
)
update_cfg(config_data, args)

# 启用调试模式
config_data["training_cfgs"]["debug_mode"] = True
config_data["training_cfgs"]["use_fabric"] = False

# 开始训练
train_and_evaluate(config_data)
```

### 方式2：通过环境变量

```bash
# 调试模式
export DEBUG_MODE=true
export CUDA_VISIBLE_DEVICES=0
python your_training_script.py

# 分布式训练模式
export DEBUG_MODE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
python your_training_script.py
```

### 方式3：通过命令行参数

```bash
# 调试模式
python examples/debug_vs_distributed_training.py --mode debug

# 分布式训练模式
python examples/debug_vs_distributed_training.py --mode distributed

# 自动模式
python examples/debug_vs_distributed_training.py --mode auto
```

## 工作流程建议

### 开发阶段
1. 使用调试模式进行代码开发和调试
2. 使用少量数据和少量epochs进行快速验证
3. 使用断点和print语句进行调试

```python
# 调试配置示例
config_data["training_cfgs"]["debug_mode"] = True
config_data["training_cfgs"]["use_fabric"] = False
config_data["training_cfgs"]["epochs"] = 5  # 少量epochs
config_data["data_cfgs"]["batch_size"] = 32  # 小batch size
```

### 验证阶段
1. 使用单GPU进行中等规模的验证
2. 确保模型正常工作
3. 检查性能指标

### 生产阶段
1. 切换到分布式训练模式
2. 使用多GPU进行大规模训练
3. 使用完整的数据集和epochs

```python
# 生产配置示例
config_data["training_cfgs"]["debug_mode"] = False
config_data["training_cfgs"]["use_fabric"] = True
config_data["training_cfgs"]["force_fabric"] = True
config_data["training_cfgs"]["epochs"] = 100  # 完整epochs
config_data["data_cfgs"]["batch_size"] = 256  # 大batch size
```

## 配置参数说明

| 参数           | 类型 | 默认值    | 说明                         |
| -------------- | ---- | --------- | ---------------------------- |
| `debug_mode`   | bool | False     | 是否启用调试模式             |
| `use_fabric`   | bool | True      | 是否使用Lightning Fabric     |
| `force_fabric` | bool | False     | 强制使用Fabric（即使单GPU）  |
| `strategy`     | str  | "auto"    | 分布式策略 (ddp, fsdp, auto) |
| `precision`    | str  | "32-true" | 训练精度                     |
| `accelerator`  | str  | "auto"    | 加速器类型                   |

## 注意事项

1. **调试模式下的限制**：
   - 只支持单GPU训练
   - 不支持混合精度训练
   - 不支持模型并行

2. **分布式模式下的限制**：
   - 调试相对复杂
   - 启动时间较长
   - 需要更多内存

3. **迁移注意事项**：
   - 现有代码中的 `total_fab` 调用会自动适配
   - 无需修改现有的训练逻辑
   - 配置文件向后兼容

## 故障排除

### 常见问题

1. **问题**：在调试模式下出现 "fabric not found" 错误
   **解决**：检查是否正确设置了 `debug_mode=True`

2. **问题**：分布式训练无法启动
   **解决**：检查 CUDA_VISIBLE_DEVICES 和 strategy 设置

3. **问题**：模型在不同模式下表现不一致
   **解决**：检查 batch_size 和 learning_rate 设置

### 日志示例

```
🐛 Debug mode enabled - disabling Lightning Fabric
✅ Normal PyTorch initialized, using device: cuda:0
🐛 Debug mode configuration created
   - Single device: [0]
   - Lightning Fabric: False
   - Debug mode: True
```

```
✅ Lightning Fabric initialized successfully
🚀 Distributed training configuration created
   - Devices: [0, 1]
   - Strategy: ddp
   - Lightning Fabric: True
   - Debug mode: False
```

## 总结

通过使用我们的 `FabricWrapper` 系统，您可以：

1. **提高开发效率**：在调试时使用简单的PyTorch，在生产时使用分布式训练
2. **降低调试难度**：避免分布式训练带来的调试复杂性
3. **保持代码一致性**：无需修改现有代码，只需更改配置
4. **灵活切换**：根据需要在不同模式之间切换

这个解决方案完美地平衡了开发效率和训练性能，让您能够专注于模型开发而不是基础设施问题。 