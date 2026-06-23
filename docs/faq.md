# Frequently Asked Questions (FAQ)

## 1. Installation Related

### Q: What should I do if I encounter CUDA-related errors during installation?
A: First, ensure your CUDA version is compatible with your PyTorch version. You can check the compatibility table on the [PyTorch website](https://pytorch.org/get-started/locally/). If issues persist, try installing the corresponding PyTorch version first, then install torchhydro.

### Q: Is GPU required?
A: No, torchhydro can run on CPU environments, but using a GPU will significantly speed up training.

## 2. Data Related

### Q: What data formats are supported?
A: Currently, we support CAMELS dataset format, NetCDF format, and common CSV formats. For other formats, you can refer to the documentation to create custom data loaders.

### Q: How to handle missing values?
A: torchhydro provides various missing value handling strategies, including mean filling, forward filling, and interpolation. You can use the `MissingValueFiller` transformer to handle missing values.

## 3. Model Related

### Q: What should I do if the model loss doesn't converge?
A: Try the following approaches:
- Adjust the learning rate
- Check if the data preprocessing is appropriate
- Use gradient clipping
- Try different model architectures

### Q: How to choose the appropriate model architecture?
A: It depends on your specific task. LSTM is suitable for time series data, CNN for spatial data, and Transformer excels in long sequence tasks. We recommend starting with simpler models.

## 4. Performance Related

### Q: How to improve training speed?
A: Consider:
- Using GPU training
- Increasing batch size
- Using data preloading
- Optimizing data preprocessing pipeline

### Q: How to handle out-of-memory issues?
A: Try:
- Reducing batch size
- Using data generators
- Reducing model complexity
- Using gradient accumulation

## 5. Other Questions

### Q: How can I contribute to the project?
A: Please refer to our [contribution guide](contributing.md). We welcome all forms of contributions, including code, documentation, and issue reporting.

### Q: Where can I get help?
A: You can:
- Check the detailed [documentation](https://OuyangWenyu.github.io/torchhydro)
- Submit an [Issue](https://github.com/OuyangWenyu/torchhydro/issues) on GitHub
- Join our community discussions