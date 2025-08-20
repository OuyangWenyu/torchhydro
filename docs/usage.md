# Usage Guide

## Quick Start

### 1. Load Dataset

```python
from torchhydro.data import CamelsDataset

# Load CAMELS dataset
dataset = CamelsDataset(
    root="path/to/camels/data",
    basin_ids=None,  # Set to None to load all basins
    input_vars=['prcp', 'temp', 'pet'],  # Input variables
    target_vars=['streamflow'],  # Target variables
    seq_length=365  # Sequence length
)
```

### 2. Create Data Loader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### 3. Define Model

```python
from torchhydro.models import LSTM

model = LSTM(
    input_size=3,  # Input feature dimension
    hidden_size=64,
    num_layers=2,
    output_size=1  # Output dimension
)
```

### 4. Train Model

```python
import torch
from torchhydro.metrics import NSE  # Nash-Sutcliffe Efficiency coefficient

# Define loss function and optimizer
criterion = NSE()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Advanced Usage

### Custom Data Preprocessing

```python
from torchhydro.transforms import Normalize, MissingValueFiller

# Define transformations
transforms = [
    MissingValueFiller(strategy='mean'),
    Normalize(method='minmax')
]

dataset = CamelsDataset(
    transforms=transforms,
    ...
)
```

### Using Pre-trained Models

```python
from torchhydro.models import load_pretrained

# Load pre-trained model
model = load_pretrained('lstm_camels_us')

# Fine-tune the model
model.fine_tune(new_dataset)
```

### Model Evaluation

```python
from torchhydro.metrics import NSE, KGE, RMSE

# Define multiple evaluation metrics
metrics = [NSE(), KGE(), RMSE()]

# Evaluate model
results = model.evaluate(test_loader, metrics)
```

## More Examples

For more detailed examples, please refer to our [examples repository](https://github.com/OuyangWenyu/torchhydro/tree/main/examples):

- Basic Streamflow Prediction
- Multi-site Prediction
- Transfer Learning
- Uncertainty Estimation
- Physics-Informed Deep Learning