# API Reference

## Data Module (`torchhydro.data`)

### Datasets
- `CamelsDataset`: CAMELS dataset loader for hydrological data
- `HydroDataset`: Base class for hydrological datasets
- `TimeSeriesDataset`: Generic time series dataset class

### Data Processing
- `DataProcessor`: Base class for data processing
- `Scaler`: Data scaling and normalization utilities
- `TimeSeriesProcessor`: Time series specific processing utilities

## Models (`torchhydro.models`)

### Base Models
- `BaseModel`: Abstract base class for all models
- `LSTM`: Long Short-Term Memory network for time series
- `GRU`: Gated Recurrent Unit network
- `Transformer`: Transformer model for hydrological forecasting

### Pre-trained Models
- `load_pretrained`: Load pre-trained model utilities
- `save_model`: Model saving utilities

## Metrics (`torchhydro.metrics`)

### Hydrological Metrics
- `NSE`: Nash-Sutcliffe Efficiency coefficient
- `KGE`: Kling-Gupta Efficiency
- `RMSE`: Root Mean Square Error
- `MAE`: Mean Absolute Error
- `PBias`: Percent Bias

## Transforms (`torchhydro.transforms`)

### Data Transformations
- `Normalize`: Data normalization
- `MissingValueFiller`: Handle missing values
- `TimeSeriesAugmentation`: Time series augmentation techniques
- `TemporalDownsampling`: Temporal resolution adjustment

## Utils (`torchhydro.utils`)

### Utility Functions
- `hydro_utils`: Common hydrological utilities
- `data_utils`: Data handling utilities
- `plot_utils`: Plotting utilities
- `time_utils`: Time series handling utilities

## Configuration (`torchhydro.config`)

### Configuration
- `HydroConfig`: Configuration class for model and training
- `DataConfig`: Configuration for data processing
- `TrainingConfig`: Training specific configuration