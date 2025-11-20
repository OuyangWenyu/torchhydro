# Usage Guide

This guide demonstrates how to use the `torchhydro` library to configure and run hydrological models. The core workflow revolves around defining a set of parameters, updating a default configuration, and launching the training and evaluation process with a single function.

## Core Concept

The main workflow consists of three steps:

1.  **Define Parameters**: You start by defining all your experiment's parameters (like model choice, dataset, variables, and hyperparameters) using the `torchhydro.configs.config.cmd` function. This creates a parameter object.
2.  **Update Configuration**: The default configuration is loaded using `torchhydro.configs.config.default_config_file()`. Then, your custom parameters are merged into it using `torchhydro.configs.config.update_cfg()`.
3.  **Train and Evaluate**: Finally, you pass the consolidated configuration dictionary to the `torchhydro.trainers.trainer.train_and_evaluate()` function, which handles the entire pipeline: data loading, model building, training, and evaluation.

```python
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 1. Define parameters for your experiment
args = cmd(...) 

# 2. Load default config and update it with your parameters
config_data = default_config_file()
update_cfg(config_data, args)

# 3. Run the training and evaluation pipeline
train_and_evaluate(config_data)
```

Below are two practical examples demonstrating this workflow.

## Example 1: Training a Standard LSTM on CAMELS Data

This example shows how to train a standard LSTM model for streamflow prediction using the CAMELS-US dataset.

### Step 1: Define Parameters

First, we define all the necessary parameters for our experiment. This includes data source, model type, variables, time periods, and training settings.

```python
import os
from torchhydro.configs.config import cmd
from torchhydro import SETTING

# It's recommended to set your local data path in SETTING
# For example: SETTING["local_data_path"]["datasets-origin"] = "/path/to/your/data"
source_path = SETTING["local_data_path"]["datasets-origin"]

args = cmd(
    # Experiment name and output directory
    sub="test_camels/exp1",
    
    # Data source configuration
    source_cfgs={
        "source_name": "camels_us",
        "source_path": source_path,
    },
    
    # Use CPU for this example
    ctx=[-1],
    
    # Model selection and hyperparameters
    model_name="CpuLSTM",
    model_hyperparam={
        "n_input_features": 23,
        "n_output_features": 1,
        "n_hidden_states": 256,
    },
    
    # Basin IDs for training and evaluation
    gage_id=[
        "01013500", "01022500", "01030500", "01031500", "01047000",
        "01052500", "01054200", "01055000", "01057000", "01170100",
    ],
    
    # Training settings
    batch_size=8,
    train_epoch=2,
    save_epoch=1,
    
    # Sequence lengths
    hindcast_length=0,
    forecast_length=20,
    
    # Time settings
    min_time_unit="D",
    min_time_interval="1",
    
    # Input and output variables
    var_t=[
        "precipitation", "daylight_duration", "solar_radiation",
        "temperature_max", "temperature_min", "vapor_pressure",
    ],
    var_out=["streamflow"],
    
    # Data components
    dataset="StreamflowDataset",
    sampler="KuaiSampler",
    scaler="DapengScaler",
    
    # Model loading configuration for evaluation
    model_loader={"load_way": "specified", "test_epoch": 2},
    
    # Date ranges for training, validation, and testing
    train_period=["2000-10-01", "2001-10-01"],
    valid_period=["2001-10-01", "2002-10-01"],
    test_period=["2002-10-01", "2003-10-01"],
    
    # Loss function and optimizer
    loss_func="RMSESum",
    opt="Adam",
    lr_scheduler={0: 1, 1: 0.5, 2: 0.2},
    
    # Tensor layout
    which_first_tensor="sequence",
)
```

### Step 2: Run the Pipeline

With the parameters defined, we simply call the update and train functions.

```python
from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# Load default config and update it
config_data = default_config_file()
update_cfg(config_data, args)

# Run training and evaluation
train_and_evaluate(config_data)
```

The library will now handle data loading, preprocessing, model training, and finally, evaluation on the test set. Results will be saved in the directory specified by the `sub` parameter (e.g., `results/test_camels/exp1`).

## Example 2: Training a Physics-Informed Model (DPL-XAJ)

This example demonstrates a more advanced use case: training a Differentiable Parameter Learning (DPL) model. Here, an LSTM is coupled with the Xinanjiang (XAJ) hydrological model. The network learns to output the parameters of the XAJ model.

### Step 1: Define DPL Parameters

The parameter definition is similar, but we specify a different model (`DplAttrXaj`), dataset (`DplDataset`), and some additional hyperparameters specific to the physics-informed approach.

```python
import os
from torchhydro.configs.config import cmd
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING

source_path = SETTING["local_data_path"]["datasets-origin"]

dpl_args = cmd(
    sub="test_camels/expdpl001",
    source_cfgs={
        "source_name": "camels_us",
        "source_path": source_path,
    },
    ctx=[0],  # Use GPU 0
    
    # DPL model selection
    model_name="DplAttrXaj",
    model_hyperparam={
        "n_input_features": 17,
        "n_output_features": 15, # Number of XAJ model parameters
        "n_hidden_states": 256,
        "kernel_size": 15,
        "warmup_length": 30, # Warm-up period for the hydrological model
        "param_limit_func": "clamp",
    },
    
    # DPL models often require a specialized dataset
    dataset="DplDataset",
    
    # Use a hybrid loss function for multiple outputs
    loss_func="MultiOutLoss",
    loss_param={
        "loss_funcs": "RMSESum",
        "data_gap": [0, 0],
        "device": [0],
        "item_weight": [1, 0], # Weight for streamflow vs. other outputs
        "limit_part": [1],
    },
    
    # Special scaler settings for physical models
    scaler="DapengScaler",
    scaler_params={
        "prcp_norm_cols": ["streamflow"],
        "gamma_norm_cols": [
            StandardVariable.PRECIPITATION,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        ],
        "pbm_norm": True,
    },
    
    gage_id=["01013500", "01022500", "01030500", "01031500"],
    train_period=["1985-10-01", "1986-04-01"],
    test_period=["2000-10-01", "2001-10-01"],
    valid_period=None,
    
    batch_size=50,
    forecast_length=60,
    warmup_length=30,
    
    # Input variables for the neural network part
    var_t=[
        StandardVariable.PRECIPITATION,
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
    ],
    # Output variables (streamflow from XAJ, plus a dummy variable)
    var_out=[StandardVariable.STREAMFLOW, StandardVariable.EVAPOTRANSPIRATION],
    n_output=2,
    
    train_epoch=2,
    opt="Adadelta",
)
```

### Step 2: Run the DPL Pipeline

The execution step is identical.

```python
from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# Load default config and update it
config_data = default_config_file()
update_cfg(config_data, dpl_args)

# Run training and evaluation
train_and_evaluate(config_data)
```

## Future Development

This guide provides a starting point. `torchhydro` is designed to be flexible and supports a growing number of models, datasets, and data sources.

-   The full list of available **models** can be found in `torchhydro/models/model_dict_function.py`.
-   The full list of available **dataset** types is in `torchhydro/datasets/data_dict.py`.
-   The supported **data sources** are defined in `torchhydro/datasets/data_sources.py`.

Future versions of this documentation will provide a comprehensive API reference and more detailed examples for all supported components.
