"""
GCGRU TgHydro Example - Autoregressive Stream Flow Forecasting with Graph Neural Network

This example demonstrates training an autoregressive GCGRU model on TgHydro dataset
for multi-day stream flow forecasting using graph structure.

To run this script:
1. Set up your data path in `hydro_setting.yml`
2. Prepare TgHydro data with the following structure:
   - attributes/           # Basin attributes
   - timeseries/          # Time series data
   - tggnn/               # TGGNN specific data
     ├── lstmpred.nc      # LSTM prediction data
     └── graph_dict.json  # Graph network structure
3. Run: uv run python experiments/gcgru_tghydro_example.py
"""

import os
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def main():
    """
    Main function to configure and run GCGRU training on TgHydro dataset
    """

    # ================================
    # 1. Data and Path Configuration
    # ================================
    source_path = SETTING["local_data_path"]["datasets-interim"]

    # ================================
    # 2. Basin and Graph Configuration
    # ================================
    # Specify the basin ID
    # TODO: Only support fixed basin ids for GNN modeling in a basin
    basin_id = [
        "sanxia_60406350",
        "sanxia_60406500",
        "sanxia_60406700",
        "sanxia_60407100",
        "sanxia_60407200",
    ]

    # ================================
    # 3. Model Configuration
    # ================================
    model_config = {
        # Model architecture
        "gcgru_out_channels": 32,  # GConvGRU output channels
        "gcgru_K": 15,  # Chebyshev filter size
        "mlp_hidden_dim": 128,  # MLP hidden dimension
        "mlp_dropout": 0.1,  # MLP dropout rate
        "forecast_length": 3,  # Must match forecast_length below
        "teacher_forcing_ratio": 0.0,
    }

    # ================================
    # 4. Training Configuration
    # ================================
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "gcgru_tghydro"),
        # Data source configuration
        source_cfgs={
            "source_name": "tghydrodatasource",
            "source_path": source_path,
            "other_settings": {
                "dataset_name": "tgproject_hengjiang",
                "time_unit": ["1D"],
            },
        },
        # Device configuration: CPU: [-1], GPU: [0], Multi-GPU: [0, 1]
        ctx=[0],
        model_type="MTL",
        # Model selection
        model_name="AutoRegressiveGCGRU",  # Use AutoRegressiveGCGRU
        model_hyperparam=model_config,
        # Dataset configuration - Use MultiFeatureStreamDataset
        dataset="MultiFeatureStreamDataset",
        gage_id=basin_id,
        # Training settings
        batch_size=16,  # Batch size for autoregressive training
        train_epoch=30,  # Number of training epochs
        save_epoch=5,  # Save model every N epochs
        # Sequence lengths
        hindcast_length=30,  # Hindcast/history window length
        forecast_length=3,  # Forecast horizon
        warmup_length=0,  # No warmup for this model
        # Time settings
        min_time_unit="D",  # Daily data
        min_time_interval="1",
        # Input variables (from TgHydro)
        # These should match the column names expected by MultiFeatureStreamDataset
        var_t=[
            "precipitation",  # Will be mapped from total_precipitation_hourly
            "lstm_streamflow",  # Will be mapped from LSTM predictions
            "observed_streamflow",  # Will be mapped from observed discharge
        ],
        t_rm_nan=1,  # Remove NaN from relevant columns (1 = True)
        var_out=["discharge"],  # Target variable
        out_rm_nan=0,  # Don't remove NaN from target columns (0 = False)
        var_c=[
            "area",
            "p_mean",
            "sgr_dk_sav",
        ],
        c_rm_nan=0,  # Don't remove NaN from constant columns (0 = False)
        # Scaler
        scaler="StandardScaler",
        # Model loading for evaluation
        model_loader={
            "load_way": "best",  # Load best model (by validation loss)
        },
        # Date ranges
        train_period=["2021-01-01", "2024-12-31"],
        valid_period=["2021-01-01", "2024-12-31"],
        test_period=["2021-01-01", "2024-12-31"],
        # Loss function and optimizer
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [0.5, 0.5],
        },
        opt="Adam",
        opt_param={"lr": 0.005},  # Learning rate through opt_param
        lr_scheduler={
            0: 1.0,
            10: 0.95,
            20: 0.90,
        },  # Learning rate decay schedule
        # PyG DataLoader and gradient clipping
        use_pyg_dataloader=True,  # Use PyG DataLoader for graph batching
        # Other settings
        which_first_tensor="batch",  # Tensor layout: "batch" first
        rs=1234,  # Random seed
    )

    # ================================
    # 5. Update Config and Run
    # ================================
    config_data = default_config_file()
    update_cfg(config_data, args)

    # Run training and evaluation
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
