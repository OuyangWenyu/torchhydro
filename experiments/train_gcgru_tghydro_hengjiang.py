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
    source_path = SETTING["local_data_path"]["datasets-origin"]
    data_dir = os.path.join(source_path, "jialingjiang_29")

    # ================================
    # 2. Basin and Graph Configuration
    # ================================
    # Specify the basin ID and target station
    basin_id = "sanxia_60703800"
    target_station = "sanxia_60703800"

    # Selected node attributes (must match those available in TgHydroDatasource)
    selected_node_attrs = ["area", "p_mean", "sgr_dk_sav"]

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
        # Node attributes
        "use_node_attr": True,  # Whether to use node attributes
        "node_attr_dim": len(selected_node_attrs),  # Node attribute dimension
    }

    # ================================
    # 4. Training Configuration
    # ================================
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "gcgru_tghydro"),
        # Data source configuration
        source_cfgs={"source_name": "TgHydroDatasource", "source_path": data_dir},
        # Device configuration: CPU: [-1], GPU: [0], Multi-GPU: [0, 1]
        ctx=[0],
        # Model selection
        model_name="MultiFeatureStreamGCGRU",  # Use MultiFeatureStreamGCGRU
        model_hyperparam=model_config,
        # Dataset configuration - Use MultiFeatureStreamDataset
        dataset="MultiFeatureStreamDataset",
        # Graph and TG configuration
        tg_cfgs={
            "basin_id": basin_id,
            "use_confluence_nodes_only": True,  # Only train on confluence nodes
            "node_attributes": selected_node_attrs,  # Will be loaded from datasource
        },
        # Training settings
        batch_size=16,  # Batch size for autoregressive training
        train_epoch=30,  # Number of training epochs
        save_epoch=5,  # Save model every N epochs
        # Sequence lengths
        rho=30,  # Hindcast/history window length
        forecast_length=3,  # Forecast horizon
        warmup_length=0,  # No warmup for this model
        # Time settings
        min_time_unit="D",  # Daily data
        min_time_interval="1",
        # Input variables (from TgHydro)
        # These should match the column names expected by MultiFeatureStreamDataset
        relevant_cols=[
            "precipitation",  # Will be mapped from total_precipitation_hourly
            "lstm_streamflow",  # Will be mapped from LSTM predictions
            "observed_streamflow",  # Will be mapped from observed discharge
        ],
        var_out=["discharge"],  # Target variable
        # Data preprocessing
        relevant_rm_nan=True,
        target_rm_nan=False,
        constant_rm_nan=False,
        # Sampler and scaler
        sampler="HydroSampler",
        scaler="DapengScaler",
        scaler_params={
            "pbm_norm": False,
            "gamma_norm_cols": ["precipitation"],
            "prcp_norm_cols": ["discharge"],
        },
        # Model loading for evaluation
        model_loader={
            "load_way": "best",  # Load best model (by validation loss)
        },
        # Date ranges
        train_period=["2021-01-01", "2024-12-31"],
        valid_period=["2021-01-01", "2024-12-31"],
        test_period=["2021-01-01", "2024-12-31"],
        # Loss function and optimizer
        loss_func="RMSESum",  # or "NSELoss", "MaskedMSE"
        opt="Adam",
        learning_rate=0.005,
        lr_scheduler={
            0: 1.0,
            10: 0.95,
            20: 0.90,
        },  # Learning rate decay schedule
        # Gradient clipping
        max_grad_norm=1.0,
        # PyG DataLoader specific settings
        use_pyg_dataloader=True,  # Use PyG DataLoader for graph batching
        # Other settings
        which_first_tensor="batch",  # Tensor layout: "batch" first
        random_seed=1234,
    )

    # ================================
    # 5. Update Config and Run
    # ================================
    config_data = default_config_file()
    update_cfg(config_data, args)

    # Print configuration summary
    print("=" * 60)
    print("GCGRU TgHydro Training Configuration")
    print("=" * 60)
    print(f"Data Directory: {data_dir}")
    print(f"Basin ID: {basin_id}")
    print(f"Model: {args['model_name']}")
    print(f"Hindcast Length (rho): {args['rho']}")
    print(f"Forecast Length: {args['forecast_length']}")
    print(f"Batch Size: {args['batch_size']}")
    print(f"Training Epochs: {args['train_epoch']}")
    print(f"Device: {'GPU' if args['ctx'][0] >= 0 else 'CPU'}")
    print(f"Use PyG DataLoader: {args.get('use_pyg_dataloader', False)}")
    print("=" * 60)

    # Run training and evaluation
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
