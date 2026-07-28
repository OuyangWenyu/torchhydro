"""
GCGRU TgHydro Example - Autoregressive Stream Flow Forecasting with Graph Neural Network

This example demonstrates training an autoregressive GCGRU model on TgHydro dataset
for multi-day stream flow forecasting using graph structure.

To run this script:
1. Set up your data path in `hydro_setting.yml`
2. Prepare TgHydro data with the following structure:
    - attributes/          # 控制流域属性数据
    - timeseries/          # 控制流域时间序列数据
    - shapes/              # 控制流域形状文件
    - intermediate/        # 区间流域数据
      ├── attributes/      # 区间流域属性数据，包含拓扑关系
      ├── timeseries/      # 区间流域时间序列数据
      └── shapes/          # 区间流域形状文件
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
        "gcgru_K": 3,  # Chebyshev filter size
        "mlp_hidden_dim": 128,  # MLP hidden dimension
        "mlp_dropout": 0.1,  # MLP dropout rate
        "forecast_length": 72,  # Must match forecast_length below
        "teacher_forcing_ratio": 0.4,
        "input_size": 7, # var_t length (4) + var_c length (3)
    }

    # ================================
    # 4. Training Configuration
    # ================================
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("hengjiang_hourly_test", "loss1-36-alltrain"),
        # Data source configuration
        source_cfgs={
            "source_name": "tghydrodatasource",
            "source_path": source_path,
            "other_settings": {
                "dataset_name": "era5hengjiangqujian_hourly",
                "time_unit": ["1h"],
            },
        },
        # Device configuration: CPU: [-1], GPU: [0], Multi-GPU: [0, 1]
        ctx=[2],
        # model_type="MTL",
        model_type="Normal",
        # Model selection
        model_name="AutoRegressiveGCGRU",  # Use AutoRegressiveGCGRU
        model_hyperparam=model_config,
        # Dataset configuration - Use TgHydroDataset
        dataset="TgHydroDataset",
        gage_id=basin_id,
        # Training settings
        batch_size=256,  # Batch size for autoregressive training
        train_epoch=10,  # Number of training epochs
        save_epoch=1,  # Save model every N epochs
        # Sequence lengths
        hindcast_length=168,  # Hindcast/history window length
        forecast_length=72,  # Forecast horizon
        warmup_length=0,  # No warmup for this model
        # Time settings
        min_time_unit="h",  # Hourly data
        min_time_interval=1,
        # Input variables (from TgHydro)
        var_t=[
            "total_precipitation_hourly",  # Will be mapped from total_precipitation_hourly
            "discharge",  # Will be mapped from observed discharge
            "lstm_pred",  # Will be mapped from LSTM predictions
            "streamflow",  # Will be mapped from observed streamflow
        ],
        t_rm_nan=1,  # Remove NaN from relevant columns (1 = True)
        var_out=["streamflow"],  # Target variable
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
        # train_period=["2021-06-01", "2021-6-11"],
        # valid_period=["2021-06-01", "2021-6-11"],
        # test_period=["2021-06-01", "2021-6-11"],
        # Loss function and optimizer
        loss_func="PyGraphLoss",
        loss_param={
            "loss_func": "RMSE",
            "horizon_range": [0, 36],
        },
        opt="Adam",
        opt_param={"lr": 0.002},  # Learning rate through opt_param
        lr_scheduler={
            0: 1.0,
            5: 0.95,
            10: 0.90,
        },  # Learning rate decay schedule
        # PyG DataLoader and gradient clipping
        use_pyg_dataloader=True,  # Use PyG DataLoader for graph batching
        # Gradient clipping configuration for stability
        # max_grad_norm is read by Fabric wrapper when enabled
        # Otherwise, standard torch.nn.utils.clip_grad_norm_ is used
        fabric_strategy=None,
        evaluator={
            "eval_way": "rolling",
            "stride": 1,
            "recover_mode": "byforecast",
        },
        rolling=1,
        metrics=["NSE", "RMSE", "KGE", "Corr", "FHV", "FLV"],
        # TgHydro specific configs: per-variable source selection
        tg_cfgs={
            "var_source_map": {
                # 输入变量的源选择（base/intermediate/both）
                "total_precipitation_hourly": "intermediate",
                "discharge": "intermediate",
                "lstm_streamflow": "intermediate",
                "streamflow": "base",
                # 常量属性也支持源选择（如部分来自中间处理）
                "area": "both",
                "p_mean": "intermediate",
                "sgr_dk_sav": "intermediate",
            },
            "default_source": "base",
            "prefer_intermediate": False,
            "combine": True,
        },
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