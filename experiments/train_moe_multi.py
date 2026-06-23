"""
This is an example of training a Mixture-of-Experts (MoE) model on MULTIPLE CAMELS datasets.

It uses the MultiSourceDataset to read from various configured sources.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.multi_dataset_reader import MultiDatasetReader, DATASET_MAPPING

import torch
import numpy as np


SEED = 42
N_FOLDS = 5


# show tensor shape in vscode debugger
def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main():
    """
    This function defines the parameters for the experimeent, updates the default configuration,
    and then runs the training and evaluation pipeline for the MoE Model using multiple datasets.
    """
    # 1. Define DATASETS to use
    # You can specify a list of datasets (e.g. ["camels_us", "camels_cl"])
    # Or use all available datasets: list(DATASET_MAPPING.keys())
    # Note: Using ALL datasets might result in very large data loading times and memory usage

    # Example: specific datasets
    DATASETS = ["camels_us", "camels_cl", "camels_aus", "camels_col"]
    # DATASETS = ["camels_us"]
    # If you want to run on ALL supported datasets, uncomment below:
    # DATASETS = list(DATASET_MAPPING.keys())

    print(f"Using datasets: {DATASETS}")

    # Use MultiDatasetReader to collect IDs for these datasets
    reader = MultiDatasetReader(datasets=DATASETS)
    # Set force_refresh=True to ensure we collect IDs for all requested datasets,
    # otherwise it might load a partial cache (e.g. only camels_us from a previous run)
    id_mapping = reader.collect_all_ids(force_refresh=False)

    # Simply flatten the list of all IDs
    all_gage_ids = []
    for ds_name in DATASETS:
        ids = id_mapping.get(ds_name, [])
        print(sorted(ids))
        all_gage_ids.extend(ids)

    # Ensure unique IDs just in case, though MultiSourceDataset handles duplicates if configured properly
    # Here we just pass the list of IDs we want to train on.
    all_gage_ids = list(set(all_gage_ids))
    print(f"Total stations found: {len(all_gage_ids)}")

    rng = np.random.default_rng(SEED)

    all_gage_ids = np.array(all_gage_ids)
    all_gage_ids.sort()
    rng.shuffle(all_gage_ids)

    folds = np.array_split(all_gage_ids, N_FOLDS)

    cv_splits = []

    for fold_idx in range(N_FOLDS):
        val_ids = folds[fold_idx].tolist()
        train_ids = np.concatenate(
            [folds[i] for i in range(N_FOLDS) if i != fold_idx]
        ).tolist()
        train_ids.sort()
        val_ids.sort()
        cv_splits.append({"fold": fold_idx, "train_ids": train_ids, "val_ids": val_ids})

        print(f"Fold {fold_idx}: " f"train={len(train_ids)}, val={len(val_ids)}")

    fold = 0
    train_ids = cv_splits[fold]["train_ids"]
    val_ids = cv_splits[fold]["val_ids"]
    # Optionally limit for testing
    # all_gage_ids = all_gage_ids[:10]

    # Define common model parameters
    n_input_features = 68
    n_output_features = 1
    n_hidden_states = 256
    forecast_length = 365
    num_experts = 8

    # We don't need a single source_path for MultiSourceDataset, but config requires it not be empty usually
    # or the dataset class handles it.
    # However, source_cfgs allows passing 'datasets' list.
    source_path = SETTING["local_data_path"]["datasets-origin"]

    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("results", "moe_multi_camels"),
        # Data source configuration for MultiSourceDataset
        source_cfgs={
            "source_name": "camels_us",  # nominal default, won't be used as single source
            "source_path": source_path,
            "datasets": DATASETS,  # Pass the list of datasets here
            "attributes_paths": {
                "camels_us": "/Users/cylenlc/Desktop/shp文件/camelsus/mean_result.csv",
                "camels_cl": "/Users/cylenlc/Desktop/shp文件/camelscl/mean_result.csv",
                "camels_aus": "/Users/cylenlc/Desktop/shp文件/camelsaus/mean_result.csv",
                "camels_col": "/Users/cylenlc/Desktop/shp文件/camelscol/mean_result.csv",
                "camels_de": "/Users/cylenlc/Desktop/shp文件/camelsde/mean_result.csv",
            },
        },
        # Use None to let Fabric handle devices based on the launch command (fabric run)
        # Setting it to "auto" or a list will conflict with CLI args
        ctx=None,
        # --- Model selection and hyperparameters for MoE Model ---
        model_name="DenseMoE",
        model_hyperparam={
            "n_experts": num_experts,
            "static_features": 64,
            # Configuration for the two LSTM experts
            "experts_cfgs": [
                {
                    "model_name": "CpuLSTM",
                    "model_hyperparam": {
                        "n_input_features": n_input_features,
                        "n_output_features": n_output_features,
                        "n_hidden_states": n_hidden_states,
                    },
                }
                for _ in range(num_experts)
            ],
            # Configuration for the gating network (a simple MLP)
            "gating_cfgs": {
                "model_name": "SimpleAnn",
                "model_hyperparam": {
                    # Input to the gate will now be from the last time step
                    "nx": 64,
                    "hidden_size": [64, 16],
                },
            },
        },
        # Basin IDs for training and evaluation
        gage_id=train_ids,
        # Training settings
        batch_size=256,
        train_epoch=20,
        save_epoch=1,
        # Sequence lengths
        hindcast_length=0,
        forecast_length=forecast_length,
        # Time settings
        min_time_unit="D",
        min_time_interval="1",
        # Input and output variables
        var_c=[],
        var_t=[
            StandardVariable.PRECIPITATION,
            # StandardVariable.DAYLIGHT_DURATION,
            # StandardVariable.SOLAR_RADIATION,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            # StandardVariable.VAPOR_PRESSURE,
        ],
        var_out=[StandardVariable.STREAMFLOW],
        # USE THE NEW MULTI-SOURCE DATASET
        dataset="MultiSourceAEFDataset",
        scaler="DapengScaler",
        model_loader={
            "load_way": "pth",
            "pth_path": f"/Users/cylenlc/work/torchhydro/experiments/results/camels/mtslstm_DapengScaler_d-h_2freq_14dwin/best_model.pth",
        },
        # Date ranges - make sure this range covers data availability for chosen datasets
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={
            "lr_factor": 0.95,
        },
        which_first_tensor="sequence",
        evaluator={
            "eval_way": "1pace",
            "pace_idx": -1,
        },
        early_stopping=True,
        rolling=1,
        patience=2,
        model_type="Normal",
        # fabric_strategy="auto",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
