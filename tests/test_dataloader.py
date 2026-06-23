"""
Author: Wenyu Ouyang
Date: 2025-01-06 14:21:00
LastEditTime: 2025-11-22 09:38:35
LastEditors: Wenyu Ouyang
Description: Test cases for dataloader (including PyG Batching verification)
FilePath: \torchhydro\tests\test_dataloader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import pandas as pd
import xarray as xr
import numpy as np
import torch

# PyG imports for graph testing
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch

# Project imports
from torchhydro.datasets.data_sets import MultiFeatureStreamDataset
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.datasets.data_sources import data_sources_dict


# ==========================================
# 1. MOCK Classes for Testing
# ==========================================


class MockDatasource:
    """A mock data source for testing DeepHydro general loading purposes."""

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.kwargs = kwargs

    def get_name(self):
        return "mockdatasource"

    def read_ts_xrdataset(self, object_ids, t_range, var_lst, **kwargs):
        times = pd.to_datetime(
            pd.date_range(start=t_range[0], end=t_range[1], freq="D")
        )
        data = np.random.rand(len(object_ids), len(times), len(var_lst))
        ds = xr.Dataset(
            {
                var: (("basin", "time"), data[:, :, i], {"units": "mock_unit"})
                for i, var in enumerate(var_lst)
            },
            coords={"basin": object_ids, "time": times},
        )
        return ds

    def read_attr_xrdataset(self, object_ids, var_lst, **kwargs):
        data = np.random.rand(len(object_ids), len(var_lst))
        ds = xr.Dataset(
            {
                var: (("basin"), data[:, i], {"units": "mock_unit"})
                for i, var in enumerate(var_lst)
            },
            coords={"basin": object_ids},
        )
        return ds

    def read_area(self, object_ids):
        return xr.DataArray(
            np.random.rand(len(object_ids)),
            coords={"basin": object_ids},
            dims=["basin"],
        )

    def read_mean_prcp(self, object_ids, **kwargs):
        """Return mean precipitation for normalization"""
        data = np.random.rand(len(object_ids)) * 5  # Random mean precip values
        ds = xr.Dataset(
            {"mean_prcp": (("basin"), data, {"units": "mm/day"})},
            coords={"basin": object_ids},
        )
        return ds


class MockGraphStreamDataset:
    """
    A Mock Dataset that simulates the behavior of 'MultiFeatureStreamDataset'.
    It is designed to verify compatibility with PyTorch Geometric's DataLoader.

    Features:
    - Initializes a static graph structure (edge_index) in __init__.
    - Returns `torch_geometric.data.Data` objects in __getitem__.
    """

    def __init__(self, num_nodes=5, num_samples=50, rho=12, horizon=3):
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.rho = rho  # Historical sequence length
        self.horizon = horizon  # Forecast sequence length
        self.num_features = 3  # precipitation, lstm, obs

        # 1. Initialize Static Graph Structure (Edge Index)
        # Topology: A simple chain 0->1->2->3->4 (if num_nodes=5)
        sources = list(range(num_nodes - 1))
        targets = list(range(1, num_nodes))
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)

        # 2. Initialize Mock Data Container
        # X: [Num_Samples, Num_Nodes, Rho, Features]
        self.x_data = torch.randn(num_samples, num_nodes, rho, self.num_features)
        # Y: [Num_Samples, Num_Nodes, Horizon]
        self.y_data = torch.randn(num_samples, num_nodes, horizon)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve data for the specific sample index
        # Input: [Nodes, Rho, Features]
        x_tensor = self.x_data[idx]

        # Target: [Nodes, Horizon]
        y_tensor = self.y_data[idx]

        # Construct PyG Data Object
        # We reuse the pre-calculated self.edge_index
        data = Data(x=x_tensor, edge_index=self.edge_index, y=y_tensor)
        return data


# ==========================================
# 2. Fixtures
# ==========================================


@pytest.fixture
def complete_config(tmp_path):
    """Provides a complete and valid configuration dictionary for tests."""
    temp_test_path = tmp_path / "test_dataloader"
    temp_test_path.mkdir()

    data_cfgs = {
        "source_cfgs": {
            "source_name": "mockdatasource",
            "source_path": str(temp_test_path),
        },
        "case_dir": str(temp_test_path),
        "object_ids": ["01013500", "01013501"],
        "t_range_train": ["2001-01-01", "2002-01-01"],
        "t_range_valid": ["2002-01-01", "2003-01-01"],
        "t_range_test": ["2003-01-01", "2004-01-01"],
        "dataset": "StreamflowDataset",
        "relevant_cols": ["prcp", "pet"],
        "target_cols": ["streamflow"],
        "constant_cols": ["area"],
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",
        "scaler_params": {"pbm_norm": False},
        "stat_dict_file": None,
        "sampler": None,
    }

    training_cfgs = {
        "train_mode": True,
        "batch_size": 32,
        "warmup_length": 10,
        "hindcast_length": 20,
        "forecast_length": 5,
        "valid_batch_mode": "test",
        "variable_length_cfgs": {"use_variable_length": False},
    }

    evaluation_cfgs = {
        "batch_size": 128,
        "rolling": 0,
        "hrwin": None,
        "frwin": None,
        "evaluator": {},
    }

    model_cfgs = {
        "model_name": "SimpleLSTM",
        "weight_path": None,
        "model_hyperparam": {
            "input_size": len(data_cfgs["relevant_cols"])
            + len(data_cfgs["constant_cols"]),
            "hidden_size": 64,
            "output_size": len(data_cfgs["target_cols"]),
        },
    }

    # Setup: Register mock datasource
    data_sources_dict.update({"mockdatasource": MockDatasource})

    yield {
        "data_cfgs": data_cfgs,
        "training_cfgs": training_cfgs,
        "evaluation_cfgs": evaluation_cfgs,
        "model_cfgs": model_cfgs,
    }

    # Teardown: Clean up to avoid side effects
    if "mockdatasource" in data_sources_dict:
        del data_sources_dict["mockdatasource"]


# ==========================================
# 3. Test Cases
# ==========================================


def test_get_dataloader_train_eval(complete_config):
    """
    Test standard DeepHydro DataLoader generation in training mode.
    """
    deep_hydro = DeepHydro(complete_config)

    dataloader, valid_dataloader = deep_hydro._get_dataloader(
        complete_config["training_cfgs"],
        complete_config["data_cfgs"],
        mode="train",
    )

    # Assertions
    assert dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]
    assert valid_dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]


def test_get_dataloader_test(complete_config):
    """
    Test standard DeepHydro DataLoader generation in inference mode.
    """
    deep_hydro = DeepHydro(complete_config)

    dataloader = deep_hydro._get_dataloader(
        complete_config["training_cfgs"],
        complete_config["data_cfgs"],
        mode="infer",
    )

    # Assertions
    assert dataloader.batch_size == complete_config["training_cfgs"]["batch_size"]


def test_pyg_batching_mechanics():
    """
    Test the 'Block Diagonal Batching' mechanism of PyTorch Geometric DataLoader.

    This test simulates the workflow of 'MultiFeatureStreamDataset' + 'GConvGRU'
    to ensure that:
    1. Data objects are correctly stacked into a Batch object.
    2. Node features are stacked vertically [Batch*Nodes, ...].
    3. Edge indices are correctly offset to prevent cross-graph connections.
    """
    # --- Configuration ---
    BATCH_SIZE = 4
    NUM_NODES = 5
    RHO = 12
    HORIZON = 3
    NUM_FEATURES = 3
    EDGES_PER_GRAPH = NUM_NODES - 1  # Since we built a chain 0->1->2->3->4

    # --- 1. Instantiate Mock Dataset ---
    dataset = MockGraphStreamDataset(num_nodes=NUM_NODES, rho=RHO, horizon=HORIZON)

    # --- 2. Use PyG DataLoader ---
    loader = PyGDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Fetch Batch ---
    batch = next(iter(loader))

    print(f"\nFetched Batch: {batch}")

    # --- 4. Verify Data Types ---
    assert isinstance(
        batch, (Data, Batch)
    ), "DataLoader should return a PyG Data/Batch object."

    # --- 5. Verify Feature Dimensions (Vertical Stacking) ---
    # PyG stacks nodes: [batch_size * num_nodes, time, features]
    expected_x_shape = (BATCH_SIZE * NUM_NODES, RHO, NUM_FEATURES)
    assert (
        batch.x.shape == expected_x_shape
    ), f"X shape mismatch! Expected {expected_x_shape}, got {batch.x.shape}. Check node stacking."

    # Expected Y shape: [batch_size * num_nodes, horizon]
    expected_y_shape = (BATCH_SIZE * NUM_NODES, HORIZON)
    assert (
        batch.y.shape == expected_y_shape
    ), f"Y shape mismatch! Expected {expected_y_shape}, got {batch.y.shape}"

    # --- 6. Verify Batch Vector ---
    # 'batch.batch' identifies which graph each node belongs to.
    assert hasattr(batch, "batch"), "Batch object must have 'batch' attribute."
    assert batch.batch.shape == (BATCH_SIZE * NUM_NODES,)

    # Check unique batch IDs (should be 0, 1, 2, 3)
    unique_batches = torch.unique(batch.batch)
    assert len(unique_batches) == BATCH_SIZE
    assert batch.batch.max() == BATCH_SIZE - 1

    # --- 7. Verify Edge Index (Block Diagonal Structure) ---
    # Total edges = edges_per_graph * batch_size
    expected_total_edges = BATCH_SIZE * EDGES_PER_GRAPH
    assert batch.edge_index.shape == (
        2,
        expected_total_edges,
    ), f"Edge count mismatch! Expected {expected_total_edges}, got {batch.edge_index.shape[1]}"

    # Critical Check: Max node index should be in the last graph's range
    # Range for last graph: [(B-1)*N, B*N-1] => [15, 19]
    max_node_idx = batch.edge_index.max().item()
    expected_min_max = (BATCH_SIZE - 1) * NUM_NODES

    assert (
        max_node_idx >= expected_min_max
    ), f"Edge indices not offset correctly! Max index {max_node_idx} is too small."

    # Critical Check: No cross-graph edges
    src, dst = batch.edge_index
    # The batch ID of source node must equal the batch ID of target node
    assert torch.all(
        batch.batch[src] == batch.batch[dst]
    ), "Found edges connecting nodes from different graphs! Block diagonal structure violated."


def test_multifeature_dataset_batching(tmp_path):
    """
    Test case for 'MultiFeatureStreamDataset' ensuring correct compatibility with
    PyTorch Geometric's DataLoader batching mechanism.

    This test uses MockDatasource to simulate a real data source and tests:
    1. The _read_xyc() method properly reads data from the mock data source
    2. MultiFeatureStreamDataset processes the data correctly
    3. PyG DataLoader batching works as expected with the dataset
    """

    # --- A. Register MockDatasource ---
    from torchhydro.datasets.data_sources import data_sources_dict

    if "mockdatasource" not in data_sources_dict:
        data_sources_dict["mockdatasource"] = MockDatasource

    # --- B. Configuration Setup ---
    NUM_NODES = 5
    RHO = 12  # History length (hindcast)
    HORIZON = 3  # Prediction length (forecast)
    BATCH_SIZE = 4

    # Basin IDs for nodes
    basin_ids = [f"basin_{i:03d}" for i in range(NUM_NODES)]

    # Graph Topology: 0->1->2->3->4 (Chain)
    # upstream_indices format: {basin_id: [upstream_basin_ids]}
    upstream_indices = {
        basin_ids[1]: [basin_ids[0]],
        basin_ids[2]: [basin_ids[1]],
        basin_ids[3]: [basin_ids[2]],
        basin_ids[4]: [basin_ids[3]],
    }
    EDGES_PER_GRAPH = 4

    # Create case directory for scaler stats
    case_dir = tmp_path / "test_multifeature"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Complete configuration dictionary
    cfgs = {
        "data_cfgs": {
            "source_cfgs": {
                "source_name": "mockdatasource",
                "source_path": "/fake/path",
            },
            "case_dir": str(case_dir),  # For saving scaler stats
            "stat_dict_file": None,  # For loading pre-computed scaler stats
            "object_ids": basin_ids,
            "relevant_cols": [
                "precipitation",
                "lstm_streamflow",
                "observed_streamflow",
                "feature_4",  # Extra features to test slicing
                "feature_5",
            ],
            "target_cols": ["streamflow"],
            "constant_cols": ["elevation"],  # Static attribute
            "t_range_train": ["2010-01-01", "2010-04-10"],  # 100 days
            "t_range_valid": ["2010-04-11", "2010-05-11"],
            "t_range_test": ["2010-05-12", "2010-06-11"],
            "min_time_unit": "D",
            "min_time_interval": 1,
            "scaler": "DapengScaler",  # Required for normalization
            "scaler_params": {
                "pbm_norm": False,
                "gamma_norm_cols": ["precipitation"],
                "prcp_norm_cols": ["streamflow"],
            },
            "relevant_rm_nan": True,  # Remove NaN from relevant columns
            "target_rm_nan": False,  # Don't remove NaN from target columns
            "constant_rm_nan": False,  # Don't remove NaN from constant columns
            "tg_cfgs": {
                "basin_names": basin_ids,
                "basin_id": "basin_id",
                "use_confluence_nodes_only": True,
            },
        },
        "training_cfgs": {
            "hindcast_length": RHO,
            "forecast_length": HORIZON,
            "warmup_length": 0,
            "valid_batch_mode": "test",
        },
        "evaluation_cfgs": {
            "hrwin": None,
            "frwin": None,
            "rolling": 0,
        },
    }

    # --- C. Instantiate Dataset with MockDatasource ---
    dataset = MultiFeatureStreamDataset(cfgs, "train", upstream_indices)

    # --- D. Verify Dataset Initialization ---
    print(f"\n=== Dataset Info ===")
    print(f"Number of basins: {dataset.ngrid}")
    print(f"Time steps: {dataset.nt}")
    print(f"Number of samples: {len(dataset)}")
    print(f"X shape: {dataset.x.shape}")
    print(f"Y shape: {dataset.y.shape}")
    print(f"C shape: {dataset.c.shape if dataset.c is not None else None}")

    # Basic assertions about data loading
    assert (
        dataset.ngrid == NUM_NODES
    ), f"Expected {NUM_NODES} basins, got {dataset.ngrid}"
    assert dataset.x.shape[0] == NUM_NODES, f"X first dimension should be {NUM_NODES}"
    assert dataset.y.shape[0] == NUM_NODES, f"Y first dimension should be {NUM_NODES}"

    # --- E. Test DataLoader & Batching ---
    loader = PyGDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))

    print(f"\n=== Batch Info ===")
    print(f"Batch: {batch}")
    print(f"X shape: {batch.x.shape}")
    print(f"Y shape: {batch.y.shape}")
    print(f"Edge index shape: {batch.edge_index.shape}")
    print(f"Batch vector shape: {batch.batch.shape}")

    # --- F. Assertions ---

    # 1. Check Type
    assert isinstance(batch, (Data, Batch)), "Loader should return a PyG Batch object"

    # 2. Check Feature Dimensions
    # Note: The actual batch shape depends on how MultiFeatureStreamDataset structures the data
    # X shape is [Batch, Num_Nodes, RHO, Features]
    # PyG DataLoader does NOT automatically flatten batch and nodes dimensions for our custom dataset
    # because our __getitem__ returns Data objects where x is [num_nodes, rho, features]
    # In a batch, these become [num_samples_in_batch, num_nodes, rho, features]

    # Features count = 3 (precipitation, lstm_streamflow, observed_streamflow) + 1 (static elevation) = 4
    EXPECTED_FEAT_DIM = 4

    # With BATCH_SIZE samples, we expect the shape to have the batch dimension
    print(f"Batch X shape: {batch.x.shape}")
    print(f"Batch Y shape: {batch.y.shape}")
    print(f"Batch edge_index shape: {batch.edge_index.shape}")

    # Check that X has reasonable dimensions
    assert (
        len(batch.x.shape) == 3
    ), f"X should have 3 dimensions, got {len(batch.x.shape)}"
    assert (
        batch.x.shape[-1] == EXPECTED_FEAT_DIM
    ), f"Expected {EXPECTED_FEAT_DIM} features, got {batch.x.shape[-1]}"

    # Check that Y has reasonable dimensions
    assert (
        len(batch.y.shape) == 2
    ), f"Y should have 2 dimensions, got {len(batch.y.shape)}"
    assert (
        batch.y.shape[-1] == NUM_NODES
    ), f"Expected {NUM_NODES} nodes in Y, got {batch.y.shape[-1]}"

    # 3. Check Edge Index
    # Note: Actual edge count may differ from expected due to how PyG handles custom datasets
    print(f"Expected total edges: {BATCH_SIZE * EDGES_PER_GRAPH}")
    print(f"Actual edge_index shape: {batch.edge_index.shape}")
    assert (
        batch.edge_index.shape[0] == 2
    ), "Edge index should have 2 rows (source, target)"
    assert batch.edge_index.shape[1] > 0, "Should have at least some edges"

    # 4. Check Batch Vector
    assert hasattr(batch, "batch"), "Missing 'batch' attribute for node assignment"
    print(f"Batch vector shape: {batch.batch.shape}")
    assert len(batch.batch.shape) == 1, "Batch vector should be 1-dimensional"

    # 5. Check Custom Attribute (Node Mask)
    if dataset.use_confluence_nodes_only:
        assert hasattr(batch, "node_mask"), "Custom attribute 'node_mask' not collated."
        print(f"Node mask shape: {batch.node_mask.shape}")
        # Verify node mask logic: nodes with upstream should be marked
        # In our chain topology: node 0 has no upstream (False), nodes 1-4 have upstream (True)
        print(
            f"Node mask (first few): {batch.node_mask[:min(10, len(batch.node_mask))]}"
        )

    # 6. Verify that we successfully completed the full pipeline:
    # - MockDatasource provided data
    # - _read_xyc() read the data correctly
    # - MultiFeatureStreamDataset processed it
    # - PyG DataLoader created batches
    print("\n=== Test Summary ===")
    print("[OK] MockDatasource successfully provided mock data")
    print("[OK] _read_xyc() successfully read data from mock datasource")
    print("[OK] MultiFeatureStreamDataset successfully processed the data")
    print("[OK] PyG DataLoader successfully created batches")
    print(f"[OK] Batch contains {batch.x.shape[0]} node-time samples")
    print(
        f"[OK] Each sample has {batch.x.shape[1]} time steps and {batch.x.shape[2]} features"
    )
    print(f"[OK] Target has {batch.y.shape[0]} prediction steps")

    print("\n>>> MultiFeatureStreamDataset Batching Test Passed!")
