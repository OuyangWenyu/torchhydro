"""
Training / inference driver for the DPL-MC model with dynamic reservoir coupling.

This module hosts the data + orchestration side of the unified DPL-MC pipeline
(previously in ``scripts/train_dplmc_rsvr.py``):

* :class:`HydroGraphDataset` -- basin-agnostic reader for the pre-packaged NetCDF
  data (control / intermediate attributes & timeseries + LSTM predictions) and
  builds the river-network topology used by the differentiable MC router.
* helper functions to build datasets / models and select loss targets.
* :func:`train` / :func:`run_inference` -- the full train / rolling-forecast loops.

The scripts only keep the configuration (``Config`` + ``BASIN_CONFIGS``) and call
:func:`train` / :func:`run_inference` here, mirroring the other torchhydro scripts.
"""

import os
import json
import random
import pickle
import ast

import numpy as np
import xarray as xr
import torch
import networkx as nx
from tqdm import tqdm

from torchhydro.models.dplmc_rsvr import DPLMC_Wrapper, ReservoirCoupler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================================
# Dataset (basin-agnostic; boundary/no-obs/reservoir behaviour driven by explicit args)
# ======================================================================================
class HydroGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, output_dir, mode, train_period, test_period,
                 convert_units=True, static_feature_list=None, warmup=72,
                 use_boundary=False, boundary_file=None, boundary_stations=None,
                 no_obs_stations=None, use_discharge=True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mode = mode
        self.convert_units = convert_units
        self.static_feature_list = static_feature_list or []
        self.warmup = warmup
        self.use_boundary = use_boundary
        self.boundary_file = boundary_file
        self.boundary_stations = boundary_stations or []
        self.no_obs_stations = no_obs_stations or []
        self.use_discharge = use_discharge
        self.boundary_mask = None
        self.boundary_ds = None
        self.no_obs_mask = None

        self._load_data(train_period, test_period)
        self._build_graph()

    def _load_data(self, train_period, test_period):
        attr_path = os.path.join(self.data_dir, 'attributes.nc')
        int_attr_path = os.path.join(self.data_dir, 'intermediate_attributes.nc')
        ts_path = os.path.join(self.data_dir, 'timeseries_1h.nc')
        int_ts_path = os.path.join(self.data_dir, 'intermediate_timeseries_1h.nc')
        lstm_path = os.path.join(self.data_dir, 'lstmpred.nc')

        self.ds_attr = xr.open_dataset(attr_path)
        self.ds_int_attr = xr.open_dataset(int_attr_path)
        self.ds_ts = xr.open_dataset(ts_path)
        self.ds_int_ts = xr.open_dataset(int_ts_path)
        self.ds_lstm = xr.open_dataset(lstm_path)

        if self.convert_units:
            print("Converting units from mm/h to m3/s...")
            if 'area' in self.ds_attr:
                factor_control = self.ds_attr['area'] / 3.6
                if 'streamflow' in self.ds_ts:
                    self.ds_ts['streamflow'] = self.ds_ts['streamflow'] * factor_control
                if 'discharge' in self.ds_ts:
                    self.ds_ts['discharge'] = self.ds_ts['discharge'] * factor_control
            else:
                print("Warning: 'area' not found in attributes.nc, skipping streamflow conversion.")
            if 'area' in self.ds_int_attr:
                factor_interval = self.ds_int_attr['area'] / 3.6
                if 'streamflow' in self.ds_lstm:
                    self.ds_lstm['streamflow'] = self.ds_lstm['streamflow'] * factor_interval
            else:
                print("Warning: 'area' not found in intermediate_attributes.nc, skipping LSTM conversion.")

        self.basins = self.ds_int_attr.basin.values
        if not np.array_equal(self.ds_ts.basin.values, self.basins):
            print("Warning: Control timeseries basin order differs. Reindexing...")
            self.ds_ts = self.ds_ts.reindex(basin=self.basins)

        lstm_basins = self.ds_lstm.basin.values
        if not np.array_equal(lstm_basins, self.basins):
            if set(lstm_basins) != set(self.basins):
                raise ValueError("LSTM prediction file has different basins than attributes file.")
            self.ds_lstm = self.ds_lstm.reindex(basin=self.basins)

        start_date, end_date = (train_period if self.mode == 'train' else test_period)
        print(f"Dataset mode: {self.mode} | Time Range: {start_date} to {end_date}")

        self.ds_ts_sel = self.ds_ts.sel(time=slice(start_date, end_date))
        self.ds_lstm_sel = self.ds_lstm.sel(time=slice(start_date, end_date))
        self.ds_int_ts_sel = self.ds_int_ts.sel(time=slice(start_date, end_date))
        if 'basin' in self.ds_int_ts_sel.dims:
            self.ds_int_ts_sel = self.ds_int_ts_sel.reindex(basin=self.basins)

        common_times = np.intersect1d(self.ds_ts_sel.time.values, self.ds_lstm_sel.time.values)
        if 'time' in self.ds_int_ts_sel.dims:
            common_times = np.intersect1d(common_times, self.ds_int_ts_sel.time.values)
        if len(common_times) < 24:
            print(f"CRITICAL WARNING: Common time range for {self.mode} too small: {len(common_times)}.")
        self.ds_ts_sel = self.ds_ts_sel.sel(time=common_times)
        self.ds_lstm_sel = self.ds_lstm_sel.sel(time=common_times)
        self.ds_int_ts_sel = self.ds_int_ts_sel.sel(time=common_times)
        print(f"Dataset length (timesteps): {len(self.ds_ts_sel.time)}")

        self.target = self.ds_ts_sel['streamflow'].transpose('time', 'basin').values[..., np.newaxis]
        self.gauge_obs = self.target[:, :, 0].astype(np.float32)

        basin_to_idx = {str(b): i for i, b in enumerate(self.basins)}
        self.no_obs_mask = np.zeros(len(self.basins), dtype=np.float32)
        for st in self.no_obs_stations:
            idx = basin_to_idx.get(str(st))
            if idx is None:
                print(f"Warning: NO_OBS station {st} not found.")
                continue
            self.no_obs_mask[idx] = 1.0
        if self.no_obs_mask.any():
            self.gauge_obs[:, self.no_obs_mask == 1.0] = np.nan
            print(f"Detected {int(self.no_obs_mask.sum())} no-observation stations.")

        self.reservoir_station_mask = np.zeros(len(self.basins), dtype=bool)
        if self.use_discharge:
            if 'discharge' in self.ds_ts_sel:
                data = self.ds_ts_sel['discharge'].transpose('time', 'basin').values
            elif 'discharge' in self.ds_int_ts_sel:
                data = self.ds_int_ts_sel['discharge'].transpose('time', 'basin').values
            else:
                data = np.full((len(self.ds_ts_sel.time), len(self.basins)), np.nan, dtype=np.float32)
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
            full_masks = []
            if 'discharge' in self.ds_ts:
                full_masks.append((~np.isnan(self.ds_ts['discharge'].transpose('time', 'basin').values)).any(axis=0))
            if 'discharge' in self.ds_int_ts:
                full_masks.append((~np.isnan(self.ds_int_ts['discharge'].transpose('time', 'basin').values)).any(axis=0))
            if full_masks:
                self.reservoir_station_mask = np.logical_or.reduce(full_masks)
            self.reservoir_discharge = data.astype(np.float32)
            print(f"Detected {int(self.reservoir_station_mask.sum())} reservoir nodes (full discharge records).")
        else:
            self.reservoir_discharge = None

        self.boundary_mask = np.zeros(len(self.basins), dtype=np.float32)
        if self.use_boundary:
            for st in self.boundary_stations:
                idx = basin_to_idx.get(str(st))
                if idx is None:
                    print(f"Warning: Boundary station {st} not found.")
                    continue
                if bool(self.reservoir_station_mask[idx]):
                    print(f"Station {st} is both boundary and reservoir. Prioritizing reservoir.")
                    continue
                self.boundary_mask[idx] = 1.0
            print(f"Detected {int(self.boundary_mask.sum())} valid boundary nodes.")
            if self.boundary_file and os.path.exists(self.boundary_file):
                self.boundary_ds = xr.open_dataset(self.boundary_file)
            else:
                print(f"Warning: Boundary file {self.boundary_file} not found!")

        self.q_prime = self.ds_lstm_sel['streamflow'].transpose('time', 'basin').values

        static_arrays = []
        for feat in self.static_feature_list:
            source, var_name = (feat.split(':', 1) if ':' in feat else (None, feat))
            if source == 'interval':
                data_static = self.ds_int_attr[var_name].values
            elif source == 'control':
                data_static = self.ds_attr[var_name].values
            elif var_name in self.ds_int_attr:
                data_static = self.ds_int_attr[var_name].values
            elif var_name in self.ds_attr:
                data_static = self.ds_attr[var_name].values
            else:
                raise ValueError(f"Static feature {var_name} not found in attributes.")
            data_static = np.nan_to_num(data_static, nan=np.nanmean(data_static))
            static_arrays.append(data_static)

        if static_arrays:
            self.static_features = np.stack(static_arrays, axis=1)
            from sklearn.preprocessing import StandardScaler
            scaler_path = os.path.join(self.output_dir, 'static_scaler.pkl')
            if self.mode == 'train':
                self.static_scaler = StandardScaler()
                self.static_features = self.static_scaler.fit_transform(self.static_features)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.static_scaler, f)
            else:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.static_scaler = pickle.load(f)
                    self.static_features = self.static_scaler.transform(self.static_features)
                else:
                    print("Warning: static_scaler.pkl not found, skipping normalization.")
        else:
            self.static_features = np.zeros((len(self.basins), 1))

    def _build_graph(self):
        num_nodes = len(self.basins)
        self.adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.in_degree = np.zeros(num_nodes)
        if 'downstream_id' not in self.ds_int_attr:
            print("Warning: 'downstream_id' not found. Assuming all sources.")
            self.node_mask = np.ones(num_nodes, dtype=np.float32)
            return
        downstream_ids = self.ds_int_attr['downstream_id'].values
        river_lengths = self.ds_int_attr['river_length'].values if 'river_length' in self.ds_int_attr else None
        river_slopes = self.ds_int_attr['river_slope'].values if 'river_slope' in self.ds_int_attr else None

        id_map = None
        if 'category_mapping' in self.ds_int_attr['downstream_id'].attrs:
            try:
                id_map = ast.literal_eval(self.ds_int_attr['downstream_id'].attrs['category_mapping'])
            except Exception as e:
                print(f"Warning: Failed to parse category_mapping: {e}.")

        basin_to_idx = {str(b): i for i, b in enumerate(self.basins)}
        for i, basin in enumerate(self.basins):
            raw_d_id = downstream_ids[i]
            if hasattr(raw_d_id, 'item'):
                raw_d_id = raw_d_id.item()
            target_basin_name = str(id_map.get(raw_d_id, raw_d_id)) if id_map is not None else str(raw_d_id)
            if target_basin_name in ['0', '', 'nan', 'None']:
                continue
            if target_basin_name in basin_to_idx:
                j = basin_to_idx[target_basin_name]
                if i != j:
                    self.adj[i, j] = 1
                    self.in_degree[j] += 1

        self.node_mask = (self.in_degree == 0).astype(np.float32)

        G_sparse = nx.DiGraph(self.adj)
        generations = list(nx.topological_generations(G_sparse))
        max_len = max((len(g) for g in generations), default=1)
        index_graph = np.full((len(generations), max_len), -1, dtype=np.int64)
        for i, g in enumerate(generations):
            index_graph[i, :len(g)] = g
        self.index_graph = index_graph

        max_upstream = max((G_sparse.in_degree(n) for n in G_sparse.nodes()), default=1) or 1
        A = np.full((num_nodes, max_upstream), -1, dtype=np.int64)
        for i in range(num_nodes):
            preds = list(G_sparse.predecessors(i))
            A[i, :len(preds)] = preds
        self.explicit_network_matrix = torch.from_numpy(A)

        length_val = np.nan_to_num(river_lengths if river_lengths is not None else np.ones(num_nodes) * 1000.0, nan=100.0)
        slope_val = np.nan_to_num(river_slopes if river_slopes is not None else np.ones(num_nodes) * 0.001, nan=0.001)
        self.mc_attributes = torch.tensor(np.stack([length_val, slope_val], axis=1), dtype=torch.float64)


# ======================================================================================
# Helpers
# ======================================================================================
def build_conditioned_gauge_obs(raw_gauge_obs, boundary_mask, no_obs_mask, warmup):
    conditioned = raw_gauge_obs.clone()
    if no_obs_mask is not None:
        conditioned[:, no_obs_mask.to(torch.bool)] = float('nan')
    keep_future = boundary_mask.to(torch.bool) if boundary_mask is not None \
        else torch.zeros(conditioned.shape[1], dtype=torch.bool, device=conditioned.device)
    if no_obs_mask is not None:
        keep_future = keep_future & (~no_obs_mask.to(torch.bool))
    conditioned[warmup:, ~keep_future] = float('nan')
    return conditioned


def load_boundary_forecast_window(dataset, ref_time, horizon, device, convert_units):
    if (not getattr(dataset, 'use_boundary', False)) or dataset.boundary_ds is None:
        return None
    out = torch.full((horizon, len(dataset.basins)), float('nan'), dtype=torch.float32, device=device)
    for idx, is_boundary in enumerate(dataset.boundary_mask):
        if is_boundary != 1.0:
            continue
        st = str(dataset.basins[idx])
        try:
            vals = dataset.boundary_ds['prediction'].sel(time=ref_time, basin=st).values[:horizon]
            if convert_units:
                if 'area' in dataset.ds_attr:
                    vals = vals * (float(dataset.ds_attr['area'].sel(basin=st).values) / 3.6)
                elif 'area' in dataset.ds_int_attr:
                    vals = vals * (float(dataset.ds_int_attr['area'].sel(basin=st).values) / 3.6)
            out[:len(vals), idx] = torch.as_tensor(vals, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"Warning: failed to load boundary forecast for {st} at {ref_time}: {e}")
    return out


def select_target_indices(dataset, cfg):
    num_nodes = len(dataset.basins)
    basins_list = [str(b) for b in dataset.basins]
    if cfg.LOSS_MODE == 'confluence':
        target_indices = np.where(dataset.node_mask == 0)[0].tolist()
        print(f"LOSS_MODE: confluence. Selected {len(target_indices)} confluence nodes.")
    elif cfg.LOSS_MODE == 'specific':
        if not cfg.TARGET_STATIONS:
            raise ValueError("LOSS_MODE is 'specific' but TARGET_STATIONS is empty.")
        target_indices = [basins_list.index(str(st)) for st in cfg.TARGET_STATIONS
                          if str(st) in basins_list]
        print(f"LOSS_MODE: specific. Selected {len(target_indices)} nodes.")
    elif cfg.LOSS_MODE == 'all':
        target_indices = list(range(num_nodes))
        print(f"LOSS_MODE: all. Selected all {num_nodes} nodes.")
    else:
        raise ValueError(f"Unknown LOSS_MODE: {cfg.LOSS_MODE}")

    no_obs_indices = set(np.where(dataset.no_obs_mask == 1.0)[0].tolist())
    if no_obs_indices:
        target_indices = [i for i in target_indices if i not in no_obs_indices]
        print(f"Excluded {len(no_obs_indices)} no-observation stations from targets.")
    if not target_indices:
        raise ValueError("No valid target stations remain.")
    return target_indices


def build_dataset(cfg, mode):
    return HydroGraphDataset(
        data_dir=cfg.DATA_DIR, output_dir=cfg.OUTPUT_DIR, mode=mode,
        train_period=cfg.TRAIN_PERIOD, test_period=cfg.TEST_PERIOD,
        convert_units=cfg.CONVERT_UNITS, static_feature_list=cfg.STATIC_FEATURE_LIST,
        warmup=cfg.WARMUP, use_boundary=cfg.USE_BOUNDARY, boundary_file=cfg.BOUNDARY_FILE,
        boundary_stations=cfg.BOUNDARY_STATIONS, no_obs_stations=cfg.NO_OBS_STATIONS,
        use_discharge=cfg.USE_DISCHARGE)


def build_model(cfg, dataset, device, target_indices):
    coupler = None
    if cfg.USE_RESERVOIR_COUPLING:
        coupler = ReservoirCoupler(cfg, dataset, device)
        if coupler.n_res == 0:
            print("[ReservoirCoupler] no reservoir nodes matched reservoir_v3; disabling coupling.")
            coupler = None
    num_features = dataset.static_features.shape[1]
    model = DPLMC_Wrapper(
        num_nodes=len(dataset.basins), num_features=num_features,
        mc_attributes=dataset.mc_attributes.to(device),
        explicit_network_matrix=dataset.explicit_network_matrix.to(device),
        index_graph=dataset.index_graph, target_indices=target_indices,
        source_mask=torch.tensor(dataset.node_mask, dtype=torch.bool, device=device),
        reservoir_coupler=coupler,
    ).to(device)
    return model, coupler


# ======================================================================================
# Training
# ======================================================================================
def train(cfg):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() and 'cuda' in cfg.DEVICE else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = build_dataset(cfg, 'train')
    test_dataset = build_dataset(cfg, 'test')

    target_indices = select_target_indices(train_dataset, cfg)
    print(f"Target indices: {target_indices}")

    model, coupler = build_model(cfg, train_dataset, device, target_indices)
    # separate coupler for the test dataset time-axis (reservoir arrays differ per period)
    test_coupler = None
    if coupler is not None:
        test_coupler = ReservoirCoupler(cfg, test_dataset, device)
        test_coupler.model = model.reservoir_coupler.model  # share weights

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.LR)
    print(f"Trainable tensors: {len(trainable)} "
          f"(reservoir {'joint' if cfg.RESERVOIR_TRAINABLE else 'frozen'})")

    q_prime_train = torch.tensor(train_dataset.q_prime, dtype=torch.float32, device=device)
    target_train = torch.tensor(train_dataset.target, dtype=torch.float32, device=device)
    static_train = torch.tensor(train_dataset.static_features, dtype=torch.float32, device=device)
    gauge_train_raw = torch.tensor(train_dataset.gauge_obs, dtype=torch.float32, device=device)
    boundary_train = torch.tensor(train_dataset.boundary_mask, dtype=torch.float32, device=device)
    no_obs_train = torch.tensor(train_dataset.no_obs_mask, dtype=torch.float32, device=device)
    reservoir_train = (torch.tensor(train_dataset.reservoir_discharge, dtype=torch.float32, device=device)
                       if train_dataset.reservoir_discharge is not None else None)
    gauge_train = build_conditioned_gauge_obs(gauge_train_raw, boundary_train, no_obs_train, cfg.WARMUP)

    q_prime_test = torch.tensor(test_dataset.q_prime, dtype=torch.float32, device=device)
    target_test = torch.tensor(test_dataset.target, dtype=torch.float32, device=device)
    static_test = torch.tensor(test_dataset.static_features, dtype=torch.float32, device=device)
    gauge_test_raw = torch.tensor(test_dataset.gauge_obs, dtype=torch.float32, device=device)
    boundary_test = torch.tensor(test_dataset.boundary_mask, dtype=torch.float32, device=device)
    no_obs_test = torch.tensor(test_dataset.no_obs_mask, dtype=torch.float32, device=device)
    reservoir_test = (torch.tensor(test_dataset.reservoir_discharge, dtype=torch.float32, device=device)
                      if test_dataset.reservoir_discharge is not None else None)
    gauge_test = build_conditioned_gauge_obs(gauge_test_raw, boundary_test, no_obs_test, cfg.WARMUP)

    def calc_loss(preds, targets, warmup):
        preds = preds[warmup:]
        targets = targets[warmup:]
        loss = 0.0
        for idx in target_indices:
            loss += torch.sqrt(torch.mean((preds[:, idx, 0] - targets[:, idx, 0]) ** 2) + 1e-8)
        return loss / len(target_indices)

    print("Starting training...")
    best_loss = float('inf')
    metrics_history = []
    metrics_path = os.path.join(cfg.OUTPUT_DIR, cfg.METRICS_FILE)

    for epoch in range(cfg.EPOCHS):
        model.train()
        model.reservoir_coupler = coupler
        optimizer.zero_grad()
        preds_train = model(q_prime_train, static_train, gauge_obs=gauge_train,
                            reservoir_discharge=reservoir_train, warmup=cfg.WARMUP,
                            window_start_abs=0, use_coupling=(coupler is not None))
        train_loss = calc_loss(preds_train, target_train, cfg.WARMUP)
        train_loss.backward()
        optimizer.step()

        model.eval()
        model.reservoir_coupler = test_coupler if test_coupler is not None else coupler
        with torch.no_grad():
            preds_test = model(q_prime_test, static_test, gauge_obs=gauge_test.clone(),
                               reservoir_discharge=reservoir_test, warmup=cfg.WARMUP,
                               window_start_abs=0,
                               use_coupling=(test_coupler is not None or coupler is not None))
            val_loss = calc_loss(preds_test, target_test, cfg.WARMUP)
            val_nse, valid = 0.0, 0
            for idx in target_indices:
                p = preds_test[cfg.WARMUP:, idx, 0].detach().cpu().numpy()
                t = target_test[cfg.WARMUP:, idx, 0].cpu().numpy()
                den = np.sum((t - np.mean(t)) ** 2)
                if den != 0:
                    val_nse += 1 - np.sum((t - p) ** 2) / den
                    valid += 1
            val_nse = val_nse / valid if valid else float('nan')
        model.reservoir_coupler = coupler

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | Val NSE: {val_nse:.4f}")
        metrics_history.append({"epoch": epoch + 1, "train_loss": float(train_loss.item()),
                                "val_loss": float(val_loss.item()),
                                "val_nse": float(val_nse) if not np.isnan(val_nse) else None})
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)

        epoch_dir = os.path.join(cfg.OUTPUT_DIR, 'epoch_pth')
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, f'model_epoch_{epoch+1}.pth'))
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_PATH))
            print(f"Saved best model (Epoch {epoch+1})")
    print("Training complete.")


# ======================================================================================
# Inference (rolling forecast)
# ======================================================================================
def run_inference(cfg):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() and 'cuda' in cfg.DEVICE else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    print("Loading test dataset...")
    test_dataset = build_dataset(cfg, 'test')
    target_indices = select_target_indices(test_dataset, cfg)
    model, coupler = build_model(cfg, test_dataset, device, target_indices)

    model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_PATH)
    print(f"Loading DPL model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.mc_router.show_progress = False

    q_prime_test = torch.tensor(test_dataset.q_prime, dtype=torch.float32, device=device)
    target_test = test_dataset.target
    static_test = torch.tensor(test_dataset.static_features, dtype=torch.float32, device=device)
    gauge_test = torch.tensor(test_dataset.gauge_obs, dtype=torch.float32, device=device)
    no_obs_test = torch.tensor(test_dataset.no_obs_mask, dtype=torch.bool, device=device)
    gauge_test[:, no_obs_test] = float('nan')
    reservoir_test = (torch.tensor(test_dataset.reservoir_discharge, dtype=torch.float32, device=device)
                      if test_dataset.reservoir_discharge is not None else None)

    horizon = cfg.FORECAST_HORIZON
    stride = cfg.INFERENCE_STRIDE
    warmup = cfg.WARMUP
    T_total = q_prime_test.shape[0]
    times = test_dataset.ds_ts_sel.time.values
    basins = test_dataset.basins

    all_preds, all_obs, all_lstm, valid_times = [], [], [], []
    forecast_starts = list(range(warmup, T_total - horizon + 1, stride))
    print(f"Rolling inference: Horizon={horizon}h, Stride={stride}h, Warmup={warmup}h, "
          f"Windows={len(forecast_starts)}, Coupling={'on' if coupler else 'off'}")

    for t0 in tqdm(forecast_starts, desc="Rolling inference", unit="window"):
        win_q = q_prime_test[t0 - warmup: t0 + horizon]
        win_gauge = gauge_test[t0 - warmup: t0 + horizon].clone()
        win_gauge[warmup:] = float('nan')
        win_boundary = load_boundary_forecast_window(test_dataset, times[t0], horizon, device, cfg.CONVERT_UNITS)
        if win_boundary is not None:
            bcols = torch.tensor(test_dataset.boundary_mask, dtype=torch.bool, device=device)
            if bcols.any():
                win_gauge[warmup:, bcols] = win_boundary[:, bcols]
        win_reservoir = reservoir_test[t0 - warmup: t0 + horizon] if reservoir_test is not None else None

        with torch.inference_mode():
            win_preds = model(win_q, static_test, gauge_obs=win_gauge,
                              reservoir_discharge=win_reservoir, warmup=warmup,
                              window_start_abs=t0 - warmup, use_coupling=(coupler is not None))

        all_preds.append(win_preds[warmup: warmup + horizon, :, 0].cpu().numpy())
        all_obs.append(target_test[t0: t0 + horizon, :, 0])
        all_lstm.append(test_dataset.q_prime[t0: t0 + horizon])
        valid_times.append(times[t0])

    preds_arr = np.array(all_preds)
    obs_arr = np.array(all_obs)
    lstm_arr = np.array(all_lstm)
    lead_times = np.arange(1, horizon + 1)

    ds_out = xr.Dataset(
        data_vars={
            'prediction': (('time', 'lead_time', 'basin'), preds_arr),
            'observation': (('time', 'lead_time', 'basin'), obs_arr),
            'streamflow_lstm': (('time', 'lead_time', 'basin'), lstm_arr),
            'manning_n': ('basin', model.neural_network.n.detach().cpu().numpy()),
            'flow_coefficient_q': ('basin', model.neural_network.q.detach().cpu().numpy()),
        },
        coords={'time': valid_times, 'lead_time': lead_times, 'basin': basins},
    )
    ds_out.lead_time.attrs = {'long_name': 'Forecast Horizon', 'units': 'hours'}
    out_file = os.path.join(cfg.OUTPUT_DIR, cfg.RESULT_FILE)
    ds_out.to_netcdf(out_file)
    print(f"Results saved to {out_file}")
