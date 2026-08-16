"""
DPL-MC model with dynamic reservoir coupling.

This module hosts the model-side components of the unified DPL-MC pipeline
(previously in ``scripts/train_dplmc_rsvr.py``):

* :class:`MLP` -- DPL router network predicting Manning ``n`` / flow coefficient ``q``.
* :class:`CoupledExplicitMC` -- differentiable Muskingum-Cunge router whose routing
  loop optionally interleaves a reservoir stepper for dynamic reservoir coupling.
* :class:`DPLMC_Wrapper` -- wraps the router + neural network into an ``nn.Module``.
* :class:`ReservoirLSTM` -- seq2seq residual reservoir LSTM (stepwise decodable).
* :class:`ReservoirCoupler` -- runs the reservoir LSTM in lock-step with the router.

The differentiable Muskingum-Cunge router (``dMC.physics.ExplicitMC``) lives in the
copied ``torchhydro/models/dmc`` package; we add it to ``sys.path`` so the original
``dMC.*`` imports keep working unchanged.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from tqdm import tqdm

# --- differentiable Muskingum-Cunge router (copied dMC package) ---
_DMC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmc")
if _DMC_DIR not in sys.path:
    sys.path.insert(0, _DMC_DIR)
from omegaconf import OmegaConf
from dMC.physics.explicit_mc import ExplicitMC  # noqa: E402


# ======================================================================================
# DPL router: MLP that predicts Manning n / flow coefficient q for the MC solver.
# ======================================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.n = None
        self.q = None

    def forward(self, inputs: torch.Tensor) -> None:
        out = self.layers(inputs)          # [num_nodes, 2]
        n_raw = out[:, 0]
        q_raw = out[:, 1]
        self.n = (0.01 + 0.14 * torch.sigmoid(n_raw)).to(torch.float64)
        self.q = (0.1 + 2.9 * torch.sigmoid(q_raw)).to(torch.float64)


class HydrofabricDict:
    pass


class NetworkDict:
    pass


class CoupledExplicitMC(ExplicitMC):
    """ExplicitMC whose routing loop optionally interleaves a reservoir stepper.

    The physics is identical to the parent's ``forward``; the only addition is that
    at reservoir nodes, during the forecast portion of a window, the propagated node
    flow is replaced by the reservoir module's outflow computed from the *routed*
    inflow at that node and timestep (i.e. dynamic, topological coupling).
    """

    def forward_coupled(self, inputs, coupler=None, warmup=0, window_start_abs=0):
        hydrofabric, observations = inputs
        attributes = hydrofabric.attributes
        source_flow = hydrofabric.source_flow
        node_interval_flow = hydrofabric.node_interval_flow
        device = source_flow.device
        gage_indices = hydrofabric.network.gage_indices
        A = hydrofabric.network.explicit_network_matrix.to(device)
        river_index_graph = torch.tensor(hydrofabric.network.index_graph, device=device)
        true_discharge = getattr(hydrofabric, 'true_discharge', None)
        if true_discharge is not None:
            true_discharge = true_discharge.to(device)

        T = source_flow.shape[0]
        N = source_flow.shape[1]
        output = torch.zeros([observations.shape[0], T], dtype=torch.float64, device=device)
        self.length = self.get_length(attributes).to(device)
        self.slope = self.get_slope(attributes).to(device)
        self._node_flow_t = torch.zeros(N, dtype=torch.float64, device=device)
        self._discharge_t = torch.zeros(N, dtype=torch.float64, device=device)

        if coupler is not None:
            coupler.reset(window_start_abs, warmup, device)

        timestep_iter = range(T)
        if getattr(self, 'show_progress', True):
            timestep_iter = tqdm(timestep_iter,
                                 desc=f"Epoch {self.epoch}: Coupled Muskingum Cunge Routing")
        for timestep in timestep_iter:
            self._node_flow_t1 = torch.zeros_like(self._node_flow_t)
            self._discharge_t1 = torch.zeros_like(self._discharge_t)
            self._simulated_q_t1 = torch.zeros_like(self._node_flow_t)

            lead = timestep - warmup
            use_reservoir = coupler is not None and lead >= 0
            if use_reservoir:
                coupler.begin_timestep(timestep)

            for j in range(river_index_graph.shape[0]):
                self._downstream_indexes = river_index_graph[j][river_index_graph[j] != -1]
                if j == 0:
                    node_flow_sim = source_flow[timestep, self._downstream_indexes].clone()
                else:
                    upstream_indexes = A[self._downstream_indexes]
                    mask = upstream_indexes[:, 0] != -1
                    node_flow_sim = node_interval_flow[timestep, self._downstream_indexes].clone()
                    if mask.any():
                        inflow = self._sum_upstream_current(self._discharge_t1, upstream_indexes[mask])
                        node_flow_sim[mask] = node_flow_sim[mask] + inflow
                self._simulated_q_t1[self._downstream_indexes] = node_flow_sim.clone()
                node_flow_prop = node_flow_sim.clone()

                # --- dynamic reservoir coupling (forecast portion only) ---
                res_local_mask = None
                if use_reservoir:
                    res_local_mask = coupler.is_reservoir[self._downstream_indexes]
                    if res_local_mask.any():
                        res_node_ids = self._downstream_indexes[res_local_mask]
                        inflow_phys = node_flow_sim[res_local_mask]
                        outflow_phys = coupler.step(res_node_ids, timestep, inflow_phys)
                        pos = torch.nonzero(res_local_mask, as_tuple=True)[0]
                        node_flow_prop = node_flow_prop.clone()
                        node_flow_prop[pos] = torch.clamp(outflow_phys, min=1e-4)

                # --- observed injection (gauge obs + warmup reservoir obs) ---
                if true_discharge is not None:
                    obs_t1 = true_discharge[timestep, self._downstream_indexes]
                    valid_mask = ~torch.isnan(obs_t1)
                    if res_local_mask is not None:
                        valid_mask = valid_mask & (~res_local_mask)
                    if valid_mask.any():
                        node_flow_prop = node_flow_prop.clone()
                        node_flow_prop[valid_mask] = torch.clamp(obs_t1[valid_mask], min=1e-4)

                self._node_flow_t1[self._downstream_indexes] = node_flow_prop
                prev_node_flow = self._node_flow_t[self._downstream_indexes]
                self._discharge_t1[self._downstream_indexes] = self.route_reach(prev_node_flow, node_flow_prop)

            for i in range(len(gage_indices)):
                output[i, timestep] = torch.sum(self._simulated_q_t1[gage_indices[i]])
            self._node_flow_t = self._node_flow_t1.clone()
            self._discharge_t = self._discharge_t1.clone()
        return output


class DPLMC_Wrapper(nn.Module):
    def __init__(self, num_nodes, num_features, mc_attributes, explicit_network_matrix,
                 index_graph, target_indices, source_mask, reservoir_coupler=None):
        super().__init__()
        self.neural_network = MLP(num_features)
        cfg = OmegaConf.create({
            "variables": {"p": 1.0, "t": 3600.0, "x": 0.2},
            "length": {"idx": 0},
            "slope": {"idx": 1, "min": 0.00001, "max": 0.1},
        })
        self.mc_router = CoupledExplicitMC(cfg, self.neural_network)

        self.hydrofabric = HydrofabricDict()
        self.hydrofabric.attributes = mc_attributes
        self.hydrofabric.network = NetworkDict()
        self.hydrofabric.network.gage_indices = [[i] for i in range(num_nodes)]
        self.hydrofabric.network.explicit_network_matrix = explicit_network_matrix
        self.hydrofabric.network.index_graph = index_graph
        self.hydrofabric.network.source_mask = source_mask.to(torch.bool)
        self.target_indices = target_indices

        self.reservoir_coupler = reservoir_coupler
        if reservoir_coupler is not None:
            # register so .to(device) / state_dict includes reservoir params
            self.reservoir_model = reservoir_coupler.model

    def forward(self, q_prime, static_features, gauge_obs=None, reservoir_discharge=None,
                warmup=0, window_start_abs=0, use_coupling=True):
        device = q_prime.device
        self.neural_network(static_features.to(torch.float32))
        self.mc_router.p_spatial = self.mc_router.p_spatial.to(device)
        self.mc_router.t = self.mc_router.t.to(device)
        self.mc_router.x_storage = self.mc_router.x_storage.to(device)

        q_prime64 = q_prime.to(torch.float64)
        source_mask = self.hydrofabric.network.source_mask.to(device)
        zeros = torch.zeros_like(q_prime64)
        self.hydrofabric.source_flow = torch.where(source_mask.unsqueeze(0), q_prime64, zeros)
        self.hydrofabric.node_interval_flow = torch.where(~source_mask.unsqueeze(0), q_prime64, zeros)

        true_discharge = None
        if gauge_obs is not None or reservoir_discharge is not None:
            true_discharge = torch.full_like(q_prime64, float('nan'), dtype=torch.float64)
            if gauge_obs is not None:
                gauge_obs = gauge_obs.to(torch.float64)
                gm = ~torch.isnan(gauge_obs)
                true_discharge[gm] = gauge_obs[gm]
            if reservoir_discharge is not None:
                reservoir_discharge = reservoir_discharge.to(torch.float64)
                rm = ~torch.isnan(reservoir_discharge)
                true_discharge[rm] = reservoir_discharge[rm]
        self.hydrofabric.true_discharge = true_discharge

        num_gages = len(self.hydrofabric.network.gage_indices)
        time_steps = q_prime.shape[0]
        observations = torch.zeros(num_gages, time_steps, dtype=torch.float64, device=device)

        coupler = self.reservoir_coupler if (use_coupling and self.reservoir_coupler is not None) else None
        out = self.mc_router.forward_coupled(
            (self.hydrofabric, observations),
            coupler=coupler,
            warmup=warmup,
            window_start_abs=window_start_abs,
        )
        return out.transpose(0, 1).unsqueeze(-1).to(torch.float32)


# ======================================================================================
# Reservoir LSTM (seq2seq residual) refactored from rsvr_test/ed_codex.py so that the
# decoder can be stepped one lead-hour at a time inside the routing loop.
# ======================================================================================
class ReservoirLSTM(nn.Module):
    def __init__(self, history_input_dim, future_input_dim, static_input_dim=0,
                 hidden_dim=64, num_layers=2, output_len=240, dropout=0.2,
                 dense_dim=64, discharge_history_index=0):
        super().__init__()
        self.history_input_dim = history_input_dim
        self.future_input_dim = future_input_dim
        self.static_input_dim = static_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len
        self.discharge_history_index = discharge_history_index
        self.static_context_dim = hidden_dim if static_input_dim > 0 else 0

        self.history_encoder = nn.LSTM(
            input_size=history_input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)

        if static_input_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_input_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        else:
            self.static_encoder = None

        decoder_input_dim = future_input_dim + 1 + self.static_context_dim
        self.decoder_cell = nn.LSTMCell(input_size=decoder_input_dim, hidden_size=hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        head_input_dim = hidden_dim + self.static_context_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, dense_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dense_dim, 1))

    def encode(self, history, static_features):
        """history: [n, T_hist, F_hist]; static_features: [n, F_static].

        Returns (h_dec, c_dec, static_context, base_y) for stepwise decoding.
        """
        _, (h_hist, c_hist) = self.history_encoder(history)
        h_dec = h_hist[-1]
        c_dec = c_hist[-1]
        static_context = None
        if self.static_input_dim > 0:
            static_context = self.static_encoder(static_features)
        last_y = history[:, -1, self.discharge_history_index:self.discharge_history_index + 1]
        base_y = torch.nan_to_num(last_y, nan=0.0, posinf=0.0, neginf=0.0)
        return h_dec, c_dec, static_context, base_y

    def decode_step(self, future_step, last_y, base_y, h_dec, c_dec, static_context):
        """One decoder lead-step.

        future_step: [n, F_future] (normalised future-known features)
        last_y / base_y: [n, 1] (normalised discharge, scaler_discharge space)
        Returns (pred_t, h_dec, c_dec) where pred_t is normalised discharge.
        """
        parts = [future_step, last_y]
        if static_context is not None:
            parts.append(static_context)
        decoder_input = torch.cat(parts, dim=-1)
        h_dec, c_dec = self.decoder_cell(decoder_input, (h_dec, c_dec))
        h_for_head = self.decoder_dropout(h_dec)
        if static_context is not None:
            head_input = torch.cat([h_for_head, static_context], dim=-1)
        else:
            head_input = h_for_head
        delta = self.head(head_input)
        pred_t = base_y + delta
        return pred_t, h_dec, c_dec


class ReservoirCoupler:
    """Runs the reservoir LSTM in lock-step with the MC routing loop.

    Responsibilities:
      * load reservoir_v3 dynamic features + static attrs for the reservoir nodes,
        aligned to the DPL routing time axis (with INPUT_LEN pre-history),
      * apply the ed_codex scalers,
      * encode reservoir history and step the decoder once per routing timestep,
      * bridge units (routing is m3/s, reservoir model is mm/h).
    """

    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.device = device
        self.input_len = cfg.RESERVOIR_INPUT_LEN
        self.output_len = cfg.RESERVOIR_OUTPUT_LEN
        self.feature_list = list(cfg.RESERVOIR_FEATURE_LIST)
        self.past_features = list(cfg.RESERVOIR_PAST_FEATURES)
        self.future_features = list(cfg.RESERVOIR_FUTURE_FEATURES)
        self.static_features = list(cfg.RESERVOIR_STATIC_FEATURES)

        f2i = {f: i for i, f in enumerate(self.feature_list)}
        self.history_idx = [f2i[f] for f in self.past_features]
        self.future_idx = [f2i[f] for f in self.future_features]
        self.discharge_history_index = self.past_features.index('discharge')
        self.streamflow_future_pos = self.future_features.index('streamflow')
        self.streamflow_feat_idx = f2i['streamflow']

        self._load_scalers()
        self._build_reservoir_nodes(dataset)
        self._load_reservoir_series(dataset)
        self._build_model()

        # per-reservoir decode state (set on begin_timestep / reset)
        self._h = None
        self._c = None
        self._static_ctx = None
        self._base_y = None
        self._last_y = None
        self._win_start = 0
        self._warmup = 0
        self._next_reencode = None

    # ---- setup helpers ----
    def _load_scalers(self):
        d = self.cfg.RESERVOIR_MODEL_DIR
        with open(os.path.join(d, 'scaler_discharge.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(d, 'target_scaler_discharge.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(os.path.join(d, 'static_scaler_discharge.pkl'), 'rb') as f:
            self.static_scaler = pickle.load(f)
        self.feat_mean = torch.tensor(self.scaler.mean_, dtype=torch.float32)
        self.feat_scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        self.tgt_mean = float(self.target_scaler.mean_[0])
        self.tgt_scale = float(self.target_scaler.scale_[0])

    def _build_reservoir_nodes(self, dataset):
        rv_attr = xr.open_dataset(os.path.join(self.cfg.RESERVOIR_DATA_DIR, 'attributes.nc'))
        rv_basins = set(str(b) for b in rv_attr.basin.values)
        rv_attr.close()
        basins = [str(b) for b in dataset.basins]
        res_flag = dataset.reservoir_station_mask
        self.node_indices = []          # global node index of each coupled reservoir
        self.res_basin_ids = []
        for i, b in enumerate(basins):
            if bool(res_flag[i]) and b in rv_basins:
                self.node_indices.append(i)
                self.res_basin_ids.append(b)
        self.n_res = len(self.node_indices)
        is_res = torch.zeros(len(basins), dtype=torch.bool)
        for i in self.node_indices:
            is_res[i] = True
        self.is_reservoir = is_res.to(self.device)
        # map global node index -> local reservoir index
        self.node_to_local = {n: k for k, n in enumerate(self.node_indices)}
        print(f"[ReservoirCoupler] coupling {self.n_res} reservoir node(s): {self.res_basin_ids}")

    def _load_reservoir_series(self, dataset):
        """Build normalised dynamic/static arrays for reservoir nodes on an
        extended time axis: index e maps to sim index (e - INPUT_LEN)."""
        sim_times = pd.to_datetime(dataset.ds_ts_sel.time.values)
        # pre-history: INPUT_LEN hourly steps immediately before sim start
        step = sim_times[1] - sim_times[0] if len(sim_times) > 1 else pd.Timedelta(hours=1)
        pre_times = pd.DatetimeIndex([sim_times[0] - step * k for k in range(self.input_len, 0, -1)])
        ext_times = pre_times.append(sim_times)
        self.T_sim = len(sim_times)
        self.E = len(ext_times)

        rv_ts = xr.open_dataset(os.path.join(self.cfg.RESERVOIR_DATA_DIR, 'timeseries_1h.nc'))
        rv_attr = xr.open_dataset(os.path.join(self.cfg.RESERVOIR_DATA_DIR, 'attributes.nc'))

        # areas (control area) for the unit bridge, from the DPL dataset
        area_da = dataset.ds_attr['area'] if 'area' in dataset.ds_attr else dataset.ds_int_attr['area']
        areas = np.array([float(area_da.sel(basin=b).values) for b in self.res_basin_ids],
                         dtype=np.float32)
        self.area = torch.tensor(areas, dtype=torch.float32, device=self.device)  # [n_res]

        n = self.n_res
        dyn = np.zeros((self.E, n, len(self.feature_list)), dtype=np.float32)
        rv_sel = rv_ts.sel(basin=self.res_basin_ids).reindex(time=ext_times)
        for fi, feat in enumerate(self.feature_list):
            if feat in rv_sel:
                arr = rv_sel[feat].transpose('time', 'basin').values.astype(np.float32)
            else:
                arr = np.zeros((self.E, n), dtype=np.float32)
            dyn[:, :, fi] = arr
        dyn = np.nan_to_num(dyn, nan=0.0)
        # normalise with scaler_discharge (per feature)
        mean = self.scaler.mean_.astype(np.float32)
        scale = self.scaler.scale_.astype(np.float32)
        dyn_norm = (dyn - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
        self.dyn_ext = torch.tensor(dyn_norm, dtype=torch.float32, device=self.device)  # [E,n,F]

        # static features
        stat = np.zeros((n, len(self.static_features)), dtype=np.float32)
        for si, feat in enumerate(self.static_features):
            if feat in rv_attr:
                stat[:, si] = np.array([float(rv_attr[feat].sel(basin=b).values)
                                        for b in self.res_basin_ids], dtype=np.float32)
        stat = np.nan_to_num(stat, nan=0.0)
        stat_norm = self.static_scaler.transform(stat).astype(np.float32)
        self.static_norm = torch.tensor(stat_norm, dtype=torch.float32, device=self.device)  # [n,S]

        rv_ts.close()
        rv_attr.close()

    def _build_model(self):
        self.model = ReservoirLSTM(
            history_input_dim=len(self.history_idx),
            future_input_dim=len(self.future_idx),
            static_input_dim=len(self.static_features),
            hidden_dim=self.cfg.RESERVOIR_HIDDEN_DIM,
            num_layers=self.cfg.RESERVOIR_NUM_LAYERS,
            output_len=self.cfg.RESERVOIR_OUTPUT_LEN,
            dropout=self.cfg.RESERVOIR_DROPOUT,
            dense_dim=self.cfg.RESERVOIR_DENSE_DIM,
            discharge_history_index=self.discharge_history_index,
        ).to(self.device)
        if self.cfg.RESERVOIR_LOAD_PRETRAINED:
            path = os.path.join(self.cfg.RESERVOIR_MODEL_DIR, self.cfg.RESERVOIR_MODEL_FILE)
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                print(f"[ReservoirCoupler] loaded pretrained reservoir weights from {path}")
            else:
                print(f"[ReservoirCoupler] WARNING: reservoir checkpoint not found at {path}")
        if not self.cfg.RESERVOIR_TRAINABLE:
            for p in self.model.parameters():
                p.requires_grad_(False)

    # ---- runtime hooks ----
    def reset(self, window_start_abs, warmup, device):
        self._win_start = int(window_start_abs)
        self._warmup = int(warmup)
        self._next_reencode = None
        self._h = self._c = self._static_ctx = self._base_y = self._last_y = None
        if self.cfg.RESERVOIR_TRAINABLE and self.model.training:
            pass
        else:
            self.model.eval()

    def _encode_at(self, decode_abs_sim):
        """(Re)encode all reservoirs with history ending at ``decode_abs_sim`` (sim idx)."""
        e0 = decode_abs_sim                     # ext index of first history step
        e1 = decode_abs_sim + self.input_len    # exclusive; last row = sim decode_abs_sim-1
        e0 = max(0, e0)
        hist = self.dyn_ext[e0:e1][:, :, self.history_idx]     # [T_hist, n, F_hist]
        hist = hist.permute(1, 0, 2).contiguous()              # [n, T_hist, F_hist]
        h, c, ctx, base_y = self.model.encode(hist, self.static_norm)
        self._h, self._c, self._static_ctx, self._base_y = h, c, ctx, base_y
        self._last_y = base_y

    def begin_timestep(self, timestep):
        lead = timestep - self._warmup
        if lead < 0:
            return
        if self._next_reencode is None or lead >= self._next_reencode:
            decode_abs = self._win_start + timestep      # sim index of this decode step
            self._encode_at(decode_abs)
            seg = 0 if self._next_reencode is None else (lead // self.output_len)
            self._next_reencode = (seg + 1) * self.output_len

    def step(self, res_node_ids, timestep, inflow_phys):
        """Advance decoder one step for the given reservoir nodes.

        res_node_ids: 1D tensor of global node indices (subset, this generation)
        inflow_phys : [k] routed inflow in m3/s
        Returns outflow in m3/s [k].
        """
        local = torch.tensor([self.node_to_local[int(n)] for n in res_node_ids.tolist()],
                             device=self.device, dtype=torch.long)
        decode_abs = self._win_start + timestep
        e = decode_abs + self.input_len          # ext index for future forcing at sim decode_abs
        e = min(e, self.E - 1)
        future_step = self.dyn_ext[e][:, self.future_idx][local].clone()   # [k, F_future]

        # bridge inflow m3/s -> mm/h -> normalised streamflow
        inflow_mmh = inflow_phys.to(torch.float32) * (3.6 / self.area[local])
        inflow_norm = (inflow_mmh - self.feat_mean[self.streamflow_feat_idx].to(self.device)) \
            / self.feat_scale[self.streamflow_feat_idx].to(self.device)
        future_step[:, self.streamflow_future_pos] = inflow_norm

        last_y = self._last_y[local]
        base_y = self._base_y[local]
        h = self._h[local]
        c = self._c[local]
        ctx = self._static_ctx[local] if self._static_ctx is not None else None
        pred_t, h_new, c_new = self.model.decode_step(future_step, last_y, base_y, h, c, ctx)

        # scatter updated decode state back
        self._h = self._h.clone(); self._h[local] = h_new
        self._c = self._c.clone(); self._c[local] = c_new
        self._last_y = self._last_y.clone(); self._last_y[local] = pred_t

        # normalised discharge -> mm/h -> m3/s
        outflow_mmh = pred_t.squeeze(-1) * self.tgt_scale + self.tgt_mean
        outflow_phys = outflow_mmh * (self.area[local] / 3.6)
        return outflow_phys.to(torch.float64)
