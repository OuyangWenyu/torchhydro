import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class ExplicitMC(nn.Module):
    def __init__(self, cfg: DictConfig, neural_network: nn.Module):
        super(ExplicitMC, self).__init__()
        self.cfg = cfg
        self.epoch = 0
        self.neural_network = neural_network
        self.p_spatial = torch.tensor(self.cfg.variables.p)
        self.t = torch.tensor(self.cfg.variables.t)
        self.x_storage = torch.tensor(self.cfg.variables.x)
        self.length = None
        self.slope = None
        self.velocity = None
        self.show_progress = True
        self._discharge_t = None
        self._discharge_t1 = None
        self._downstream_indexes = None
        self._q_t = None

    @property
    def n(self) -> torch.Tensor:
        return self.neural_network.n

    @property
    def q_spatial(self) -> torch.Tensor:
        return self.neural_network.q

    def get_length(self, attributes: torch.Tensor) -> torch.Tensor:
        length_idx = self.cfg.length.idx
        return attributes[:, length_idx].clone()

    def get_slope(self, attributes: torch.Tensor) -> torch.Tensor:
        slope_idx = self.cfg.slope.idx
        s0_ = attributes[:, slope_idx].clone()
        return torch.clamp(s0_, min=self.cfg.slope.min, max=self.cfg.slope.max)

    @staticmethod
    @torch.jit.script
    def _get_velocity(q_t, _n, _q_spatial, _s0, p_spatial) -> torch.Tensor:
        """
        Since this is an explicit solver, we need to index the values that we're calculating
        :param downstream_indexes: Reach indexes
        """
        q_t = torch.nan_to_num(q_t, nan=1e-4, posinf=1e6, neginf=1e-4)
        _n = torch.nan_to_num(_n, nan=0.05, posinf=0.15, neginf=0.01)
        _q_spatial = torch.nan_to_num(_q_spatial, nan=1.0, posinf=3.0, neginf=0.1)
        _s0 = torch.nan_to_num(_s0, nan=1e-3, posinf=0.1, neginf=1e-5)
        numerator = torch.clamp(q_t * _n * (_q_spatial + 1), min=1e-8)
        denominator = torch.clamp(p_spatial * torch.pow(_s0, 0.5), min=1e-8)
        exponent = torch.div(3.0, 5.0 + 3.0 * _q_spatial)
        depth = torch.pow(torch.div(numerator, denominator), exponent)
        depth = torch.nan_to_num(depth, nan=1e-4, posinf=1e4, neginf=1e-4)
        v = torch.div(1, _n) * torch.pow(depth, (2 / 3)) * torch.pow(_s0, (1 / 2))
        c_ = torch.clamp(torch.nan_to_num(v, nan=0.3, posinf=15.0, neginf=0.3), 0.3, 15)
        c = c_ * 5 / 3
        return c

    def route_reach(
        self,
        node_flow_t: torch.Tensor,
        node_flow_t1: torch.Tensor,
    ) -> torch.Tensor:
        q_t = torch.nan_to_num(self._discharge_t[self._downstream_indexes], nan=1e-4, posinf=1e6, neginf=1e-4)
        length = torch.nan_to_num(self.length[self._downstream_indexes], nan=100.0, posinf=1e6, neginf=100.0)
        if self.n.dim() == 0:
            _n = self.n
        else:
            _n = self.n[self._downstream_indexes]
        if self.q_spatial.dim() == 0:
            _q_spatial = self.q_spatial
        else:
            _q_spatial = self.q_spatial[self._downstream_indexes]
        _s0 = torch.nan_to_num(self.slope[self._downstream_indexes], nan=1e-3, posinf=0.1, neginf=1e-5)
        velocity = self._get_velocity(q_t, _n, _q_spatial, _s0, self.p_spatial)
        lateral = torch.zeros_like(node_flow_t1)
        return self._muskingum_cunge(
            torch.nan_to_num(node_flow_t, nan=1e-4, posinf=1e6, neginf=1e-4),
            torch.nan_to_num(node_flow_t1, nan=1e-4, posinf=1e6, neginf=1e-4),
            q_t,
            lateral,
            length,
            velocity,
            self.x_storage,
            self.t,
        )

    @staticmethod
    @torch.jit.script
    def _muskingum_cunge(
        i_t: torch.Tensor,
        i_t1: torch.Tensor,
        q_t: torch.Tensor,
        q_prime_segment: torch.Tensor,
        length: torch.Tensor,
        velocity: torch.Tensor,
        x_storage: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        An explicit Muskingum Cunge Solution
        :param downstream_indexes: Reach indexes
        :param i: timestep
        :param i_t: inflow at time t
        :param i_t1: discharge at time t
        :param q_prime: lateral inflow at reaches
        :return:
        """
        velocity = torch.clamp(torch.nan_to_num(velocity, nan=0.3, posinf=15.0, neginf=0.3), min=0.3)
        length = torch.nan_to_num(length, nan=100.0, posinf=1e6, neginf=100.0)
        k = length / velocity
        denom = torch.clamp((2.0 * k * (1.0 - x_storage)) + t, min=1e-8)
        c_1 = (t - (2.0 * k * x_storage)) / denom
        c_2 = (t + (2.0 * k * x_storage)) / denom
        c_3 = (2.0 * k * (1.0 - x_storage) - t) / denom
        c_4 = (2.0 * t) / denom
        q_t1 = (c_1 * i_t1) + (c_2 * i_t) + (c_3 * q_t) + (c_4 * q_prime_segment)
        return torch.clamp(torch.nan_to_num(q_t1, nan=1e-4, posinf=1e6, neginf=1e-4), min=1e-4)

    def read_inflow(self, upstream_indexes):
        return self._read_inflow(
            self._discharge_t, self._discharge_t1, upstream_indexes
        )

    @staticmethod
    @torch.jit.script
    def _sum_upstream_current(discharge_t1, upstream_indexes):
        device = discharge_t1.device
        inflow = torch.zeros(len(upstream_indexes), dtype=torch.float64, device=device)
        for i, row in enumerate(upstream_indexes):
            valid_indices = row[row != -1]
            if valid_indices.numel() > 0:
                inflow[i] = discharge_t1[valid_indices].sum()
        return inflow

    @staticmethod
    @torch.jit.script
    def _read_inflow(discharge_t, discharge_t1, upstream_indexes):
        device = discharge_t.device
        single_index_mask = (upstream_indexes != -1).sum(dim=1) == 1
        masked_indexes = upstream_indexes[single_index_mask]
        single_indices = masked_indexes[masked_indexes != -1]

        many_index_mask = (upstream_indexes != -1).sum(dim=1) > 1
        i_t = torch.zeros(len(upstream_indexes), dtype=torch.float64, device=device)
        i_t1 = torch.zeros(len(upstream_indexes), dtype=torch.float64, device=device)
        i_t[single_index_mask] = discharge_t[single_indices]
        i_t1[single_index_mask] = discharge_t1[single_indices]
        if many_index_mask.sum() > 0:
            masked_indexes = upstream_indexes[many_index_mask]
            actual_indices = torch.arange(len(many_index_mask), device=device)[many_index_mask]
            for i, row in enumerate(masked_indexes):
                valid_indices = row[row != -1]
                idx = actual_indices[i]
                i_t[idx] = discharge_t[valid_indices].sum()
                i_t1[idx] = discharge_t1[valid_indices].sum()
        return i_t, i_t1

    def forward(self, inputs):
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

        output = torch.zeros([observations.shape[0], source_flow.shape[0]], dtype=torch.float64, device=device)
        self.length = self.get_length(attributes).to(device)
        self.slope = self.get_slope(attributes).to(device)
        self._node_flow_t = torch.zeros(source_flow.shape[1], dtype=torch.float64, device=device)
        self._discharge_t = torch.zeros(source_flow.shape[1], dtype=torch.float64, device=device)

        timestep_iter = range(len(source_flow))
        if self.show_progress:
            timestep_iter = tqdm(timestep_iter, desc=f"Epoch {self.epoch}: Explicit Muskingum Cunge Routing")
        for timestep in timestep_iter:
            self._node_flow_t1 = torch.zeros_like(self._node_flow_t)
            self._discharge_t1 = torch.zeros_like(self._discharge_t)
            self._simulated_q_t1 = torch.zeros_like(self._node_flow_t)
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
                if true_discharge is not None:
                    obs_t1 = true_discharge[timestep, self._downstream_indexes]
                    valid_mask = ~torch.isnan(obs_t1)
                    if valid_mask.any():
                        node_flow_prop[valid_mask] = torch.clamp(obs_t1[valid_mask], min=1e-4)
                self._node_flow_t1[self._downstream_indexes] = node_flow_prop
                prev_node_flow = self._node_flow_t[self._downstream_indexes]
                self._discharge_t1[self._downstream_indexes] = self.route_reach(prev_node_flow, node_flow_prop)
            for i in range(len(gage_indices)):
                output[i, timestep] = torch.sum(self._simulated_q_t1[gage_indices[i]])
            self._node_flow_t = self._node_flow_t1.clone()
            self._discharge_t = self._discharge_t1.clone()
        return output
