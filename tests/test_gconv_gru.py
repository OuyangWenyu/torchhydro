import torch

import pytest

from torchhydro.models.gconv_gru import GConvGRULayer, MultiFeatureStreamGCGRU


class TestGConvGRULayer:

    @pytest.fixture
    def setup_layer(self) -> GConvGRULayer:
        in_channels = 10
        out_channels = 32
        K = 3
        return GConvGRULayer(in_channels, out_channels, K)

    def test_initialization(self, setup_layer: GConvGRULayer) -> None:
        layer = setup_layer
        assert layer.out_channels == 32

    def test_forward(self, setup_layer: GConvGRULayer) -> None:
        layer = setup_layer
        batch_size = 4
        seq_len = 6
        num_nodes = 5
        in_channels = 10
        x_seq = torch.randn(batch_size, seq_len, num_nodes, in_channels)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
        outputs, h_last = layer(x_seq, edge_index)
        assert outputs.shape == (seq_len, batch_size, num_nodes, 32)
        assert h_last.shape == (batch_size, num_nodes, 32)


class TestMultiFeatureStreamGCGRU:

    @pytest.fixture
    def setup_model(self) -> MultiFeatureStreamGCGRU:
        num_nodes = 5
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
        config = {
            "node_attr_dim": 2,
            "use_node_attr": True,
            "gcgru_out_channels": 16,
            "gcgru_K": 2,
            "mlp_hidden_dim": 32,
            "mlp_dropout": 0.1,
            "forecast_length": 3,
        }
        node_attributes = torch.randn(num_nodes, 2)
        return MultiFeatureStreamGCGRU(edge_index, config, node_attributes)

    def test_initialization(self, setup_model: MultiFeatureStreamGCGRU) -> None:
        model = setup_model
        assert model.time_series_dim == 3
        assert model.static_attr_dim == 2
        assert model.input_feature_dim == 5

    def test_forward_pass(self, setup_model: MultiFeatureStreamGCGRU) -> None:
        model = setup_model
        batch_size = 4
        hindcast_length = 10
        num_nodes = 5
        # Input features: precipitation, lstm_flow, measured_flow (3) + static attrs (2) = 5
        x = torch.randn(batch_size, hindcast_length, num_nodes, 5)
        predictions = model(x)
        # Forecast length is 3, so output should be [batch_size, 3, num_nodes]
        assert predictions.shape == (batch_size, 3, num_nodes)

    def test_forward_pass_single_step(
        self, setup_model: MultiFeatureStreamGCGRU
    ) -> None:
        model = setup_model
        model.config["forecast_length"] = 1
        # Re-init output layer to match new forecast length
        model.output_mlp[-1] = torch.nn.Linear(model.config["mlp_hidden_dim"], 1)
        batch_size = 4
        hindcast_length = 10
        num_nodes = 5
        x = torch.randn(batch_size, hindcast_length, num_nodes, 5)
        predictions = model(x)
        # Forecast length is 1, so output should be [batch_size, num_nodes] (squeezed)
        assert predictions.shape == (batch_size, num_nodes)

    def test_autoregressive_predict(self, setup_model: MultiFeatureStreamGCGRU) -> None:
        model = setup_model
        batch_size = 4
        hindcast_length = 10
        num_nodes = 5
        forecast_steps = 3
        # Initial input
        x = torch.randn(batch_size, hindcast_length, num_nodes, 5)
        # Future inputs
        precipitation_future = torch.randn(batch_size, forecast_steps, num_nodes)
        lstm_future = torch.randn(batch_size, forecast_steps, num_nodes)
        predictions = model.autoregressive_predict(
            x,
            forecast_steps,
            precipitation_future=precipitation_future,
            lstm_future=lstm_future,
        )
        assert predictions.shape == (batch_size, forecast_steps, num_nodes)
