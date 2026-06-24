import torch
import pytest

from torchhydro.models.gconv_gru import MultiFeatureStreamGCGRU, MultiFeatureStreamGCLSTM


@pytest.fixture
def edge_index():
    """Simple chain graph: 0->1->2->3->4."""
    return torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )


class TestMultiFeatureStreamGCGRU:
    NUM_NODES = 5
    SEQ_LEN = 13  # hindcast(10) + forecast(3)
    FORECAST_LEN = 3
    INPUT_SIZE = 5

    @pytest.fixture
    def model(self):
        return MultiFeatureStreamGCGRU(
            input_size=self.INPUT_SIZE,
            gcgru_out_channels=16,
            gcgru_K=2,
            mlp_hidden_dim=32,
            mlp_dropout=0.1,
            forecast_length=self.FORECAST_LEN,
        )

    def test_initialization(self, model):
        assert model.forecast_length == self.FORECAST_LEN
        assert model.gcgru_K == 2

    def test_forward_pass(self, model, edge_index):
        x = torch.randn(self.NUM_NODES, self.SEQ_LEN, self.INPUT_SIZE)
        with torch.no_grad():
            out = model(x, edge_index)
        # output: [total_nodes, forecast_length]
        assert out.shape == (self.NUM_NODES, self.FORECAST_LEN)

    def test_forward_pass_no_edge_weight(self, model, edge_index):
        x = torch.randn(self.NUM_NODES, self.SEQ_LEN, self.INPUT_SIZE)
        with torch.no_grad():
            out = model(x, edge_index, edge_weight=None)
        assert out.shape == (self.NUM_NODES, self.FORECAST_LEN)

    def test_invalid_input_dims(self, model, edge_index):
        """Model should raise ValueError for non-3D input."""
        x_bad = torch.randn(self.SEQ_LEN, self.INPUT_SIZE)
        with pytest.raises(ValueError, match="Expected x with dims"):
            model(x_bad, edge_index)


class TestMultiFeatureStreamGCLSTM:
    NUM_NODES = 5
    SEQ_LEN = 13
    FORECAST_LEN = 3
    INPUT_SIZE = 5

    @pytest.fixture
    def model(self):
        return MultiFeatureStreamGCLSTM(
            input_size=self.INPUT_SIZE,
            gclstm_out_channels=16,
            gclstm_K=2,
            mlp_hidden_dim=32,
            mlp_dropout=0.1,
            forecast_length=self.FORECAST_LEN,
        )

    def test_initialization(self, model):
        assert model.forecast_length == self.FORECAST_LEN
        assert model.gclstm_K == 2

    def test_forward_pass(self, model, edge_index):
        x = torch.randn(self.NUM_NODES, self.SEQ_LEN, self.INPUT_SIZE)
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (self.NUM_NODES, self.FORECAST_LEN)
