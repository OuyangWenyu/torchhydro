import torch
import torch.nn as nn


# Figure 4 shows that only static features need embedding
# However neuralhydrology/neuralhydrology/modelzoo/inputlayer.py shows the opposite
# self.dynamics_embedding, self.dynamics_output_size = 
#   self._get_embedding_net(cfg.dynamics_embedding, dynamics_input_size, 'dynamics')
 
class StaticFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout):
        super(StaticFeatureEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, static_features):
        return self.embedding(static_features)

class HindcastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HindcastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        return output, h, c


class ForecastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ForecastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, h, c):
        output, _ = self.lstm(x, (h, c))
        return output

class HiddenStateTransferNet(nn.Module):
    def __init__(self, hidden_dim):
        super(HiddenStateTransferNet, self).__init__()
        self.linear_transfer = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear_transfer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    def forward(self, hidden, cell):
        transfer_hidden = self.nonlinear_transfer(hidden)
        transfer_cell = self.linear_transfer(cell)
        return transfer_hidden, transfer_cell

class ModelOutputHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ModelOutputHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.head(x)


class SequentialForecastLSTM(nn.Module):
    def __init__(
            self, 
            static_input_dim, 
            dynamic_input_dim, 
            embedding_dim, 
            hidden_dim, 
            output_dim,
        ):
        super(SequentialForecastLSTM, self).__init__()
        self.output_dim = output_dim
        self.static_embedding = StaticFeatureEmbedding(static_input_dim, embedding_dim)
        self.hindcast_lstm = HindcastLSTM(dynamic_input_dim + embedding_dim, hidden_dim)
        self.forecast_lstm = ForecastLSTM(dynamic_input_dim + embedding_dim, hidden_dim)
        self.hiddenstatetransfer = HiddenStateTransferNet(hidden_dim=hidden_dim)
        self.output_head = ModelOutputHead(hidden_dim, output_dim)

    def forward(self, *src):
        hindcast_features, static_features_hindcast, forecast_features, static_features_forecast = src
        # Hindcast LSTM
        static_embedded = self.static_embedding(static_features_hindcast)
        static_embedded = static_embedded.unsqueeze(1).expand(-1, hindcast_features.size(1), -1)
        hindcast_input = torch.cat([hindcast_features, static_embedded], dim=-1)
        _, h, c = self.hindcast_lstm(hindcast_input)
        # HiddenStateTransfer
        h, c = self.hiddenstatetransfer(h, c)
        # Forecast LSTM
        static_embedded = self.static_embedding(static_features_forecast)
        static_embedded = static_embedded.unsqueeze(1).expand(-1, forecast_features.size(1), -1)
        forecast_input = torch.cat([forecast_features, static_embedded], dim=-1)
        forecast_output = self.forecast_lstm(forecast_input, h.unsqueeze(0), c.unsqueeze(0))

        # 输出头
        output = self.output_head(forecast_output)
        return output