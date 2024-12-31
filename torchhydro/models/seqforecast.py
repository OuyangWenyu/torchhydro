import torch
import torch.nn as nn


# Figure 4 shows that only static features need embedding
# However neuralhydrology/neuralhydrology/modelzoo/inputlayer.py shows the opposite
# self.dynamics_embedding, self.dynamics_output_size =
#   self._get_embedding_net(cfg.dynamics_embedding, dynamics_input_size, 'dynamics')


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout=0):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, static_features):
        return self.embedding(static_features)


class HindcastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(HindcastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        return output, h, c


class ForecastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(ForecastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

    def forward(self, x, h, c):
        output, _ = self.lstm(x, (h, c))
        return output


class HiddenStateTransferNet(nn.Module):
    def __init__(self, hindcast_hidden_dim, forecast_hidden_dim, dropout=0):
        super(HiddenStateTransferNet, self).__init__()
        self.linear_transfer = nn.Linear(hindcast_hidden_dim, forecast_hidden_dim)
        self.nonlinear_transfer = nn.Sequential(
            nn.Linear(hindcast_hidden_dim, forecast_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden, cell):
        transfer_hidden = self.nonlinear_transfer(hidden)
        transfer_cell = self.linear_transfer(cell)
        return transfer_hidden, transfer_cell


class ModelOutputHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ModelOutputHead, self).__init__()
        self.head = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.head(x)


class SequentialForecastLSTM(nn.Module):
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        static_embedding_dim,
        dynamic_embedding_dim,
        hindcast_hidden_dim,
        forecast_hidden_dim,
        output_dim,
        hindcast_len,
        embedding_dropout,
        handoff_dropout,
        lstm_dropout,
    ):
        """_summary_

        Parameters
        ----------
        static_input_dim : _type_
            _description_
        dynamic_input_dim : _type_
            _description_
        static_embedding_dim : _type_
            _description_
        dynamic_embedding_dim : _type_
            _description_
        hidden_dim : _type_
            _description_
        output_dim : _type_
            _description_
        hindcast_len : int, optional
            length of hindcast output to calculate loss, by default 0
        """
        super(SequentialForecastLSTM, self).__init__()
        self.output_dim = output_dim
        self.hindcast_len = hindcast_len
        self.dynamic_embedding_dim = dynamic_embedding_dim
        if static_embedding_dim > 0:
            self.static_embedding = FeatureEmbedding(
                static_input_dim, static_embedding_dim, embedding_dropout
            )
        if dynamic_embedding_dim > 0:
            self.dynamic_embedding = FeatureEmbedding(
                dynamic_input_dim, dynamic_embedding_dim, embedding_dropout
            )
        self.static_embedding_dim = static_embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        hindcast_input_dim = (
            dynamic_embedding_dim if dynamic_embedding_dim != 0 else dynamic_input_dim
        ) + (static_embedding_dim if static_embedding_dim != 0 else static_input_dim)
        forecast_input_dim = (
            dynamic_embedding_dim if dynamic_embedding_dim != 0 else dynamic_input_dim
        ) + (static_embedding_dim if static_embedding_dim != 0 else static_input_dim)
        self.hindcast_lstm = HindcastLSTM(
            hindcast_input_dim, hindcast_hidden_dim, lstm_dropout
        )
        self.forecast_lstm = ForecastLSTM(
            forecast_input_dim, forecast_hidden_dim, lstm_dropout
        )
        self.hiddenstatetransfer = HiddenStateTransferNet(
            hindcast_hidden_dim, forecast_hidden_dim, dropout=handoff_dropout
        )
        self.hindcast_output_head = ModelOutputHead(hindcast_hidden_dim, output_dim)
        self.forecast_output_head = ModelOutputHead(forecast_hidden_dim, output_dim)

    def _perform_embedding(self, static_features, dynamic_features):
        if self.dynamic_embedding_dim > 0:
            dynamic_embedded = self.dynamic_embedding(dynamic_features)
        else:
            dynamic_embedded = dynamic_features
        if self.static_embedding_dim > 0:
            static_embedded = self.static_embedding(static_features)
        else:
            static_embedded = static_features
        static_embedded = static_embedded.unsqueeze(1).expand(
            -1, dynamic_embedded.size(1), -1
        )
        return torch.cat([dynamic_embedded, static_embedded], dim=-1)

    def forward(self, *src):
        (
            hindcast_features,
            forecast_features,
            static_features,
        ) = src

        # Hindcast LSTM
        hindcast_input = self._perform_embedding(static_features, hindcast_features)
        hincast_output, h, c = self.hindcast_lstm(hindcast_input)

        if self.hindcast_len > 0:
            hincast_output = self.hindcast_output_head(
                hincast_output[:, -self.hindcast_len :, :]
            )

        h, c = self.hiddenstatetransfer(h, c)

        # Forecast LSTM
        forecast_input = self._perform_embedding(static_features, forecast_features)
        forecast_output = self.forecast_lstm(forecast_input, h, c)
        forecast_output = self.forecast_output_head(forecast_output)
        if self.hindcast_len > 0:
            forecast_output = torch.cat([hincast_output, forecast_output], dim=1)
        return forecast_output
