import torch
from torch import nn
from torchhydro.models.balstm_model import SimpleBALSTM,BALSTM


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        input_size_sta,
        input_size_dyn,
        input_size_glo,
        hidden_size,
        output_size,
        dropout=0.4,
        num_layers=2
    ):
        super(GlobalEncoder, self).__init__()
        self.balstm = SimpleBALSTM(
            input_size_sta,
            input_size_dyn,
            input_size_glo,
            hidden_size,
            output_size,
            dropout=dropout,
            num_layers=num_layers
        )

    def forward(self, x_s, x_d, x_g):
        outputs, (hidden, cell) = self.balstm(x_s, x_d, x_g)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        x0 = self.pre_fc(input)
        x1 = self.pre_relu(x0)
        output_, (hidden_, cell_) = self.lstm(x1, (hidden, cell))
        output_dr = self.dropout(output_)
        output = self.fc_out(output_dr)
        return output, hidden_, cell_


class StateTransferNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(StateTransferNetwork, self).__init__()
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, cell):
        transfer_hidden = torch.tanh(self.fc_hidden(hidden))
        transfer_cell = self.fc_cell(cell)
        return transfer_hidden, transfer_cell


class SimpleBALSTM_EncDec(nn.Module):
    def __init__(
        self,
        input_size_sta,
        input_size_dyn,
        input_size_glo,
        hidden_size,
        output_size,
        de_input_size,
        forecast_length,
        hindcast_output_window=0,
        teacher_forcing_ratio=0,
    ):
        super(SimpleBALSTM_EncDec, self).__init__()
        self.trg_len = forecast_length
        self.hindcast_output_window = hindcast_output_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.output_size = output_size
        self.global_encoder = GlobalEncoder(
            input_size_sta, input_size_dyn, input_size_glo, hidden_size, output_size
        )

        self.decoder = Decoder(
            input_dim=de_input_size, hidden_dim=hidden_size, output_dim=output_size
        )
        self.transfer = StateTransferNetwork(hidden_dim=hidden_size)

    def forward(self, *src):
        if len(src) == 5:
            xs, xt, xg, decoder_input, trgs = src
        else:
            xs, xt, xg, decoder_input = src
            device = decoder_input.device
            trgs = torch.full(
                (
                    decoder_input.shape[0],  # batch_size
                    self.hindcast_output_window + self.trg_len,  # seq
                    self.output_size,  # features
                ),
                float("nan"),
            ).to(device)
        encoder_outputs, hidden_, cell_ = self.global_encoder(xs, xt, xg) #eo(batch_size,seq_len,1) hidden(num_layers,batch_size,hidden_size)
        hidden, cell = self.transfer(hidden_, cell_)
        outputs = []
        current_input = encoder_outputs[:, -1, :].unsqueeze(1)

        for t in range(self.trg_len):
            p = decoder_input[:, t, :].unsqueeze(1)
            current_input = torch.cat((current_input, p), dim=2)
            output, hidden, cell = self.decoder(current_input, hidden, cell)
            outputs.append(output.squeeze(1))
            trg = trgs[:, (self.hindcast_output_window + t), :].unsqueeze(1)
            valid_mask = ~torch.isnan(trg)
            random_vals = torch.rand_like(valid_mask, dtype=torch.float)
            use_teacher_forcing = (
                random_vals < self.teacher_forcing_ratio
            ) * valid_mask
            current_input = torch.where(
                torch.isnan(trg),  # if trg is nan
                output,  # then use output
                trg * use_teacher_forcing
                + output
                * (~use_teacher_forcing),  # else calculate with teacher forcing
            )

        outputs = torch.stack(outputs, dim=1)
        if self.hindcast_output_window > 0:
            prec_outputs = encoder_outputs[:, -self.hindcast_output_window :, :]
            outputs = torch.cat((prec_outputs, outputs), dim=1)
        return outputs
