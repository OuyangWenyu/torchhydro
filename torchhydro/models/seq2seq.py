"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:32:26
LastEditTime: 2024-04-17 12:33:34
LastEditors: Xinzhuo Wu
Description:
FilePath: /torchhydro/torchhydro/models/seq2seq.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return F.softmax(energy.squeeze(2), dim=1)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.shape[1]
        hidden_transformed = self.W_q(hidden).repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs_transformed = self.W_k(encoder_outputs)
        combined = torch.tanh(hidden_transformed + encoder_outputs_transformed)
        scores = self.v(combined).squeeze(2)
        return F.softmax(scores, dim=1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, hidden):
        hidden_dim = encoder_outputs.shape[2]
        hidden_expanded = hidden.unsqueeze(1)
        scores = torch.bmm(
            hidden_expanded, encoder_outputs.transpose(1, 2)
        ) / math.sqrt(hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights.squeeze(1)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pre_fc(x)
        x = self.pre_relu(x)
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        x = self.pre_fc(input)
        x = self.pre_relu(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.dropout(output)
        output = self.fc_out(output)
        return output, hidden, cell


class StateTransferNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(StateTransferNetwork, self).__init__()
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, cell):
        transfer_hidden = torch.tanh(self.fc_hidden(hidden))
        transfer_cell = self.fc_cell(cell)
        return transfer_hidden, transfer_cell


class GeneralSeq2Seq(nn.Module):
    def __init__(
        self,
        en_input_size,
        de_input_size,
        output_size,
        hidden_size,
        forecast_length,
        prec_window=0,
        teacher_forcing_ratio=0.5,
    ):
        super(GeneralSeq2Seq, self).__init__()
        self.trg_len = forecast_length
        self.prec_window = prec_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = Encoder(
            input_dim=en_input_size, hidden_dim=hidden_size, output_dim=output_size
        )
        self.decoder = Decoder(
            input_dim=de_input_size, hidden_dim=hidden_size, output_dim=output_size
        )
        self.transfer = StateTransferNetwork(hidden_dim=hidden_size)

    def forward(self, *src):
        if len(src) == 3:
            src1, src2, trgs = src
        else:
            src1, src2 = src
            trgs = None
        encoder_outputs, hidden, cell = self.encoder(src1)
        hidden, cell = self.transfer(hidden, cell)
        outputs = []
        current_input = encoder_outputs[:, -1, :].unsqueeze(1)

        for t in range(self.trg_len):
            p = src2[:, t, :].unsqueeze(1)
            current_input = torch.cat((current_input, p), dim=2)
            output, hidden, cell = self.decoder(current_input, hidden, cell)
            outputs.append(output.squeeze(1))
            if trgs is None or self.teacher_forcing_ratio <= 0:
                current_input = output
            else:
                sm_trg = trgs[:, (self.prec_window + t), 1].unsqueeze(1).unsqueeze(1)
                if not torch.any(torch.isnan(sm_trg)).item():
                    use_teacher_forcing = random.random() < self.teacher_forcing_ratio
                    str_trg = output[:, :, 0].unsqueeze(2)
                    current_input = (
                        torch.cat((str_trg, sm_trg), dim=2)
                        if use_teacher_forcing
                        else output
                    )
                else:
                    current_input = output

        outputs = torch.stack(outputs, dim=1)
        prec_outputs = encoder_outputs[:, -self.prec_window, :].unsqueeze(1)
        outputs = torch.cat((prec_outputs, outputs), dim=1)
        return outputs


class DataEnhancedModel(GeneralSeq2Seq):
    def __init__(self, hidden_length, **kwargs):
        super(DataEnhancedModel, self).__init__(**kwargs)
        self.lstm = nn.LSTM(1, hidden_length, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_length, 6)

    def forward(self, *src):
        src1, src2, token = src
        processed_src1 = torch.unsqueeze(src1[:, :, 0], dim=2)
        out_src1, _ = self.lstm(processed_src1)
        out_src1 = self.fc(out_src1)
        combined_input = torch.cat((out_src1, src1[:, :, 1:]), dim=2)
        return super(DataEnhancedModel, self).forward(combined_input, src2, token)


class DataFusionModel(DataEnhancedModel):
    def __init__(self, input_dim, **kwargs):
        super(DataFusionModel, self).__init__(**kwargs)
        self.input_dim = input_dim

        self.fusion_layer = nn.Conv1d(
            in_channels=input_dim, out_channels=1, kernel_size=1
        )

    def forward(self, *src):
        src1, src2, token = src
        if self.input_dim == 3:
            processed_src1 = self.fusion_layer(
                src1[:, :, 0:3].permute(0, 2, 1)
            ).permute(0, 2, 1)
            combined_input = torch.cat((processed_src1, src1[:, :, 3:]), dim=2)
        else:
            processed_src1 = self.fusion_layer(
                src1[:, :, 0:2].permute(0, 2, 1)
            ).permute(0, 2, 1)
            combined_input = torch.cat((processed_src1, src1[:, :, 2:]), dim=2)

        return super(DataFusionModel, self).forward(combined_input, src2, token)


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class Transformer(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        n_decoder_output,
        channels=256,
        num_embeddings=512,
        nhead=8,
        num_layers=8,
        dropout=0.1,
        prec_window=0,
    ):
        super().__init__()

        self.input_pos_embedding = torch.nn.Embedding(num_embeddings, channels)
        self.target_pos_embedding = torch.nn.Embedding(num_embeddings, channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers)

        self.input_projection = nn.Linear(n_encoder_inputs, channels)
        self.output_projection = nn.Linear(n_decoder_inputs, channels)

        self.linear = nn.Linear(channels, n_decoder_output)

        self.do = nn.Dropout(p=dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        src = self.do(src)
        return src

    def decode_trg(self, trg, memory):
        trg_start = self.output_projection(trg)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)
        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start
        trg_mask = gen_trg_mask(out_sequence_len, trg.device)
        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start
        out = self.do(out)
        out = self.linear(out)
        return out

    def forward(self, *x):
        src, trg = x
        src = self.encode_src(src)
        return self.decode_trg(trg=trg, memory=src)
