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


class SMEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SMEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        input_channels=None,
        mode="single",
        prec_window=0,
        num_layers=1,
    ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.prec_window = prec_window
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.mode = mode
        if self.mode == "single":
            self.sm_encoder1 = SMEncoder(input_channels, output_channels=1)
            self.sm_encoder2 = SMEncoder(input_channels, output_channels=1)

    def forward(self, x):
        if self.mode == "single":
            src1, src2 = x
            sm_encoded1 = self.sm_encoder1(src2[:, :, :, 0])
            sm_encoded2 = self.sm_encoder1(src2[:, :, :, 1])
            src_combined = torch.cat((src1, sm_encoded1), dim=2)
            src_combined = torch.cat((src_combined, sm_encoded2), dim=2)
            outputs, (hidden, cell) = self.lstm(src_combined)
        else:
            outputs, (hidden, cell) = self.lstm(x)
        pred_outputs = self.fc(outputs)
        prec_outputs = self.fc2(outputs)
        token = prec_outputs[:, -1, :].unsqueeze(1)

        if self.prec_window != 0:
            prec_outputs = prec_outputs[:, -self.prec_window, :].unsqueeze(1)
        else:
            prec_outputs = None
        return (
            pred_outputs,
            hidden,
            cell,
            token,
            prec_outputs,
        )


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


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            hidden_dim + output_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = DotProductAttention()

    def forward(self, input, hidden, cell, encoder_outputs):
        attention_weights = self.attention(encoder_outputs, hidden.squeeze(0))
        weighted_context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((input, weighted_context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class GeneralSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        forecast_length,
        cnn_size=None,
        model_mode="single",
        prec_window=0,
        teacher_forcing_ratio=0.5,
    ):
        super(GeneralSeq2Seq, self).__init__()
        self.mode = model_mode
        self.trg_len = forecast_length
        self.prec_window = prec_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder1 = Encoder(
            input_dim=input_size,
            hidden_dim=hidden_size,
            input_channels=cnn_size,
            mode=self.mode,
            prec_window=prec_window,
        )
        self.decoder1 = Decoder(output_dim=output_size, hidden_dim=hidden_size)
        self.encoder2 = (
            Encoder(
                input_dim=2,
                hidden_dim=hidden_size,
                mode=self.mode,
                prec_window=prec_window,
            )
            if self.mode != "single"
            else None
        )
        self.decoder2 = (
            Decoder(output_dim=1, hidden_dim=hidden_size)
            if self.mode != "single"
            else None
        )

    def forward(self, *src):
        if self.mode != "single":
            return self.process_dual(src, self.trg_len)
        return self.process_single(src, self.trg_len)

    def process_single(self, src, trg_len):
        outputs = self.process_encoder_decoder(
            self.encoder1, self.decoder1, src[:1], trg_len, src[2]
        )
        return outputs.permute(1, 0, 2)

    def process_dual(self, src, trg_len):
        src1, src2, src3 = src
        outputs1 = self.process_encoder_decoder(
            self.encoder1, self.decoder1, src1, trg_len, src3
        )
        outputs2 = self.process_encoder_decoder(
            self.encoder2, self.decoder2, src2, trg_len, src3
        )

        runoff_coefficients = torch.sigmoid(outputs2)
        final_outputs = outputs1 * runoff_coefficients
        return final_outputs.permute(1, 0, 2)

    def process_encoder_decoder(self, encoder, decoder, src, trg_len, trgs):
        encoder_outputs, hidden, cell, current_input, prec_outputs = encoder(src)

        outputs = []
        for t in range(trg_len):
            output, hidden, cell = decoder(current_input, hidden, cell, encoder_outputs)
            outputs.append(output)

            if isinstance(self.encoder1, type(self.encoder1)):
                trg = trgs[:, (self.prec_window + t), :].unsqueeze(1)
                if torch.any(torch.isnan(trg)).item():
                    current_input = output.unsqueeze(1)
                else:
                    use_teacher_forcing = random.random() < self.teacher_forcing_ratio
                    current_input = trg if use_teacher_forcing else output.unsqueeze(1)

            current_input = output.unsqueeze(1)
        outputs = torch.stack(outputs, dim=0)

        if prec_outputs is not None:
            outputs = torch.cat((prec_outputs.permute(1, 0, 2), outputs), dim=0)

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
