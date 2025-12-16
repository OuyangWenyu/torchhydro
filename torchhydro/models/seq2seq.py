"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:32:26
LastEditTime: 2024-11-05 19:10:02
LastEditors: Wenyu Ouyang
Description:
FilePath: \torchhydro\torchhydro\models\seq2seq.py
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
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # a nonlinear layer to transform the input
        x0 = self.pre_fc(x)
        x1 = self.pre_relu(x0)
        # the LSTM layer
        outputs_, (hidden, cell) = self.lstm(x1)
        # a dropout layer
        dr_outputs = self.dropout(outputs_)
        # final linear layer
        outputs = self.fc(dr_outputs)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
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


class GeneralSeq2Seq(nn.Module):
    def __init__(
        self,
        en_input_size,
        de_input_size,
        output_size,
        hidden_size,
        forecast_length,
        hindcast_output_window=0,
        teacher_forcing_ratio=0.5,
    ):
        """General Seq2Seq model

        Parameters
        ----------
        en_input_size : _type_
            the size of the input of the encoder
        de_input_size : _type_
            the size of the input of the decoder
        output_size : _type_
            the size of the output, same for encoder and decoder
        hidden_size : _type_
            the size of the hidden state of LSTMs
        forecast_length : _type_
            the length of the forecast, i.e., the periods of decoder outputs
        hindcast_output_window : int, optional
            the encoder's final several outputs in the final output;
            default is 0 which means no encoder output is included in the final output;
        teacher_forcing_ratio : float, optional
            the probability of using teacher forcing
        """
        super(GeneralSeq2Seq, self).__init__()
        self.trg_len = forecast_length
        self.hindcast_output_window = hindcast_output_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.output_size = output_size
        self.encoder = Encoder(
            input_dim=en_input_size, hidden_dim=hidden_size, output_dim=output_size
        )
        self.decoder = Decoder(
            input_dim=de_input_size, hidden_dim=hidden_size, output_dim=output_size
        )
        self.transfer = StateTransferNetwork(hidden_dim=hidden_size)

    def _teacher_forcing_preparation(self, trgs):
        # teacher forcing preparation
        valid_mask = ~torch.isnan(trgs)
        random_vals = torch.rand_like(valid_mask, dtype=torch.float)
        return (random_vals < self.teacher_forcing_ratio) * valid_mask

    def forward(self, *src):
        if len(src) == 3:
            encoder_input, decoder_input, trgs = src
        else:
            encoder_input, decoder_input = src
            device = decoder_input.device
            trgs = torch.full(
                (
                    self.hindcast_output_window + self.trg_len,  # seq
                    decoder_input.shape[1],  # batch_size
                    self.output_size,  # features
                ),
                float("nan"),
            ).to(device)
        trgs_q = trgs[:, :, :1]
        trgs_s = trgs[:, :, 1:]
        trgs = torch.cat((trgs_s, trgs_q), dim=2)  # sq
        encoder_outputs, hidden_, cell_ = self.encoder(encoder_input)  # sq
        hidden, cell = self.transfer(hidden_, cell_)
        outputs = []
        prev_output = encoder_outputs[-1, :, :].unsqueeze(0)  # sq
        _, batch_size, _ = decoder_input.size()

        outputs = torch.zeros(self.trg_len, batch_size, self.output_size).to(
            decoder_input.device
        )
        use_teacher_forcing = self._teacher_forcing_preparation(trgs)
        for t in range(self.trg_len):
            pc = decoder_input[t : t + 1, :, :]  # sq
            obs = trgs[self.hindcast_output_window + t, :, :].unsqueeze(0)  # sq
            safe_obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)
            prev_output = torch.where(  # sq
                use_teacher_forcing[t : t + 1, :, :],
                safe_obs,
                prev_output,
            )
            current_input = torch.cat((pc, prev_output), dim=2)  # pcsq
            output, hidden, cell = self.decoder(current_input, hidden, cell)
            outputs[t, :, :] = output.squeeze(0)  # sq
        if self.hindcast_output_window > 0:
            prec_outputs = encoder_outputs[-self.hindcast_output_window :, :, :]
            outputs = torch.cat((prec_outputs, outputs), dim=0)
        outputs_s = outputs[:, :, :1]
        outputs_q = outputs[:, :, 1:]
        outputs = torch.cat((outputs_q, outputs_s), dim=2)  # qs
        return outputs


class GeneralSeq2SeqNoForecastInput(GeneralSeq2Seq):
    """Seq2Seq model without forecast period forcing input (ablation study version).

    This model is designed for ablation studies where we want to test the model's
    ability to predict future streamflow WITHOUT any future rainfall information.

    Key differences from GeneralSeq2Seq:
    - Decoder does NOT receive forecast period rainfall
    - Decoder only uses previous output (autoregressive)
    - de_input_size should equal output_size (no forcing input)

    Parameters
    ----------
    en_input_size : int
        Encoder input size (e.g., rainfall features for hindcast period)
    de_input_size : int
        Decoder input size (should equal output_size since no forcing input)
        This parameter is kept for compatibility but will be ignored
    output_size : int
        Output size (e.g., streamflow)
    hidden_size : int
        Hidden state size for LSTM
    forecast_length : int
        Number of forecast timesteps
    hindcast_output_window : int, optional
        Number of hindcast outputs to include in final output (default: 0)
    teacher_forcing_ratio : float, optional
        Probability of using teacher forcing (default: 0.5)

    Example
    -------
    >>> model = GeneralSeq2SeqNoForecastInput(
    ...     en_input_size=1,     # encoder: rainfall
    ...     de_input_size=1,     # ignored (will use output_size internally)
    ...     output_size=1,       # output: streamflow
    ...     hidden_size=16,
    ...     forecast_length=24,
    ...     hindcast_output_window=56,
    ...     teacher_forcing_ratio=0.5
    ... )
    """

    def __init__(
        self,
        en_input_size,
        de_input_size,  # This will be ignored, kept for compatibility
        output_size,
        hidden_size,
        forecast_length,
        hindcast_output_window=0,
        teacher_forcing_ratio=0.5,
    ):
        # Call parent init with de_input_size = output_size
        # because decoder only receives previous output (no forcing input)
        super(GeneralSeq2SeqNoForecastInput, self).__init__(
            en_input_size=en_input_size,
            de_input_size=output_size,  # Override: decoder only uses prev output
            output_size=output_size,
            hidden_size=hidden_size,
            forecast_length=forecast_length,
            hindcast_output_window=hindcast_output_window,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        print(f"[GeneralSeq2SeqNoForecastInput] Initialized without forecast forcing input")
        print(f"  Decoder input: only previous output (no rainfall), size={output_size}")

    def forward(self, *src):
        """Forward pass without using forecast period forcing input.

        Parameters
        ----------
        src : tuple
            Can be either:
            - (encoder_input, decoder_input, targets) for training with teacher forcing
            - (encoder_input, decoder_input) for inference
            Note: decoder_input will be IGNORED (not used in computation)

        Returns
        -------
        torch.Tensor
            Output predictions
        """
        if len(src) == 3:
            encoder_input, decoder_input, trgs = src
            # decoder_input is ignored in this model
        else:
            encoder_input, decoder_input = src
            # decoder_input is ignored in this model
            device = encoder_input.device
            trgs = torch.full(
                (
                    self.hindcast_output_window + self.trg_len,
                    encoder_input.shape[1],
                    self.output_size,
                ),
                float("nan"),
            ).to(device)

        # Process targets (same as parent)
        trgs_q = trgs[:, :, :1]
        trgs_s = trgs[:, :, 1:]
        trgs = torch.cat((trgs_s, trgs_q), dim=2)

        # Encoder processes hindcast period
        encoder_outputs, hidden_, cell_ = self.encoder(encoder_input)
        hidden, cell = self.transfer(hidden_, cell_)

        # Initialize outputs and prev_output
        _, batch_size, _ = encoder_input.size()
        outputs = torch.zeros(self.trg_len, batch_size, self.output_size).to(
            encoder_input.device
        )
        prev_output = encoder_outputs[-1, :, :].unsqueeze(0)

        # Prepare teacher forcing
        use_teacher_forcing = self._teacher_forcing_preparation(trgs)

        # Decoder loop: autoregressive prediction WITHOUT forecast forcing input
        for t in range(self.trg_len):
            # Teacher forcing: use real observation if available
            obs = trgs[self.hindcast_output_window + t, :, :].unsqueeze(0)
            safe_obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)
            prev_output = torch.where(
                use_teacher_forcing[t : t + 1, :, :],
                safe_obs,
                prev_output,
            )

            # KEY DIFFERENCE: current_input = prev_output ONLY (no forcing input)
            # In parent class: current_input = cat(decoder_forcing, prev_output)
            # In this class: current_input = prev_output
            current_input = prev_output

            # Decoder step
            output, hidden, cell = self.decoder(current_input, hidden, cell)
            outputs[t, :, :] = output.squeeze(0)
            prev_output = output

        # Include hindcast outputs if specified
        if self.hindcast_output_window > 0:
            prec_outputs = encoder_outputs[-self.hindcast_output_window :, :, :]
            outputs = torch.cat((prec_outputs, outputs), dim=0)

        # Reorder outputs (same as parent)
        outputs_s = outputs[:, :, :1]
        outputs_q = outputs[:, :, 1:]
        outputs = torch.cat((outputs_q, outputs_s), dim=2)

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
        hindcast_output_window=0,
    ):
        """TODO: hindcast_output_window seems not used

        Parameters
        ----------
        n_encoder_inputs : _type_
            _description_
        n_decoder_inputs : _type_
            _description_
        n_decoder_output : _type_
            _description_
        channels : int, optional
            _description_, by default 256
        num_embeddings : int, optional
            _description_, by default 512
        nhead : int, optional
            _description_, by default 8
        num_layers : int, optional
            _description_, by default 8
        dropout : float, optional
            _description_, by default 0.1
        hindcast_output_window : int, optional
            _description_, by default 0
        """
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
