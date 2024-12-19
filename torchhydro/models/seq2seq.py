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
from torch.nn import LayerNorm
import math
import random
import dgl
from dgl.nn.pytorch.conv import gatv2conv
from torchhydro.models.minlstm import MinLSTM

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
                        else output)
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


class EncoderGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph, device, num_layers=1, dropout=0.3, num_heads=4):
        super(EncoderGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        # self.pre_relu = nn.ReLU()
        self.graph = graph.to(device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=False)
        # (fc_src): Linear(in_features=256, out_features=2048, bias=True), 2048=8*256
        self.gnn = gatv2conv.GATv2Conv(hidden_dim, hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size=256, seq_length=240, features=24)
        x = self.pre_fc(x)
        # x = self.pre_relu(x)
        outputs, (hidden, cell) = self.lstm(x)
        '''
        node_amount = self.graph.num_nodes()
        dim_diff = node_amount - outputs.size(0)
        if dim_diff >= 0:
            out_res_matrix = torch.zeros(dim_diff, outputs.size(1), outputs.size(2)).to(x.device)
            outputs = torch.cat([outputs, out_res_matrix], dim=0)
        else:
            self.graph = dgl.add_nodes(self.graph, abs(dim_diff))
            self.graph = dgl.add_self_loop(self.graph)
        '''
        gnn_outputs = torch.tensor([]).to(x.device)
        for i in range(outputs.shape[1]):
            output_g = self.gnn(graph=self.graph, feat=outputs[:, i, :].unsqueeze(1))
            gnn_outputs = torch.cat([gnn_outputs, output_g], dim=1)
        gnn_outputs = self.dropout(gnn_outputs)
        gnn_outputs = self.fc(gnn_outputs)
        # gnn_outputs = gnn_outputs * torch.nn.sigmoid(gnn_outputs)
        return gnn_outputs, hidden, cell


class DecoderGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, graph, device, num_layers=1, dropout=0.3, num_heads=1):
        super(DecoderGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        # self.pre_relu = nn.ReLU()
        self.graph = graph.to(device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=False)
        self.gnn = gatv2conv.GATv2Conv(hidden_dim, hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        x = self.pre_fc(input)
        # x = self.pre_relu(x)
        gnn_output = torch.tensor([]).to(x.device)
        hiddens = torch.tensor([]).to(x.device)
        cells = torch.tensor([]).to(x.device)
        # outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        '''
        node_amount = self.graph.num_nodes()
        dim_diff = node_amount - x.size(0)
        if dim_diff >= 0:
            out_res_matrix = torch.zeros(dim_diff, x.size(1), x.size(2)).to(x.device)
            x = torch.cat([x, out_res_matrix], dim=0)
        else:
            self.graph = dgl.add_nodes(self.graph, abs(dim_diff))
            self.graph = dgl.add_self_loop(self.graph)
        '''
        for i in range(x.shape[1]):
            output, (hidden, cell) = self.lstm(x, (hidden[:, i, :].unsqueeze(1), cell[:, i, :].unsqueeze(1)))
            output = self.gnn(graph=self.graph, feat=output[:, i, :])
            gnn_output = torch.cat([gnn_output, output], dim=1)
            hiddens = torch.cat([hiddens, hidden], dim=1)
            cells = torch.cat([cells, cell], dim=1)
        gnn_output = self.dropout(gnn_output)
        gnn_output = self.fc_out(gnn_output)
        return gnn_output, hiddens, cells


class Seq2SeqGNN(nn.Module):
    def __init__(
        self,
        en_input_size,
        de_input_size,
        output_size,
        hidden_size,
        forecast_length,
        graph,
        device="cpu",
        prec_window=0,
        teacher_forcing_ratio=0.5,
    ):
        # 为对齐维度，可能GNN层不能沿用LSTM的输入输出维度
        super(Seq2SeqGNN, self).__init__()
        self.trg_len = forecast_length
        self.prec_window = prec_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.graph = dgl.add_self_loop(graph).to(device)
        self.encoder = EncoderGNN(input_dim=en_input_size, hidden_dim=hidden_size, output_dim=output_size,
                                  graph=self.graph, device=device)
        self.decoder = DecoderGNN(
            input_dim=de_input_size, hidden_dim=hidden_size, output_dim=output_size, graph=self.graph, device=device)

    def forward(self, *src):
        if len(src) == 3:
            src1, src2, trgs = src
        else:
            src1, src2 = src
            trgs = None
        encoder_outputs, hidden, cell = self.encoder(src1)
        outputs = []
        current_input = encoder_outputs[:src2.size(0), -1, :].unsqueeze(1)
        for t in range(self.trg_len):
            p = src2[:, t, :].unsqueeze(1)
            current_input = torch.cat((current_input[:src2.size(0), :, :], p), dim=2)
            output, hidden, cell = self.decoder(current_input, hidden, cell)
            outputs.append(output.squeeze(1))
            if trgs is None or self.teacher_forcing_ratio <= 0:
                current_input = output[:src2.size(0), :, :]
            else:
                sm_trg = trgs[:src2.size(0), (self.prec_window + t), 1].unsqueeze(1).unsqueeze(1)
                if not torch.any(torch.isnan(sm_trg)).item():
                    use_teacher_forcing = random.random() < self.teacher_forcing_ratio
                    # 是否应该限制output.shape[0]?
                    str_trg = output[:src2.size(0), :, 0].unsqueeze(2)
                    current_input = (torch.cat((str_trg, sm_trg), dim=2) if use_teacher_forcing else output)
                else:
                    current_input = output[:src2.size(0), :, :]
        outputs = torch.stack(outputs, dim=1)[:src2.size(0), :, :]
        prec_outputs = encoder_outputs[:src2.size(0), -self.prec_window, :].unsqueeze(1)
        outputs = torch.cat((prec_outputs, outputs), dim=1)
        return outputs

class EncoderMinLSTMGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph, seq_length, dropout=0.1, num_heads=4):
        super(EncoderMinLSTMGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.ln0 = LayerNorm(input_dim)
        # self.pre_fc = nn.Linear(input_dim, hidden_dim)
        # self.pre_relu = nn.ReLU()
        self.graph = dgl.batch([graph] * seq_length)
        self.gnn0 = gatv2conv.GATv2Conv(input_dim, input_dim, num_heads=num_heads)
        self.gnn1 = gatv2conv.GATv2Conv(input_dim * num_heads, input_dim, num_heads=1)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.concat([x[:, :, 0].unsqueeze(-1), torch.log10(x[:, :, 1:]+1)], dim=-1)
        self.graph = self.graph.to(x.device)
        # lnx = self.ln0(x)
        # x = self.pre_fc(x)
        # 将nan预处理为0是否正确？
        x = torch.where(x.isnan(), 0, x)
        gnn_outputs0 = self.gnn0(self.graph, feat=x)
        gnn_outputs1 = self.gnn1(self.graph, feat=gnn_outputs0.reshape(gnn_outputs0.shape[0], -1))
        gnn_outputs1 = gnn_outputs1.reshape(self.hidden_dim, self.seq_length, self.input_dim)
        # PreNorm, https://kexue.fm/archives/9009
        gnn_outputs = torch.where(torch.isnan(gnn_outputs1), x, gnn_outputs1)
        gnn_outputs = x + gnn_outputs
        gnn_outputs = self.dropout(gnn_outputs)
        gnn_outputs = self.fc(self.ln0(gnn_outputs))
        return gnn_outputs


class DecoderMinLSTMGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device, dropout=0.1):
        super(DecoderMinLSTMGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = MinLSTM(input_dim, input_dim, device=device)
        self.lstm1 = MinLSTM(input_dim, input_dim, device=device)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        # input = torch.where(torch.isnan(input), 0, input)
        outputs = self.lstm1(self.lstm(input))
        outputs = torch.where(torch.isnan(outputs), input, outputs)
        outputs = input + outputs
        gnn_output = self.fc_out(self.ln1(outputs))
        return gnn_output #, hiddens, cells


class Seq2Seq_Min_LSTM_GNN(nn.Module):
    def __init__(
        self,
        en_input_size,
        de_input_size,
        output_size,
        hidden_size,
        forecast_history,
        forecast_length,
        graph,
        device="cpu",
        prec_window=0,
        teacher_forcing_ratio=0,
    ):
        # 考虑采取mixup和swish激活函数改善指标
        super(Seq2Seq_Min_LSTM_GNN, self).__init__()
        self.trg_len = forecast_length
        self.prec_window = prec_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.forecast_history = forecast_history
        self.graph = dgl.add_self_loop(graph)
        self.encoder = EncoderMinLSTMGNN(input_dim=en_input_size, hidden_dim=hidden_size, output_dim=output_size,
                                         graph=self.graph, seq_length=forecast_history)
        self.decoder = DecoderMinLSTMGNN(
            input_dim=de_input_size, hidden_dim=hidden_size, output_dim=output_size, device=device)

    def forward(self, *src):
        if len(src) == 3:
            src1, src2, trgs = src
        else:
            src1, src2 = src
            trgs = None
        encoder_outputs = self.encoder(src1)
        outputs = []
        current_input = encoder_outputs[:src2.size(0), -1, :].unsqueeze(1)
        for t in range(self.trg_len):
            p = torch.log(src2[:, t, :].unsqueeze(1)+1)
            current_input = torch.cat((current_input[:src2.size(0), :, :], p), dim=2)
            output = self.decoder(current_input)
            outputs.append(output.squeeze(1))
            if trgs is None or self.teacher_forcing_ratio <= 0:
                current_input = output[:src2.size(0), :, :]
            else:
                sm_trg = trgs[:src2.size(0), (self.prec_window + t), 1].unsqueeze(1).unsqueeze(1)
                if not torch.any(torch.isnan(sm_trg)).item():
                    use_teacher_forcing = random.random() < self.teacher_forcing_ratio
                    str_trg = output[:src2.size(0), :, 0].unsqueeze(2)
                    current_input = (torch.cat((str_trg, sm_trg), dim=2) if use_teacher_forcing else output)
                else:
                    current_input = output[:src2.size(0), :, :]
        outputs = torch.stack(outputs, dim=1)[:src2.size(0), :, :]
        prec_outputs = encoder_outputs[:src2.size(0), -self.prec_window, :].unsqueeze(1)
        outputs = torch.cat((prec_outputs, outputs), dim=1)
        return outputs
