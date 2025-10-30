"""
Author: Wenyu Ouyang
Date: 2024-12-31 18:20:31
LastEditTime: 2025-01-01 18:12:08
LastEditors: Wenyu Ouyang
Description: 
FilePath: \torchhydro\tests\test_seqforecast.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import torch
from torchhydro.models.seqforecast import (
    SequentialForecastLSTM,
    FeatureEmbedding,
    HiddenStateTransferNet,
    HindcastLSTM,
    ForecastLSTM,
    ModelOutputHead,
)


def test_feature_embedding_forward():
    input_dim = 10
    embedding_dim = 5
    batch_size = 2

    model = FeatureEmbedding(input_dim, embedding_dim)
    static_features = torch.randn(batch_size, input_dim)

    output = model.forward(static_features)

    assert output.shape == (batch_size, embedding_dim)
    assert isinstance(output, torch.Tensor)


def test_hindcast_lstm_forward():
    input_dim = 10
    hidden_dim = 20
    batch_size = 2
    seq_len = 5

    model = HindcastLSTM(input_dim, hidden_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    output, h, c = model.forward(x)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert h.shape == (1, batch_size, hidden_dim)
    assert c.shape == (1, batch_size, hidden_dim)
    assert isinstance(output, torch.Tensor)
    assert isinstance(h, torch.Tensor)
    assert isinstance(c, torch.Tensor)


def test_forecast_lstm_forward():
    input_dim = 10
    hidden_dim = 20
    batch_size = 2
    seq_len = 5

    model = ForecastLSTM(input_dim, hidden_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    h = torch.randn(1, batch_size, hidden_dim)
    c = torch.randn(1, batch_size, hidden_dim)

    output = model.forward(x, h, c)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert isinstance(output, torch.Tensor)


def test_hidden_state_transfer_net_forward():
    hindcast_hidden_dim = 20
    forecast_hidden_dim = 30
    batch_size = 2

    model = HiddenStateTransferNet(hindcast_hidden_dim, forecast_hidden_dim)
    hidden = torch.randn(1, batch_size, hindcast_hidden_dim)
    cell = torch.randn(1, batch_size, hindcast_hidden_dim)

    transfer_hidden, transfer_cell = model.forward(hidden, cell)

    assert transfer_hidden.shape == (1, batch_size, forecast_hidden_dim)
    assert transfer_cell.shape == (1, batch_size, forecast_hidden_dim)
    assert isinstance(transfer_hidden, torch.Tensor)
    assert isinstance(transfer_cell, torch.Tensor)


def test_model_output_head_forward():
    hidden_dim = 20
    output_dim = 10
    batch_size = 2
    seq_len = 5

    model = ModelOutputHead(hidden_dim, output_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)

    output = model.forward(x)

    assert output.shape == (batch_size, seq_len, output_dim)
    assert isinstance(output, torch.Tensor)


def test_sequential_forecast_lstm_forward():
    static_input_dim = 10
    dynamic_input_dim = 15
    static_embedding_dim = 5
    sta_embed_hidden_dim = 5
    dynamic_embedding_dim = 7
    dyn_embed_hidden_dim = 7
    hindcast_hidden_dim = 20
    forecast_hidden_dim = 25
    output_dim = 1
    hindcast_output_window = 3
    embedding_dropout = 0.1
    handoff_dropout = 0.1
    lstm_dropout = 0.1

    batch_size = 2
    seq_len = 5
    forecast_len = 4

    model = SequentialForecastLSTM(
        static_input_dim,
        dynamic_input_dim,
        static_embedding_dim,
        sta_embed_hidden_dim,
        dynamic_embedding_dim,
        dyn_embed_hidden_dim,
        hindcast_hidden_dim,
        forecast_hidden_dim,
        output_dim,
        hindcast_output_window,
        embedding_dropout,
        handoff_dropout,
        lstm_dropout,
    )

    hindcast_features = torch.randn(batch_size, seq_len, dynamic_input_dim)
    forecast_features = torch.randn(batch_size, forecast_len, dynamic_input_dim)
    static_features = torch.randn(batch_size, static_input_dim)

    output = model.forward(hindcast_features, forecast_features, static_features)

    expected_output_len = (
        hindcast_output_window + forecast_len
        if hindcast_output_window > 0
        else forecast_len
    )
    assert output.shape == (batch_size, expected_output_len, output_dim)
    assert isinstance(output, torch.Tensor)
