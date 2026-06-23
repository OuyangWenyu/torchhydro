"""
Author: Wenyu Ouyang
Date: 2025-01-01 10:20:02
LastEditTime: 2025-11-06 15:59:06
LastEditors: Wenyu Ouyang
Description: test function for multi-layer perceptron model
FilePath: \torchhydro\tests\test_ann.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torch import nn
from torchhydro.models.ann import Mlp, SimpleAnn


def test_get_activation_tanh() -> None:
    model = SimpleAnn(1, 1)
    activation = model._get_activation("tanh")
    assert isinstance(activation, nn.Tanh)


def test_get_activation_sigmoid() -> None:
    model = SimpleAnn(1, 1)
    activation = model._get_activation("sigmoid")
    assert isinstance(activation, nn.Sigmoid)


def test_get_activation_relu() -> None:
    model = SimpleAnn(1, 1)
    activation = model._get_activation("relu")
    assert isinstance(activation, nn.ReLU)


def test_get_activation_linear() -> None:
    model = SimpleAnn(1, 1)
    activation = model._get_activation("linear")
    assert isinstance(activation, nn.Identity)


def test_get_activation_not_implemented() -> None:
    model = SimpleAnn(1, 1)
    with pytest.raises(NotImplementedError):
        model._get_activation("unsupported_activation")


def test_forward_single_layer() -> None:
    model = SimpleAnn(3, 2, hidden_size=0)
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_forward_single_layer_wd() -> None:
    model = SimpleAnn(3, 2, hidden_size=0, dr=0.2)
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_forward_multiple_layers() -> None:
    model = SimpleAnn(3, 2, hidden_size=[4, 5], dr=[0.1, 0.2])
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_forward_multiple_layers_wd() -> None:
    model = SimpleAnn(3, 2, hidden_size=[4, 5], dr=[0.1, 0.2, 0.3])
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_forward_with_dropout() -> None:
    model = SimpleAnn(3, 2, hidden_size=[4, 5], dr=[0.1, 0.2])
    model.train()  # Enable dropout
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_forward_activation() -> None:
    model = SimpleAnn(3, 2, hidden_size=[4, 5], activation="sigmoid")
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_mlp_forward_single_layer() -> None:
    model = Mlp(3, 2, hidden_size=0)
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_mlp_forward_multiple_layers() -> None:
    model = Mlp(3, 2, hidden_size=[4, 5], dr=[0.1, 0.2])
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_mlp_forward_with_dropout() -> None:
    model = Mlp(3, 2, hidden_size=[4, 5], dr=[0.1, 0.2, 0.3])
    model.train()  # Enable dropout
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_mlp_forward_activation() -> None:
    model = Mlp(3, 2, hidden_size=[4, 5], activation="sigmoid")
    x = torch.randn(5, 3)
    output = model.forward(x)
    assert output.shape == (5, 2)


def test_mlp_forward_mismatched_layers() -> None:
    with pytest.raises(IndexError):
        Mlp(3, 2, hidden_size=[4, 5], dr=[0.1])  # Mismatched dropout layers
