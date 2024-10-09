"""
Author: Wenyu Ouyang
Date: 2024-10-09 14:34:41
LastEditTime: 2024-10-09 15:44:55
LastEditors: Wenyu Ouyang
Description: Test functions for dropout
FilePath: \torchhydro\tests\test_dropout.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torch.autograd import Function
from torchhydro.models.dropout import create_mask
from torchhydro.models.dropout import DropMask


def test_create_mask_shape():
    x = torch.randn(10, 10)
    dr = 0.5
    mask = create_mask(x, dr)
    assert mask.shape == x.shape, "Mask shape should match input shape"


def test_create_mask_values():
    x = torch.randn(10, 10)
    dr = 0.5
    mask = create_mask(x, dr)
    assert torch.all((mask == 0) | (mask == 2)), "Mask values should be either 0 or 2"


def test_create_mask_dropout_rate():
    x = torch.randn(1000, 1000)
    dr = 0.5
    mask = create_mask(x, dr)
    dropout_rate = (mask == 0).float().mean().item()
    assert (
        abs(dropout_rate - dr) < 0.05
    ), "Dropout rate should be close to the specified rate"


def test_create_mask_no_dropout():
    x = torch.randn(10, 10)
    dr = 0.0
    mask = create_mask(x, dr)
    assert torch.all(mask == 1), "Mask should be all ones when dropout rate is 0"


def test_create_mask_full_dropout():
    x = torch.randn(10, 10)
    dr = 1.0
    mask = create_mask(x, dr)
    assert torch.all(mask == 0), "Mask should be all zeros when dropout rate is 1"


class MockContext(Function):
    def __init__(self):
        super(MockContext, self).__init__()
        self.master_train = None
        self.inplace = None
        self.mask = None


def test_forward_train_inplace():
    ctx = MockContext()
    input = torch.randn(10, 10)
    mask = torch.randint(0, 2, (10, 10)).float()
    ctx.master_train = True
    ctx.inplace = True
    output = DropMask.forward(ctx, input, mask, train=True, inplace=True)
    assert torch.equal(
        output, input * mask
    ), "Output should be input multiplied by mask when training and inplace"


def test_forward_train_not_inplace():
    ctx = MockContext()
    input = torch.randn(10, 10)
    mask = torch.randint(0, 2, (10, 10)).float()
    ctx.master_train = True
    ctx.inplace = False
    output = DropMask.forward(ctx, input, mask, train=True, inplace=False)
    assert torch.equal(
        output, input * mask
    ), "Output should be input multiplied by mask when training and not inplace"
    assert not torch.equal(
        output, input
    ), "Output should not be the same as input when not inplace"


def test_forward_no_train():
    ctx = MockContext()
    input = torch.randn(10, 10)
    mask = torch.randint(0, 2, (10, 10)).float()
    ctx.master_train = False
    ctx.inplace = False
    output = DropMask.forward(ctx, input, mask, train=False, inplace=False)
    assert torch.equal(
        output, input
    ), "Output should be the same as input when not training"


def test_forward_no_train_inplace():
    ctx = MockContext()
    input = torch.randn(10, 10)
    mask = torch.randint(0, 2, (10, 10)).float()
    ctx.master_train = False
    ctx.inplace = True
    output = DropMask.forward(ctx, input, mask, train=False, inplace=True)
    assert torch.equal(
        output, input
    ), "Output should be the same as input when not training and inplace"


def test_backward_train():
    ctx = MockContext()
    ctx.master_train = True
    ctx.mask = torch.randint(0, 2, (10, 10)).float()
    grad_output = torch.randn(10, 10)
    grad_input, _, _, _ = DropMask.backward(ctx, grad_output)
    assert torch.equal(
        grad_input, grad_output * ctx.mask
    ), "Gradient input should be masked when training"


def test_backward_no_train():
    ctx = MockContext()
    ctx.master_train = False
    ctx.mask = torch.randint(0, 2, (10, 10)).float()
    grad_output = torch.randn(10, 10)
    grad_input, _, _, _ = DropMask.backward(ctx, grad_output)
    assert torch.equal(
        grad_input, grad_output
    ), "Gradient input should be the same as gradient output when not training"
