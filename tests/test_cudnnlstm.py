"""
Author: Wenyu Ouyang
Date: 2024-10-09 14:22:18
LastEditTime: 2024-10-09 16:53:57
LastEditors: Wenyu Ouyang
Description: Test functions for CudnnLstmModel
FilePath: \torchhydro\tests\test_cudnnlstm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import torch
import pytest
from torchhydro.models.cudnnlstm import CudnnLstmModel
from torchhydro.models.cudnnlstm import CudnnLstmModel, CudnnLstm


def test_mc_dropout_eval():
    # Monte Carlo Dropout sampling during evaluation
    model = CudnnLstmModel(
        n_input_features=10, n_output_features=1, n_hidden_states=50, dr=0.5
    )
    model = model.to("cuda:0")
    # simulate input data
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    input_data = input_data.to("cuda:0")
    # multiple forward passes to estimate the model's uncertainty using Monte Carlo Dropout
    num_samples = 10
    mc_outputs = []
    # set to training mode to enable Monte Carlo Dropout
    model.train()
    # during evaluation, we don't need to calculate gradients
    with torch.no_grad():
        for _ in range(num_samples):
            # force the model to use dropout
            output = model(input_data, do_drop_mc=True)
            mc_outputs.append(output)

    # stack the outputs along the first dimension
    mc_outputs = torch.stack(mc_outputs)

    # calculate the mean and variance of the outputs to estimate the model's uncertainty
    mean_output = mc_outputs.mean(dim=0)
    variance_output = mc_outputs.var(dim=0)

    print("mean value: ", mean_output)
    print("var value: ", variance_output)


def test_setstate_with_all_weights():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    # state_dict returns a dictionary containing whole weights and bias of the module
    state_dict = model.state_dict()
    state_dict["all_weights"] = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    model.__setstate__(state_dict)

    assert model._all_weights == [["w_ih", "w_hh", "b_ih", "b_hh"]]


def test_setstate_without_all_weights():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    state_dict = model.state_dict()

    model.__setstate__(state_dict)

    assert model._all_weights == [["w_ih", "w_hh", "b_ih", "b_hh"]]


def test_setstate_with_non_string_all_weights():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    state_dict = model.state_dict()
    state_dict["all_weights"] = [[0, 1, 2, 3]]

    model.__setstate__(state_dict)

    assert model._all_weights == [["w_ih", "w_hh", "b_ih", "b_hh"]]


def test_setstate_with_data_ptrs():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    state_dict = model.state_dict()
    state_dict["_data_ptrs"] = [1, 2, 3]

    model.__setstate__(state_dict)

    assert model.__dict__.get("_data_ptrs") == [1, 2, 3]


def test_reset_mask():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    model.reset_mask()

    assert model.mask_w_ih is not None, "mask_w_ih should not be None after reset_mask"
    assert model.mask_w_hh is not None, "mask_w_hh should not be None after reset_mask"
    assert (
        model.mask_w_ih.shape == model.w_ih.shape
    ), "mask_w_ih should have the same shape as w_ih"
    assert (
        model.mask_w_hh.shape == model.w_hh.shape
    ), "mask_w_hh should have the same shape as w_hh"


def test_reset_mask_with_different_dropout():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.3)
    model.reset_mask()

    assert model.mask_w_ih is not None, "mask_w_ih should not be None after reset_mask"
    assert model.mask_w_hh is not None, "mask_w_hh should not be None after reset_mask"
    assert (
        model.mask_w_ih.shape == model.w_ih.shape
    ), "mask_w_ih should have the same shape as w_ih"
    assert (
        model.mask_w_hh.shape == model.w_hh.shape
    ), "mask_w_hh should have the same shape as w_hh"


def test_reset_mask_with_zero_dropout():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.0)
    model.reset_mask()

    assert model.mask_w_ih is not None, "mask_w_ih should not be None after reset_mask"
    assert model.mask_w_hh is not None, "mask_w_hh should not be None after reset_mask"
    assert (
        model.mask_w_ih.shape == model.w_ih.shape
    ), "mask_w_ih should have the same shape as w_ih"
    assert (
        model.mask_w_hh.shape == model.w_hh.shape
    ), "mask_w_hh should have the same shape as w_hh"


def test_forward_with_mc_dropout():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    model = model.to("cuda:0")
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    input_data = input_data.to("cuda:0")
    model.train()  # Ensure dropout is enabled
    output, (hy, cy) = model(input_data, do_drop_mc=True)

    assert output.shape == (20, 5, 20), "Output shape mismatch"
    assert hy.shape == (1, 5, 20), "Hidden state shape mismatch"
    assert cy.shape == (1, 5, 20), "Cell state shape mismatch"


def test_forward_with_dropout():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    model = model.to("cuda:0")
    input_data = input_data.to("cuda:0")
    model.train()
    output, (hy, cy) = model(input_data, do_drop_mc=False)

    assert output.shape == (20, 5, 20), "Output shape mismatch"
    assert hy.shape == (1, 5, 20), "Hidden state shape mismatch"
    assert cy.shape == (1, 5, 20), "Cell state shape mismatch"


def test_forward_with_zero_dropout():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.0)
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    model = model.to("cuda:0")
    input_data = input_data.to("cuda:0")
    model.train()  # Dropout rate is zero, so dropout should be disabled
    output, (hy, cy) = model(input_data, do_drop_mc=True)

    assert output.shape == (20, 5, 20), "Output shape mismatch"
    assert hy.shape == (1, 5, 20), "Hidden state shape mismatch"
    assert cy.shape == (1, 5, 20), "Cell state shape mismatch"


def test_forward_with_dropout_false():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    model = model.to("cuda:0")
    input_data = input_data.to("cuda:0")
    model.train()  # Ensure dropout is enabled
    output, (hy, cy) = model(input_data, do_drop_mc=False, dropout_false=True)

    assert output.shape == (20, 5, 20), "Output shape mismatch"
    assert hy.shape == (1, 5, 20), "Hidden state shape mismatch"
    assert cy.shape == (1, 5, 20), "Cell state shape mismatch"


def test_forward_with_initial_states():
    model = CudnnLstm(input_size=10, hidden_size=20, dr=0.5)
    input_data = torch.randn(20, 5, 10)  # [seq_len, batch_size, input_size]
    hx = torch.randn(1, 5, 20)
    cx = torch.randn(1, 5, 20)
    model = model.to("cuda:0")
    input_data = input_data.to("cuda:0")
    hx = hx.to("cuda:0")
    cx = cx.to("cuda:0")
    model.train()  # Ensure dropout is enabled
    output, (hy, cy) = model(input_data, hx=hx, cx=cx, do_drop_mc=True)

    assert output.shape == (20, 5, 20), "Output shape mismatch"
    assert hy.shape == (1, 5, 20), "Hidden state shape mismatch"
    assert cy.shape == (1, 5, 20), "Cell state shape mismatch"
