"""
Author: Wenyu Ouyang
Date: 2023-07-11 17:39:09
LastEditTime: 2024-10-09 15:27:49
LastEditors: Wenyu Ouyang
Description: Functions for dropout. Code is from Kuai Fang's repo: hydroDL
FilePath: \torchhydro\torchhydro\models\dropout.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import torch.nn


def create_mask(x, dr):
    """
    Dropout method in Gal & Ghahramami: A Theoretically Grounded Application of Dropout in RNNs.
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    dr: float
        dropout rate

    Returns
    -------
    torch.Tensor
        mask tensor
    """
    # x.new() creates a new tensor with the same data type as x
    # bernoulli_(1-dr) creates a tensor with the same shape as x, filled with 0 or 1, where 1 has a probability of 1-dr
    # div_(1-dr) divides the tensor by 1-dr, so that the expected value of the tensor is the same as x, for example, if dr=0.5, then the expected value of the tensor is 2*x
    # detach_() can be used to detach the tensor from the computation graph so that the gradient is not calculated
    # if dr=1, then the tensor is all zeros, the results are all NaNs if using the code with bernoulli_(1 - dr).div_(1 - dr).detach_()
    # so we need to add a special case for dr=1
    if dr == 1:
        # add a warning message
        print("Warning: dropout rate is 1, directly set 0.")
        return x.new().resize_as_(x).zero_().detach_()
    return x.new().resize_as_(x).bernoulli_(1 - dr).div_(1 - dr).detach_()


class DropMask(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, mask, train=False, inplace=False):
        """_summary_

        Parameters
        ----------
        ctx : autograd.Function
            ctx is a context object that can be used to store information for backward computation
        input : _type_
            _description_
        mask : _type_
            _description_
        train : bool, optional
            if the model is in training mode, by default False
        inplace : bool, optional
            inplace operation, by default False

        Returns
        -------
        _type_
            _description_
        """
        ctx.master_train = train
        ctx.inplace = inplace
        ctx.mask = mask

        if not ctx.master_train:
            # if not in training mode, just return the input
            return input
        if ctx.inplace:
            # mark_dirty() is used to mark the input as dirty, meaning inplace operation is performed
            # make it dirty so that the gradient is calculated correctly during backward
            ctx.mark_dirty(input)
            output = input
        else:
            # clone the input tensor so that avoid changing the input tensor
            output = input.clone()
        # inplace multiplication with the mask
        output.mul_(ctx.mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward method for DropMask
        staticmethod means that the method belongs to the class itself and not to the object of the class

        Parameters
        ----------
        ctx : _type_
            store information for backward computation
        grad_output : _type_
            gradient of the downstream layer

        Returns
        -------
        _type_
            _description_
        """
        if ctx.master_train:
            # if in training mode, return the gradient multiplied by the mask
            return grad_output * ctx.mask, None, None, None
        else:
            # if not in training mode, return the gradient directly
            return grad_output, None, None, None
