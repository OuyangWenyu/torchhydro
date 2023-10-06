"""
Author: Wenyu Ouyang
Date: 2023-07-11 17:39:09
LastEditTime: 2023-10-06 18:12:54
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
    x
    dr

    Returns
    -------

    """
    return x.new().resize_as_(x).bernoulli_(1 - dr).div_(1 - dr).detach_()


class DropMask(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, mask, train=False, inplace=False):
        ctx.master_train = train
        ctx.inplace = inplace
        ctx.mask = mask

        if not ctx.master_train:
            return input
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output.mul_(ctx.mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.master_train:
            return grad_output * ctx.mask, None, None, None
        else:
            return grad_output, None, None, None
