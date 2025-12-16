"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2025-07-13 16:36:07
LastEditors: Wenyu Ouyang
Description: Loss functions
FilePath: \torchhydro\torchhydro\models\crits.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union
import torch
from torch import distributions as tdist, Tensor
from torchhydro.models.model_utils import get_the_device
from abc import ABC, abstractmethod


def deal_gap_data(output, target, data_gap, device):
    """
    How to handle with gap data

    When there are NaN values in observation, we will perform a "reduce" operation on prediction.
    For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
    then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
    Notice: when "sum", actually final index is not chosen,
    because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
    Just as the rho is 5, the final one is not chosen

    Parameters
    ----------
    output
        model output for k-th variable
    target
        target for k-th variable
    data_gap
        data_gap=1: reduce is sum
        data_gap=2: reduce is mean
    device
        where to save the data

    Returns
    -------
    tuple[tensor, tensor]
        output and target after dealing with gap
    """
    # all members in a batch has different NaN-gap, so we need a loop
    seg_p_lst = []
    seg_t_lst = []
    for j in range(target.shape[1]):
        non_nan_idx = torch.nonzero(
            ~torch.isnan(target[:, j]), as_tuple=False
        ).squeeze()
        if len(non_nan_idx) < 1:
            raise ArithmeticError("All NaN elements, please check your data")

        # 使用 cumsum 生成 scatter_index
        is_not_nan = ~torch.isnan(target[:, j])
        cumsum_is_not_nan = torch.cumsum(is_not_nan.to(torch.int), dim=0)
        first_non_nan = non_nan_idx[0]
        scatter_index = torch.full_like(
            target[:, j], fill_value=-1, dtype=torch.long
        )  # 将所有值初始化为 -1
        scatter_index[first_non_nan:] = cumsum_is_not_nan[first_non_nan:] - 1
        scatter_index = scatter_index.to(device=device)

        # 创建掩码，只保留有效的索引
        valid_mask = scatter_index >= 0

        if data_gap == 1:
            seg = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(0, scatter_index[valid_mask], output[valid_mask, j])
            # for sum, better exclude final non-nan value as it didn't include all necessary periods
            seg_p_lst.append(seg[:-1])
            seg_t_lst.append(target[non_nan_idx[:-1], j])

        elif data_gap == 2:
            counts = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(
                0,
                scatter_index[valid_mask],
                torch.ones_like(output[valid_mask, j], dtype=output.dtype),
            )
            seg = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(0, scatter_index[valid_mask], output[valid_mask, j])
            seg = seg / counts.clamp(min=1)
            # for mean, we can include all periods
            seg_p_lst.append(seg)
            seg_t_lst.append(target[non_nan_idx, j])
        else:
            raise NotImplementedError(
                "We have not provided this reduce way now!! Please choose 1 or 2!!"
            )

    p = torch.cat(seg_p_lst)
    t = torch.cat(seg_t_lst)
    return p, t


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior="gauss"):
        super(SigmaLoss, self).__init__()
        self.reduction = "elementwise_mean"
        self.prior = None if prior == "" else prior.split("+")

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == "gauss":
                loss = torch.exp(-s).mul((p - t) ** 2) / 2 + s / 2
            elif self.prior[0] == "invGamma":
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = (
                    torch.exp(-s).mul((p - t) ** 2 + c2 / nt) / 2
                    + (1 / 2 + c1 / nt) * s
                )
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class NSELoss(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask]) > 0:
                p0 = output[:, ii, 0]
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST) + 0.1) ** 2)
                # original NSE
                # temp = SSRes / SST
                losssum = losssum + temp
                nsample = nsample + 1
        return losssum / nsample


class MASELoss(torch.nn.Module):
    def __init__(self, baseline_method):
        """
        This implements the MASE loss function (e.g. MAE_MODEL/MAE_NAIEVE)
        """
        super(MASELoss, self).__init__()
        self.method_dict = {
            "mean": lambda x, y: torch.mean(x, 1).unsqueeze(1).repeat(1, y[1], 1)
        }
        self.baseline_method = self.method_dict[baseline_method]

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, train_data: torch.Tensor, m=1
    ) -> torch.Tensor:
        # Ugh why can't all tensors have batch size... Fixes for modern
        if len(train_data.shape) < 3:
            train_data = train_data.unsqueeze(0)
        if m == 1 and len(target.shape) == 1:
            output = output.unsqueeze(0)
            output = output.unsqueeze(2)
            target = target.unsqueeze(0)
            target = target.unsqueeze(2)
        if len(target.shape) == 2:
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
        result_baseline = self.baseline_method(train_data, output.shape)
        MAE = torch.nn.L1Loss()
        mae2 = MAE(output, target)
        mase4 = MAE(result_baseline, target)
        # Prevent divison by zero/loss exploding
        if mase4 < 0.001:
            mase4 = 0.001
        return mae2 / mase4


class RMSELoss(torch.nn.Module):
    def __init__(self, variance_penalty=0.0):
        """
        Calculate RMSE

        using:
            target -> True y
            output -> Prediction by model
            source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3

        Parameters
        ----------
        variance_penalty
            penalty for big variance; default is 0
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if len(output) <= 1 or self.variance_penalty <= 0.0:
            return torch.sqrt(self.mse(target, output))
        diff = torch.sub(target, output)
        std_dev = torch.std(diff)
        var_penalty = self.variance_penalty * std_dev

        return torch.sqrt(self.mse(target, output)) + var_penalty


class MAPELoss(torch.nn.Module):
    """
    Returns MAPE using:
    target -> True y
    output -> Predtion by model
    """

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if len(output) > 1:
            return torch.mean(
                torch.abs(torch.sub(target, output) / target)
            ) + self.variance_penalty * torch.std(torch.sub(target, output))
        else:
            return torch.mean(torch.abs(torch.sub(target, output) / target))


class MAELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # Create a mask to filter out NaN values
        mask = ~torch.isnan(target)

        # Apply the mask to both target and output
        target = target[mask]
        output = output[mask]

        # Calculate MAE for the non-NaN values
        if self.reduction == "mean":  # Return mean MAe
            return torch.mean(torch.abs(target - output))
        elif self.reduction == "none":
            return torch.abs(target - output)
        else:
            raise ValueError(
                "Reduction must be 'mean' or 'none', got {}".format(self.reduction)
            )


class PenalizedMSELoss(torch.nn.Module):
    """
    Returns MSE using:
    target -> True y
    output -> Predtion by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self.mse(target, output) + self.variance_penalty * torch.std(
            torch.sub(target, output)
        )


class GaussianLoss(torch.nn.Module):
    def __init__(self, mu=0, sigma=0):
        """Compute the negative log likelihood of Gaussian Distribution
        From https://arxiv.org/abs/1907.00235
        """
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        loss = -tdist.Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss) / (loss.size(0) * loss.size(1))


class QuantileLoss(torch.nn.Module):
    """From https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629"""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            mask = ~torch.isnan(target[:, :, i])
            errors = target[:, :, i][mask] - preds[:, :, i][mask]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.cat(losses, dim=0), dim=0)


class NegativeLogLikelihood(torch.nn.Module):
    """
    target -> True y
    output -> predicted distribution
    """

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.distributions, target: torch.Tensor):
        """
        calculates NegativeLogLikelihood
        """
        return -output.log_prob(target).sum()


def l1_regularizer(model, lambda_l1=0.01):
    """
    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
    """
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
        return lossl1


def orth_regularizer(model, lambda_orth=0.01):
    """
    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
    """
    lossorth = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            param_flat = model_param_value.view(model_param_value.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            lossorth += lambda_orth * sym.sum()

        return lossorth


class RmseLoss(torch.nn.Module):
    def __init__(self):
        """
        RMSE loss which could ignore NaN values

        Now we only support 3-d tensor and 1-d tensor
        """
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        if target.dim() == 1:
            mask = target == target
            p = output[mask]
            t = target[mask]
            return torch.sqrt(((p - t) ** 2).mean())
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            p = torch.where(torch.isnan(p), torch.full_like(p, 0), p)
            t = t0[mask]
            t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
            temp = torch.sqrt(((p - t) ** 2).mean())
            loss = loss + temp
        return loss


class MultiOutLoss(torch.nn.Module):
    def __init__(
        self,
        loss_funcs: Union[torch.nn.Module, list],
        data_gap: list = None,
        device: list = None,
        limit_part: list = None,
        item_weight: list = None,
    ):
        """
        Loss function for multiple output

        Parameters
        ----------
        loss_funcs
            The loss functions for each output
        data_gap
            It belongs to the feature dim.
            If 1, then the corresponding value is uniformly-spaced with NaN values filling the gap;
            in addition, the first non-nan value means the aggregated value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's sum, although the next 3 values are nan
            hence the calculation is a little different;
            if 2, the first non-nan value means the average value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's mean value;
            default is [0, 2]
        device
            the number of device: -1 -> "cpu" or "cuda:x" (x is 0, 1 or ...)
        limit_part
            when transfer learning, we may ignore some part;
            the default is None, which means no ignorance;
            other choices are list, such as [0], [0, 1] or [1,2,..];
            0 means the first variable;
            tensor is [seq, time, var] or [time, seq, var]
        item_weight
            use different weight for each item's loss;
            for example, the default values [0.5, 0.5] means 0.5 * loss1 + 0.5 * loss2
        """
        if data_gap is None:
            data_gap = [0, 2]
        if device is None:
            device = [0]
        if item_weight is None:
            item_weight = [0.5, 0.5]
        super(MultiOutLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.data_gap = data_gap
        self.device = get_the_device(device)
        self.limit_part = limit_part
        self.item_weight = item_weight

    def forward(self, output: Tensor, target: Tensor):
        """
        Calculate the sum of losses for different variables

        When there are NaN values in observation, we will perform a "reduce" operation on prediction.
        For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
        then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
        Notice: when "sum", actually final index is not chosen,
        because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
        Just as the rho is 5, the final one is not chosen


        Parameters
        ----------
        output
            the prediction tensor; 3-dims are time sequence, batch and feature, respectively
        target
            the observation tensor

        Returns
        -------
        Tensor
            Whole loss
        """
        n_out = target.shape[-1]
        loss = 0
        for k in range(n_out):
            if self.limit_part is not None and k in self.limit_part:
                continue
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if self.data_gap[k] > 0:
                p, t = deal_gap_data(p0, t0, self.data_gap[k], self.device)
            if type(self.loss_funcs) is list:
                temp = self.item_weight[k] * self.loss_funcs[k](p, t)
            else:
                temp = self.item_weight[k] * self.loss_funcs(p, t)
            # sum of all k-th loss
            if torch.isnan(temp).any():
                continue
            loss = loss + temp
        return loss


# ref: https://github.com/median-research-group/LibMTL
class UncertaintyWeights(torch.nn.Module):
    r"""Uncertainty Weights (UW).

    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
    and implemented by us.

    """

    def __init__(
        self,
        loss_funcs: Union[torch.nn.Module, list],
        data_gap: list = None,
        device: list = None,
        limit_part: list = None,
    ):
        if data_gap is None:
            data_gap = [0, 2]
        if device is None:
            device = [0]
        super(UncertaintyWeights, self).__init__()
        self.loss_funcs = loss_funcs
        self.data_gap = data_gap
        self.device = get_the_device(device)
        self.limit_part = limit_part

    def forward(self, output, target, log_vars):
        """

        Parameters
        ----------
        output
        target
        log_vars
            sigma in uncertainty weighting;
            default is None, meaning we manually set weights for different target's loss;
            more info could be seen in
            https://libmtl.readthedocs.io/en/latest/docs/_autoapi/LibMTL/weighting/index.html#LibMTL.weighting.UW

        Returns
        -------
        torch.Tensor
            multi-task loss by uncertainty weighting method
        """
        n_out = target.shape[-1]
        loss = 0
        for k in range(n_out):
            precision = torch.exp(-log_vars[k])
            if self.limit_part is not None and k in self.limit_part:
                continue
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if self.data_gap[k] > 0:
                p, t = deal_gap_data(p0, t0, self.data_gap[k], self.device)
            if type(self.loss_funcs) is list:
                temp = self.loss_funcs[k](p, t)
            else:
                temp = self.loss_funcs(p, t)
            loss += torch.sum(precision * temp + log_vars[k], -1)
        return loss


# ref: https://openaccess.thecvf.com/content_ECCV_2018/html/Michelle_Guo_Focus_on_the_ECCV_2018_paper.html
class DynamicTaskPrior(torch.nn.Module):
    r"""Dynamic Task Prioritization

    This method is proposed in https://openaccess.thecvf.com/content_ECCV_2018/html/Michelle_Guo_Focus_on_the_ECCV_2018_paper.html
    In contrast to UW and other curriculum learning methods, where easy tasks are prioritized above difficult tasks,
    It shows the importance of prioritizing difficult tasks first.
    It automatically prioritize more difficult tasks by adaptively adjusting the mixing weight of each task's loss.
    Here we choose correlation as KPI. As KPI must be in [0,1], we set (corr+1)/2 as KPI
    """

    def __init__(
        self,
        loss_funcs: Union[torch.nn.Module, list],
        data_gap: list = None,
        device: list = None,
        limit_part: list = None,
        gamma=2,
        alpha=0.5,
    ):
        """

        Parameters
        ----------
        loss_funcs
        data_gap
        device
        limit_part
        gamma
            the example-level focusing parameter
        alpha
            default is 1, which means we only use the newest KPI value
        """
        if data_gap is None:
            data_gap = [0, 2]
        if device is None:
            device = [0]
        super(DynamicTaskPrior, self).__init__()
        self.loss_funcs = loss_funcs
        self.data_gap = data_gap
        self.device = get_the_device(device)
        self.limit_part = limit_part
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target, kpi_last=None):
        """
        Parameters
        ----------
        output
            model's prediction
        target
            observation

        kpi_last
            the KPI value of last iteration; each element for an output
            It use a moving average KPI as the weighting coefficient: KPI_i = alpha * KPI_i + (1-alpha) * KPI_{i-1}

        Returns
        -------
        torch.Tensor
            multi-task loss by Dynamic Task Prioritization method
        """
        n_out = target.shape[-1]
        loss = 0
        kpis = torch.zeros(n_out).to(self.device)
        for k in range(n_out):
            if self.limit_part is not None and k in self.limit_part:
                continue
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if self.data_gap[k] > 0:
                p, t = deal_gap_data(p0, t0, self.data_gap[k], self.device)
            if type(self.loss_funcs) is list:
                temp = self.loss_funcs[k](p, t)
            else:
                temp = self.loss_funcs(p, t)
            # kpi must be in [0, 1], as corr's range is [-1, 1], just trans corr to  (corr+1)/2
            kpi = (torch.corrcoef(torch.stack([p, t], 1).T)[0, 1] + 1) / 2
            if self.alpha < 1:
                assert kpi_last is not None
                kpi = kpi * self.alpha + kpi_last[k] * (1 - self.alpha)
                # if we exclude kpi from the backward, it trans to a normal multi-task model
                # kpi = kpi.detach().clone() * self.alpha + kpi_last[k] * (1 - self.alpha)
            kpis[k] = kpi
            # focal loss
            fl = -((1 - kpi) ** self.gamma) * torch.log(kpi)
            loss += torch.sum(fl * temp, -1)
        # if kpi has grad_fn, backward will repeat. It won't work
        return loss, kpis.detach().clone()


class MultiOutWaterBalanceLoss(torch.nn.Module):
    def __init__(
        self,
        loss_funcs: Union[torch.nn.Module, list],
        data_gap: list = None,
        device: list = None,
        limit_part: list = None,
        item_weight: list = None,
        alpha=0.5,
        beta=0.0,
        wb_loss_func=None,
        means=None,
        stds=None,
    ):
        """
        Loss function for multiple output considering water balance

        loss = alpha * water_balance_loss + (1-alpha) * mtl_loss

        This loss function is only for p, q, et now
        we use the difference between p_obs_mean-q_obs_mean-et_obs_mean and p_pred_mean-q_pred_mean-et_pred_mean as water balance loss
        which is the difference between (q_obs_mean + et_obs_mean) and (q_pred_mean + et_pred_mean)

        Parameters
        ----------
        loss_funcs
            The loss functions for each output
        data_gap
            It belongs to the feature dim.
            If 1, then the corresponding value is uniformly-spaced with NaN values filling the gap;
            in addition, the first non-nan value means the aggregated value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's sum, although the next 3 values are nan
            hence the calculation is a little different;
            if 2, the first non-nan value means the average value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's mean value;
            default is [0, 2]
        device
            the number of device: -1 -> "cpu" or "cuda:x" (x is 0, 1 or ...)
        limit_part
            when transfer learning, we may ignore some part;
            the default is None, which means no ignorance;
            other choices are list, such as [0], [0, 1] or [1,2,..];
            0 means the first variable;
            tensor is [seq, time, var] or [time, seq, var]
        item_weight
            use different weight for each item's loss;
            for example, the default values [0.5, 0.5] means 0.5 * loss1 + 0.5 * loss2
        alpha
            the weight of the water-balance item's loss
        beta
            the weight of real water-balance item's loss, et_mean/p_mean + q_mean/p_mean = 1 can be a loss.
            It is not strictly correct as training batch only have about one year data, but still could be a constraint
        wb_loss_func
            the loss function for water balance item, by default it is None, which means we use function in loss_funcs
        """
        if data_gap is None:
            data_gap = [0, 2]
        if device is None:
            device = [0]
        if item_weight is None:
            item_weight = [0.5, 0.5]
        super(MultiOutWaterBalanceLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.data_gap = data_gap
        self.device = get_the_device(device)
        self.limit_part = limit_part
        self.item_weight = item_weight
        self.alpha = alpha
        self.beta = beta
        self.wb_loss_func = wb_loss_func
        self.means = means
        self.stds = stds

    def forward(self, output: Tensor, target: Tensor):
        """
        Calculate the sum of losses for different variables and water-balance loss

        When there are NaN values in observation, we will perform a "reduce" operation on prediction.
        For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
        then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
        Notice: when "sum", actually final index is not chosen,
        because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
        Just as the rho is 5, the final one is not chosen


        Parameters
        ----------
        output
            the prediction tensor; 3-dims are time sequence, batch and feature, respectively
        target
            the observation tensor

        Returns
        -------
        Tensor
            Whole loss
        """
        n_out = target.shape[-1]
        loss = 0
        p_means = []
        t_means = []
        all_means = self.means
        all_stds = self.stds
        for k in range(n_out):
            if self.limit_part is not None and k in self.limit_part:
                continue
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            # for water balance loss
            if all_means is not None:
                # denormalize for q and et
                p1 = p0 * all_stds[k] + all_means[k]
                t1 = t0 * all_stds[k] + all_means[k]
                p2 = (10**p1 - 0.1) ** 2
                t2 = (10**t1 - 0.1) ** 2
                p_mean = torch.nanmean(p2, dim=0)
                t_mean = torch.nanmean(t2, dim=0)
            else:
                p_mean = torch.nanmean(p0, dim=0)
                t_mean = torch.nanmean(t0, dim=0)
            p_means.append(p_mean)
            t_means.append(t_mean)
            # for mtl normal loss
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if self.data_gap[k] > 0:
                p, t = deal_gap_data(p0, t0, self.data_gap[k], self.device)
            if type(self.loss_funcs) is list:
                temp = self.item_weight[k] * self.loss_funcs[k](p, t)
            else:
                temp = self.item_weight[k] * self.loss_funcs(p, t)
            # sum of all k-th loss
            loss = loss + temp
        # water balance loss
        p_mean_q_plus_et = torch.sum(torch.stack(p_means), dim=0)
        t_mean_q_plus_et = torch.sum(torch.stack(t_means), dim=0)
        wb_ones = torch.ones_like(t_mean_q_plus_et)
        if self.wb_loss_func is None:
            if type(self.loss_funcs) is list:
                # if wb_loss_func is None, we use the first loss function in loss_funcs
                wb_loss = self.loss_funcs[0](p_mean_q_plus_et, t_mean_q_plus_et)
                wb_1loss = self.loss_funcs[0](p_mean_q_plus_et, wb_ones)
            else:
                wb_loss = self.loss_funcs(p_mean_q_plus_et, t_mean_q_plus_et)
                wb_1loss = self.loss_funcs(p_mean_q_plus_et, wb_ones)
        else:
            wb_loss = self.wb_loss_func(p_mean_q_plus_et, t_mean_q_plus_et)
            wb_1loss = self.wb_loss_func(p_mean_q_plus_et, wb_ones)
        return (
            self.alpha * wb_loss
            + (1 - self.alpha - self.beta) * loss
            + self.beta * wb_1loss
        )


class FloodBaseLoss(torch.nn.Module, ABC):
    """
    Abstract base class for flood-related loss functions.

    All flood-related loss functions should inherit from this class.
    The labels tensor is expected to have the flood mask as the last column.
    """

    def __init__(self):
        super(FloodBaseLoss, self).__init__()

    @abstractmethod
    def compute_flood_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the flood-aware loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        pass

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that calls the abstract compute_flood_loss method.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        return self.compute_flood_loss(predictions, targets, flood_mask)


class FloodLoss(FloodBaseLoss):
    def __init__(
        self,
        loss_func: Union[torch.nn.Module, str] = "MSELoss",
        flood_weight: float = 1.0,    
        non_flood_weight: float = 0.5,
        flood_strategy: str = "weight",
        flood_focus_factor: float = 2.0,
        use_score_weighting: bool = False,
        score_weight_min: float = 0.5,
        score_weight_max: float = 2.0,
        score_alpha: float = 2.0,
        device: list = None,
        **kwargs,
    ):
        """
        General flood-aware loss function with configurable base loss and strategy.

        Parameters
        ----------
        loss_func : Union[torch.nn.Module, str]
            Base loss function to use. Can be a PyTorch loss function or string name.
            Supported strings: "MSELoss", "MAELoss", "RMSELoss", "L1Loss"
        flood_weight : float
            Weight multiplier for flood events when using binary weighting (use_score_weighting=False), default is 2.0
        non_flood_weight : float
            Weight multiplier for non-flood events (flood_mask <= 0), default is 1.0
        flood_strategy : str
            Strategy for handling flood events:
            - "weight": Apply higher weights to flood events
            - "focal": Use focal loss approach based on flood event frequency
        flood_focus_factor : float
            Factor for focal loss when using "focal" strategy, default is 2.0
        use_score_weighting : bool
            If True, use data quality scores (from flood_mask values 1-100) as loss weights.
            flood_mask: -1/0 = non-flood (weight=non_flood_weight), 1-100 = flood quality score
            Default is False (use binary weighting: 0/1).
        score_weight_min : float
            Minimum weight for lowest quality score (score=1), default is 0.5
        score_weight_max : float
            Maximum weight for highest quality score (score=100), default is 2.0
        score_alpha : float
            Power factor controlling high-quality data weight proportion.
            weight(score) = base + (score/100)^alpha * (max - base)
            - alpha > 1: emphasize high-quality data (steeper curve)
            - alpha < 1: smoother weight distribution
            - alpha = 1: linear mapping
            Default is 2.0 (quadratic, emphasizing high scores).
        device : list
            Device configuration, default is None (auto-detect)
        """
        super(FloodLoss, self).__init__()
        self.flood_weight = flood_weight
        self.non_flood_weight = non_flood_weight
        self.flood_strategy = flood_strategy
        self.flood_focus_factor = flood_focus_factor
        self.use_score_weighting = use_score_weighting
        self.score_weight_min = score_weight_min
        self.score_weight_max = score_weight_max
        self.score_alpha = score_alpha
        self.device = get_the_device(device if device is not None else [0])

        # Initialize epoch-level statistics for score weighting
        self.reset_statistics()

        # Initialize base loss function
        self.base_loss_func = self._initialize_base_loss(loss_func, kwargs)

    def reset_statistics(self):
        """Reset epoch-level statistics for score weighting."""
        self.epoch_stats = {
            'total_batches': 0,
            'total_flood_events': 0,
            'score_sum': 0.0,
            'score_min': float('inf'),
            'score_max': float('-inf'),
            'all_scores': [],
            'high_quality_count': 0,  # score >= 90
            'perfect_score_count': 0,  # score == 100
            'perfect_score_positions': [],  # List of (batch_idx, timestep, batch_num)
            'weight_sum': 0.0,
            'weight_min': float('inf'),
            'weight_max': float('-inf'),
        }

    def _initialize_base_loss(self, loss_func, kwargs):
        """Initialize base loss function."""
        if isinstance(loss_func, str):
            loss_dict = {
                # NOTE: reduction="none" is important, otherwise the loss will be reduced to a scalar
                "MSELoss": torch.nn.MSELoss(reduction="none"),
                "MAELoss": torch.nn.L1Loss(reduction="none"),
                "L1Loss": torch.nn.L1Loss(reduction="none"),
                "RMSELoss": RMSELoss(),
                "HybridLoss": HybridLoss(
                    kwargs.get("mae_weight", 0.5), reduction="none"
                ),
            }
            if loss_func in loss_dict:
                return loss_dict[loss_func]
            else:
                raise ValueError(f"Unsupported loss function string: {loss_func}")
        else:
            return loss_func

    def compute_flood_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware loss using the specified strategy.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len, 1] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        # Ensure flood_mask has correct shape
        if flood_mask.dim() == 3 and flood_mask.shape[-1] == 1:
            flood_mask = flood_mask.squeeze(-1)  # Remove last dimension if it's 1

        if self.flood_strategy == "weight":
            return self._compute_weighted_loss(predictions, targets, flood_mask)
        elif self.flood_strategy == "focal":
            return self._compute_focal_loss(predictions, targets, flood_mask)
        else:
            raise ValueError(f"Unsupported flood strategy: {self.flood_strategy}")

    def _compute_weighted_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with higher weights for flood events."""
        # Handle warmup_length: predictions may have more timesteps than targets
        # If predictions shape is [batch, seq_pred, features] and targets is [batch, seq_target, features]
        # where seq_pred > seq_target, we need to slice predictions to match targets
        if predictions.shape[1] > targets.shape[1]:
            # Take the last seq_target timesteps from predictions
            seq_diff = predictions.shape[1] - targets.shape[1]
            predictions = predictions[:, seq_diff:, :]

        # Compute base loss
        if isinstance(self.base_loss_func, RMSELoss):
            # Special handling for RMSELoss which doesn't have reduction="none"
            base_loss = torch.pow(predictions - targets, 2)
        else:
            mask = ~torch.isnan(targets)
            predictions = predictions[mask]
            targets = targets[mask]
            flood_mask = flood_mask[
                mask.squeeze(-1)
            ]  # Ensure flood_mask matches predictions/targets
            base_loss = self.base_loss_func(predictions, targets)
        # return base_loss

        # Apply flood weights
        if self.use_score_weighting:
            # Use data quality scores (1-100) as loss weights
            # flood_mask: -1/0 = non-flood, 1-100 = flood quality score
            weights = torch.full_like(
                flood_mask, self.non_flood_weight, dtype=predictions.dtype
            )

            # For flood events (flood_mask >= 1), compute score-based weights
            is_flood = flood_mask >= 1

            # Increment batch counter
            self.epoch_stats['total_batches'] += 1

            if is_flood.any():
                scores = flood_mask[is_flood]
                num_flood_events = is_flood.sum().item()

                # Accumulate statistics
                self.epoch_stats['total_flood_events'] += num_flood_events
                self.epoch_stats['score_sum'] += scores.sum().item()
                self.epoch_stats['score_min'] = min(self.epoch_stats['score_min'], scores.min().item())
                self.epoch_stats['score_max'] = max(self.epoch_stats['score_max'], scores.max().item())

                # Store all scores for median calculation
                self.epoch_stats['all_scores'].extend(scores.cpu().tolist())

                # Count high-quality events (score >= 90)
                high_quality_mask = scores >= 90
                self.epoch_stats['high_quality_count'] += high_quality_mask.sum().item()

                # Find perfect score events (score == 100)
                perfect_mask = scores == 100
                if perfect_mask.any():
                    num_perfect = perfect_mask.sum().item()
                    self.epoch_stats['perfect_score_count'] += num_perfect

                    # Get positions in the original flood_mask tensor
                    flood_indices = torch.nonzero(is_flood, as_tuple=False)
                    perfect_indices = flood_indices[perfect_mask]

                    # Store positions (batch_idx, timestep, current_batch_num)
                    for idx in perfect_indices:
                        if idx.dim() == 1 and len(idx) == 2:
                            batch_idx, timestep = idx[0].item(), idx[1].item()
                            self.epoch_stats['perfect_score_positions'].append(
                                (batch_idx, timestep, self.epoch_stats['total_batches'])
                            )
                        elif idx.dim() == 1 and len(idx) == 1:
                            timestep = idx[0].item()
                            self.epoch_stats['perfect_score_positions'].append(
                                (0, timestep, self.epoch_stats['total_batches'])
                            )

                # Normalize scores to [0, 1]: (score / 100)
                normalized_scores = scores / 100.0
                normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)

                # Apply power transformation with alpha to control high-quality emphasis
                # weight(score) = base + (score/100)^alpha * (max - base)
                score_weights = self.score_weight_min + torch.pow(normalized_scores, self.score_alpha) * (
                    self.score_weight_max - self.score_weight_min
                )

                # Accumulate weight statistics
                self.epoch_stats['weight_sum'] += score_weights.sum().item()
                self.epoch_stats['weight_min'] = min(self.epoch_stats['weight_min'], score_weights.min().item())
                self.epoch_stats['weight_max'] = max(self.epoch_stats['weight_max'], score_weights.max().item())

                weights[is_flood] = score_weights

        else:
            # Increment batch counter for binary weighting mode
            self.epoch_stats['total_batches'] += 1

            # Use original binary weighting (backward compatible)
            # flood_mask >= 1: flood event, otherwise: non-flood
            weights = torch.full_like(
                flood_mask, self.non_flood_weight, dtype=predictions.dtype
            )
            weights[flood_mask >= 1] = self.flood_weight

        # Apply weights to loss
        if base_loss.dim() == 3:  # [batch, seq, features]
            weighted_loss = base_loss * weights.unsqueeze(-1)
        else:  # [batch, seq]
            weighted_loss = base_loss * weights
        valid_mask = ~torch.isnan(weighted_loss)
        weighted_loss = weighted_loss[valid_mask]

        if isinstance(self.base_loss_func, RMSELoss):
            return torch.sqrt(weighted_loss.mean())
        else:
            return torch.mean(weighted_loss)

    def _compute_focal_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss that emphasizes flood events."""
        # Handle warmup_length: predictions may have more timesteps than targets
        if predictions.shape[1] > targets.shape[1]:
            # Take the last seq_target timesteps from predictions
            seq_diff = predictions.shape[1] - targets.shape[1]
            predictions = predictions[:, seq_diff:, :]

        # Compute base loss
        if isinstance(self.base_loss_func, RMSELoss):
            base_loss = torch.pow(predictions - targets, 2)
        else:
            base_loss = self.base_loss_func(predictions, targets)

        # Calculate flood ratio for focal weighting
        flood_ratio = flood_mask.float().mean(dim=1, keepdim=True)  # [batch_size, 1]

        # Focal weight: higher weight when flood events are rare
        focal_weight = (1 - flood_ratio) ** self.flood_focus_factor

        # Separate flood and normal events
        flood_loss = base_loss * flood_mask.unsqueeze(-1).float()
        normal_loss = base_loss * (1 - flood_mask.unsqueeze(-1).float())

        # Apply focal weight to flood events
        weighted_flood_loss = flood_loss * focal_weight.unsqueeze(-1)

        # Combine losses
        total_loss = weighted_flood_loss + normal_loss

        if isinstance(self.base_loss_func, RMSELoss):
            return torch.sqrt(total_loss.mean())
        else:
            return total_loss.mean()


class PESLoss(torch.nn.Module):
    def __init__(self):
        """
        PES Loss: MSE × sigmoid(MSE)

        This loss function applies a sigmoid activation to MSE and then multiplies it with MSE,
        creating a non-linear penalty that increases more gradually for larger errors.
        """
        super(PESLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_value = self.mse(output, target)
        sigmoid_mse = torch.sigmoid(mse_value)
        return mse_value * sigmoid_mse


class HybridLoss(torch.nn.Module):
    def __init__(self, mae_weight: float = 0.5, reduction: str = "mean"):
        """
        Hybrid Loss: PES loss + mae_weight × MAE

        Combines PES loss (MSE × sigmoid(MSE)) with Mean Absolute Error.

        Parameters
        ----------
        mae_weight : float
            Weight for the MAE component, default is 0.5
        reduction : str
            Reduction method for the loss, default is "mean". Can be "mean" or "none".
            If "none", returns the loss without reduction.
        """
        super(HybridLoss, self).__init__()
        self.pes_loss = PESLoss()
        self.mae = MAELoss(reduction=reduction)
        self.mae_weight = mae_weight
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        pes = self.pes_loss(output, target)
        mae = self.mae(output, target)
        if self.reduction == "none":
            return pes + self.mae_weight * mae
        elif self.reduction == "mean":
            loss = pes + self.mae_weight * mae
            valid_mask = ~torch.isnan(loss)
            return torch.mean(loss[valid_mask])
        else:
            raise ValueError(
                f"Unsupported reduction method: {self.reduction}. Use 'mean' or 'none'."
            )


class PeakShapeLoss(FloodBaseLoss):
    """
    Peak Shape Loss for flood events.

    This loss focuses on three key hydrological indicators for flood peak prediction:
    1. Peak Magnitude: Maximum flow value during flood event
    2. Peak Timing: Time of peak occurrence (with asymmetric penalty for late/early)
    3. Flood Volume: Total volume of water during flood event

    Loss = peak_magnitude_weight * L_magnitude +
           peak_timing_weight * L_timing +
           flood_volume_weight * L_volume

    Where:
    - L_magnitude: absolute difference in peak flow values
    - L_timing: time shift with asymmetric penalty (late > early)
    - L_volume: difference in cumulative flood volume
    """

    def __init__(
        self,
        loss_func: Union[torch.nn.Module, str] = "MSELoss",
        peak_magnitude_weight: float = 1.0,
        peak_timing_weight: float = 1.0,
        flood_volume_weight: float = 1.0,
        late_penalty_factor: float = 2.0,
        early_penalty_factor: float = 0.5,
        flood_weight: float = 1.0,
        non_flood_weight: float = 0.5,
        device: list = None,
        **kwargs,
    ):
        """
        Initialize Peak Shape Loss.

        Parameters
        ----------
        loss_func : Union[torch.nn.Module, str]
            Base loss function, default is "MSELoss"
        peak_magnitude_weight : float
            Weight for peak magnitude loss, default is 1.0
        peak_timing_weight : float
            Weight for peak timing loss, default is 1.0
        flood_volume_weight : float
            Weight for flood volume loss, default is 1.0
        late_penalty_factor : float
            Penalty multiplier for late peak predictions, default is 2.0
        early_penalty_factor : float
            Penalty multiplier for early peak predictions, default is 0.5
        flood_weight : float
            Weight multiplier for flood events in global loss, default is 1.0
        non_flood_weight : float
            Weight multiplier for non-flood events, default is 0.5
        device : list
            Device configuration, default is None (auto-detect)
        """
        super(PeakShapeLoss, self).__init__()
        self.peak_magnitude_weight = peak_magnitude_weight
        self.peak_timing_weight = peak_timing_weight
        self.flood_volume_weight = flood_volume_weight
        self.late_penalty_factor = late_penalty_factor
        self.early_penalty_factor = early_penalty_factor
        self.flood_weight = flood_weight
        self.non_flood_weight = non_flood_weight
        self.device = get_the_device(device if device is not None else [0])

        # Initialize base loss function
        self.base_loss_func = self._initialize_base_loss(loss_func, kwargs)

        print(f"[PeakShapeLoss] Initialized with:")
        print(f"  Peak magnitude weight: {peak_magnitude_weight}")
        print(f"  Peak timing weight: {peak_timing_weight} (late penalty: {late_penalty_factor}x, early: {early_penalty_factor}x)")
        print(f"  Flood volume weight: {flood_volume_weight}")
        print(f"  Flood weight: {flood_weight} vs non-flood: {non_flood_weight}")

    def _initialize_base_loss(self, loss_func, kwargs):
        """Initialize base loss function."""
        if isinstance(loss_func, str):
            loss_dict = {
                "MSELoss": torch.nn.MSELoss(reduction="none"),
                "MAELoss": torch.nn.L1Loss(reduction="none"),
                "L1Loss": torch.nn.L1Loss(reduction="none"),
                "RMSELoss": RMSELoss(),
                "HybridLoss": HybridLoss(
                    kwargs.get("mae_weight", 0.5), reduction="none"
                ),
            }
            if loss_func in loss_dict:
                return loss_dict[loss_func]
            else:
                raise ValueError(f"Unsupported loss function string: {loss_func}")
        else:
            return loss_func

    def compute_flood_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute peak-based loss with three hydrological indicators.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        # Ensure flood_mask has correct shape
        if flood_mask.dim() == 3 and flood_mask.shape[-1] == 1:
            flood_mask = flood_mask.squeeze(-1)

        # Handle warmup_length: predictions may have more timesteps than targets
        if predictions.shape[1] > targets.shape[1]:
            seq_diff = predictions.shape[1] - targets.shape[1]
            predictions = predictions[:, seq_diff:, :]
            flood_mask = flood_mask[:, seq_diff:]

        batch_size, seq_len, num_features = targets.shape

        # Extract streamflow (first feature)
        pred_flow = predictions[:, :, 0]  # [batch_size, seq_len]
        target_flow = targets[:, :, 0]  # [batch_size, seq_len]

        magnitude_losses = []
        timing_losses = []
        volume_losses = []

        for b in range(batch_size):
            # Get flood mask for this sample
            flood_mask_b = flood_mask[b]  # [seq_len]

            # Skip if no flood events
            if flood_mask_b.sum() == 0:
                continue

            # Extract flood period data
            pred_flood = pred_flow[b]  # [seq_len]
            target_flood = target_flow[b]  # [seq_len]

            # Skip if contains NaN
            if torch.isnan(target_flood).any() or torch.isnan(pred_flood).any():
                continue

            # === 1. Peak Magnitude Loss ===
            # Find peak values in the entire sequence
            pred_peak_value = torch.max(pred_flood)
            target_peak_value = torch.max(target_flood)

            magnitude_loss = torch.abs(pred_peak_value - target_peak_value)
            magnitude_losses.append(magnitude_loss)

            # === 2. Peak Timing Loss (asymmetric penalty) ===
            # Find peak positions
            pred_peak_idx = torch.argmax(pred_flood)
            target_peak_idx = torch.argmax(target_flood)

            # Time difference (positive = late, negative = early)
            time_diff = pred_peak_idx.float() - target_peak_idx.float()

            # Asymmetric penalty
            if time_diff > 0:
                # Late prediction: higher penalty
                timing_loss = self.late_penalty_factor * torch.abs(time_diff)
            else:
                # Early prediction: lower penalty
                timing_loss = self.early_penalty_factor * torch.abs(time_diff)

            timing_losses.append(timing_loss)

            # === 3. Flood Volume Loss ===
            # Compute cumulative volume during flood period
            # Only sum over flood mask to focus on flood events
            pred_volume = torch.sum(pred_flood * flood_mask_b)
            target_volume = torch.sum(target_flood * flood_mask_b)

            volume_loss = torch.abs(pred_volume - target_volume)
            volume_losses.append(volume_loss)

        # Compute mean losses across batch
        if len(magnitude_losses) == 0:
            # Fallback: if no valid samples, use global MSE
            print("[PeakShapeLoss] Warning: No valid flood samples found, using global MSE")
            mask = ~torch.isnan(targets)
            base_loss = self.base_loss_func(predictions[mask], targets[mask])
            return base_loss.mean()

        magnitude_loss_mean = torch.stack(magnitude_losses).mean()
        timing_loss_mean = torch.stack(timing_losses).mean()
        volume_loss_mean = torch.stack(volume_losses).mean()

        # Combined loss with equal weights (as requested)
        total_loss = (
            self.peak_magnitude_weight * magnitude_loss_mean +
            self.peak_timing_weight * timing_loss_mean +
            self.flood_volume_weight * volume_loss_mean
        )

        return total_loss


class HybridFloodloss(FloodBaseLoss):
    def __init__(self, mae_weight=0.5):
        """
        Hybrid Flood Loss: PES loss + mae_weight × MAE with flood weighting

        Combines PES loss (MSE × sigmoid(MSE)) with Mean Absolute Error,
        applying flood weighting to the loss.

        The difference from FloodLoss is that this class filter flood events first then calculate loss,
        because Hybrid does sigmoid on MSE, when the non-flood-weight is 0, which means we do not want to
        calculate loss on non-flood events, so we need to filter them out first.

        Parameters
        ----------
        mae_weight : float
            Weight for the MAE component, default is 0.5
        flood_weight : float
            Weight multiplier for flood events, default is 2.0
        """
        super(HybridFloodloss, self).__init__()
        self.mae_weight = mae_weight

    def compute_flood_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware loss using the specified strategy.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len, 1] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        boolean_mask = flood_mask.to(torch.bool)
        predictions = predictions[boolean_mask]
        targets = targets[boolean_mask]

        base_loss_func = HybridLoss(self.mae_weight)
        return base_loss_func(predictions, targets)
