"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-05-17 11:29:13
LastEditors: Wenyu Ouyang
Description: Loss functions
FilePath: \torchhydro\torchhydro\models\crits.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union
import torch
from torch import distributions as tdist, Tensor
from torchhydro.models.model_utils import get_the_device


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
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # Create a mask to filter out NaN values
        mask = ~torch.isnan(target)

        # Apply the mask to both target and output
        target = target[mask]
        output = output[mask]

        # Calculate MAE for the non-NaN values
        return torch.mean(torch.abs(target - output))


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
    It automatically prioritize more difficult tasks by adaptively adjusting the mixing weight of each task’s loss.
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
