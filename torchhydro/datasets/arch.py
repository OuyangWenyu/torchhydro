

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict

from hydroutils import hydro_time

from torchhydro.datasets.data_sources import data_sources_dict


class Arch(object):
    """
    Autoregressive Conditional Heteroscedasticity model, ARCH.
    time series imputation

    """
    def __init__(
        self,
        x,
    ):
        """ """
        self.original_dataset = None
        self.statistic_dict = None
        self.deficient_dataset = None
        self.degree_m = 0  # arch degree
        self.r = None  # coefficient of arch model
        self.x = x
        self.e = None  # error
        self.length = len(x)

    def cal_statistics(self):
        """calculate statistics"""


    def cal_7_stat_inds(self, x):
        """
        Calculate seven statistics indices of a series: point number, mean value, standard deviation, min,
        percentile 25 50 75 and max.

        Parameters
        ----------
        x
            input data

        Returns
        -------
        list
            [mean, std, min, p25, p50, p75, max]
        """
        num_point = x.shape[0]
        mean = np.mean(x).astype(float)
        std = np.std(x).astype(float)
        min_ = np.min(x).astype(float)
        p25 = np.percentile(x, 25).astype(float)
        p50 = np.percentile(x, 50).astype(float)
        p75 = np.percentile(x, 75).astype(float)
        max_ = np.max(x).astype(float)

        if std < 0.001:
            std = 1
        return [num_point, mean, std, min_, p25, p50, p75, max_]

    def deficient_dataset(self):
        """generate deficient dataset."""


    def analysis_dataset(self):
        """ analysis dataset."""

    def cal_mse(self, x, y):
        """calculate mean squared error."""
        return np.mean((y - x) ** 2)

    def cal_spearman(self, x, y):
        """calculate spearman correlation."""

    def fluctuate_rate(self, r, y):
        """ calculate fluctuation rate."""
        yy = np.power(y, 2)
        yy[0] = np.power(y[0], 0)
        std = r * yy

        return std

    def imputation(self, std, e):
        """imputation"""
        y = std * e

        return y

    def arch_function(
        self,
        e: list,
        degree: int=1,
    ):
        """
        error arch
        Parameters
        ----------
        e
        degree

        Returns
        -------

        """

        a0 = 0.5
        a1 = 0.2
        e_c = [0]*self.length
        e_c[0] = e[0]
        for i in range(1,self.length):
            e_c[i] = a0 + a1 * e[i] * e[i]

        return e_c

    def mle(
        self,
        x,
        y,
        e,
    ):
        """
        max likelihood evaluation.
        Parameters
        ----------
        x
        y
        e

        Returns
        -------

        """
        b0 = 0.5
        b1 = 0.2
        wt = 3
        pi = np.pi
        sum_lene = 0
        sum_se = 0
        for i in range(self.length):
            p_e = pow(e[i], 2)
            ln_i = np.log(p_e)
            sum_lene = sum_lene + ln_i
            p_y = pow(y[i]-b0-b1*x[i], 2)
            se_i = p_y / p_e
            sum_se = sum_se + se_i

        # mle
        lnLt = -0.5*np.log(2*pi) - sum_lene / (2 * self.length) - sum_se / (2 * self.length)

        return lnLt

    def garch_function(
        self,
        std,
        e,
    ):
        """
        garch function, single step.
        Parameters
        ----------
        std
        e

        Returns
        -------

        """
        a0 = 0.5
        a1 = 0.2
        b1 = 0.3
        std = a0 + a1*pow(e, 2) + b1*pow(std, 2)
        return std

    def mle_garch(
        self,
        std,
        e,
    ):
        """

        Parameters
        ----------
        std
        e

        Returns
        -------

        """
        pi = np.pi
        sum_ = 0
        for i in range(self.length):
            p_std = pow(std[i], 2)
            logstd_i = np.log(p_std)
            p_e = pow(e[i], 2)
            se_i = p_e / p_std
            sum_ = sum_ + logstd_i + se_i
        lt = -0.5*self.length*np.log(2*pi) - 0.5*(sum_)
        return lt

    def rho(
        self,
        et,
        ek,
    ):
        """
        relation rate
        Parameters
        ----------
        et

        Returns
        -------

        """
        p_e = pow(et, 2)
        p_e_t = pow(ek, 2)
        var_e = np.var(p_e)
        cov_ek = np.cov(p_e, p_e_t)
        rho_k = cov_ek / var_e

        return rho_k

    def condition_check(
        self,
        x,
    ):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        k = 7
        for i in range(k,self.length):
            rho_k = self.rho(x[i+k], x[i])
            if rho_k > 0.0:
