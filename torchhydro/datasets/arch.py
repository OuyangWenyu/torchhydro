

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

    ):
        """ """
        self.original_dataset = None
        self.statistic_dict = None
        self.deficient_dataset = None
        self.degree_m = 0  # arch degree
        self.r = None  # coefficient of arch model

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

    def evaluate_para(self):
        """ evaluate the parameters of arch."""
        