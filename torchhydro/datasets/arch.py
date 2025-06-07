import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict
from typing import Optional

from hydroutils import hydro_time

from torchhydro.datasets.data_sources import data_sources_dict


class Arch(object):
    """
    Autoregressive Conditional Heteroscedasticity model, ARCH.  Based on ARIMA.
    autoregression integrated moving average, ARIMA.
    time series imputation
    σ(t)^2 = α0 + α1*a(t-1)^2 + α2*a(t-2)^2 + ... + αp*a(t-p)^2

    AR，auto-regression model.
    MA, moving average model.
    I, integrate model.
    ARCH, autoregressive conditional heteroscedasticity model.

    distribution of series -> check relationship -> mean value function

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
        self.mean = np.mean(x)
        self.p = None  # degree of autoregression
        self.q = None  # degree of moving average
        self.d = None  # degree of integrate
        self.p0 = 0.05  # significance level of p_check
        self.t_critical_table = None  # critical table of t check statistic   应用时间序列分析（第四版） 王燕 p228    todo:
        self.fi = None
        self.sigma = None

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
        degree: int = 1,
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
                rho_k = self.rho(x[i+k], x[i])

    def cov(
        self,
        x,
        y: Optional = None,
        mean_x: Optional = None,
        mean_y: Optional = None,
    ):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        if mean_x is None:
            mean_x = np.mean(x)
        if y is None:
            y = x
            mean_y = mean_x
        if mean_y is None:
            mean_y = np.mean(y)
        # n_x = x.shape[0]
        n_x = len(x)
        cov = 0
        for i in range(n_x):
            x_i = x[i] - mean_x
            y_i = y[i] - mean_y
            xy_i = x_i * y_i
            cov = cov + xy_i
        cov = cov / n_x
        return cov

    def correlation_coefficient(
        self,
        x,
        y,
        mean_x: Optional = None,
        mean_y: Optional = None,
    ):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        if ((mean_x is not None) and (mean_y is None)) or ((mean_x is None) and (mean_y is not None)):
            raise ValueError("Error: Please set mean_x and mean_y at the same time or do not set both.")
        cov_xy = self.cov(x, y, mean_x=mean_x, mean_y=mean_y)
        std_x = pow(self.cov(x, mean_x=mean_x), 0.5)
        std_y = pow(self.cov(y, mean_x=mean_y), 0.5)
        rho_xy = cov_xy / (std_x * std_y)
        return rho_xy

    def autocorrelation_coefficient(
        self,
        x,
        p,
    ):
        """
        unbiased acf.
        Parameters
        ----------
        x
        p

        Returns
        -------

        """
        # n_x = x.shape[0]
        n_x = len(x)
        if p > n_x:
            raise ValueError("Error: p could not be larger than the length of x.")
        mean_x = np.mean(x)
        if p == 0:
            x_ = x[:]
        else:
            x_ = x[:-p]
        y_ = x[p:]
        rho_p_xx = self.correlation_coefficient(x_, y_, mean_x, mean_x)

        return rho_p_xx

    def autocorrelation_function(
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
        n_x = len(x)
        if n_x < 50:  # hydrology statistics p318
            m = int(n_x / 4) - 1
        else:
            m = int(n_x / 4)
            if m < 10:
                m = n_x - 10
        p = list(range(0, m))
        acf = [0]*len(p)
        for i in range(len(p)):
            acf[i] = self.autocorrelation_coefficient(x, p[i])
        return acf

    def partial_autocorrelation_function(
        self,
        x,
        k: int = None,
    ):
        """
        # todo:
        Parameters
        ----------
        x
        k

        Returns
        -------

        """
        if k is None:
            n_x = len(x)
            if n_x < 50:  # hydrology statistics p318
                m = int(n_x / 4) - 1
            else:
                m = int(n_x / 4)
                if m < 10:
                    m = n_x - 10
            k = m  # the max degree of pacf
        r_k = self.autocorrelation_function(x)
        # R
        R = np.zeros((k, k))
        for i in range(k):
            kk = 0
            for j in range(k):
                if i < j:
                    R[i, j] = r_k[j]
                    kk = kk + 1
                else:
                    R[i, j] = r_k[i-j]
                    kk = kk + 1
        r_k_ = np.transpose(r_k)
        try:
            R_1 = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        pacf_k = np.matmul(R_1, r_k_)

        return pacf_k, R

    def p_check(
        self,
        x,
    ):
        """
        p check for stability of series. Z-check.
        Parameters
        ----------
        x

        Returns
        -------

        """
        x_mean = np.mean(x)
        u0 = 0.5
        sigma = np.std(x)
        n_x = len(x)
        z = (x_mean - u0) / (sigma / pow(n_x, 0.5))
        p_check = np.abs(z) < self.p0
        return p_check

    def adf_check(
        self,
        x,
    ):
        """
        Augmented Dickey-Fuller Tested
        ADF check.  unit root.
        assumption：
            H0: true.
            H1: false.
        (10%, 5%, 1%) -> (90%, 95%, 99%)
        degree of intergre: 0, 1, 2
        t:
        p:
        critical value(1%, 5%, 10%)
        Parameters
        ----------
        x

        Returns
        -------

        """
        t = 10
        a = 0
        b = 0
        r = 0
        d_xt_ = 0
        e_t = 0
        d_xt = a + b * t + r * x[t-1] + d_xt_ + e_t

    def ar_least_squares_estimation(
        self,
        x,
        p: int = 2,
    ):
        """
        least squares estimation of autoregressive.
        minimize the square summation of residual error -> parameters of autoregressive -> estimate value
        numerical analysis page 67-71.
        Parameters
        ----------
        x: time series
        p: the degree of autoregressive.
        Returns
        -------

        """
        n_x = len(x)

        # construct matrix
        xf = x[p:]
        xf = np.transpose(xf)
        xp = []
        for i in range(n_x-2):
            xp_i = x[i:i+p]
            xp_i.reverse()
            xp.append(xp_i)

        # matrix operations, calculate the coefficient matrix.
        a, R_2 = self.ordinary_least_squares(xp, xf)

        return a, R_2

    def ordinary_least_squares(
        self,
        A,
        Y,
        b_s_a: Optional = False,
    ):
        """
        ordinary least squares, ols.
        minimize the square summation of residual error -> parameters of model.
        a = (A' * A)_1 * A' * Y
        Parameters
        ----------
        A: matrix
        Y: matrix
        b_s_a: bool,

        Returns
        -------

        """
        # matrix operations, calculate the coefficient matrix.
        At = np.transpose(A)
        B = np.matmul(At, A)
        try:
            B_1 = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        a = np.matmul(B_1, At)
        a = np.matmul(a, Y)  # parameters

        # R_2, coefficient of determination
        y_ = np.matmul(A, a)
        R = self.correlation_coefficient(Y, y_)
        R_2 = pow(R, 2)

        if b_s_a:  # 计量经济学 第三章
            n_y = len(Y)
            e = Y - y_
            e = np.absolute(e)
            e_t = np.transpose(e)
            var_e = np.matmul(e_t, e)
            var_e = var_e / n_y
            std_e = np.sqrt(var_e)
            x = A[:, 0]             # 计量经济学导论现代观点 第三章
            var_x = self.cov(x)
            std_x = np.sqrt(var_x)
            x_A = A[:, 1:]
            _, x_R_2 = self.ordinary_least_squares(x_A, x)
            se_a0 = std_e / (np.sqrt(n_y) * std_x * np.sqrt(1 - x_R_2))  # standard error of a[0]

            return a, R_2, se_a0

        return a, R_2

    def adf_least_squares_estimation(
        self,
        x,
        p,
    ):
        """
        least squares estimation of Augmented Dickey-Fuller Tested.
        minimize the square summation of residual error -> parameters of adf model.
        dx(t) = rho*x(t-1) + b_1*dx(t-1) + b_2*dx(t-2) + ... + b_(p-1)*dx(t-(p-1)) + e(t)
        Parameters
        ----------
        x: time series
        p: the degree of adf.
        Returns
        -------

        """
        n_x = len(x)

        # construct matrix
        dx = [0]*(n_x-1)
        for i in range(1, n_x):
            dx[i-1] = x[i] - x[i-1]
        dxf = dx[p-1:]
        dxf = np.transpose(dxf)
        xp = [0]*(n_x-p)
        xp_i = [0]*p
        for i in range(p-1, n_x-1):
            xp_i[0] = x[i]
            xp_ii = dx[i-(p-1):i]
            xp_ii.reverse()
            xp_i[1:] = xp_ii[:]
            xp[i-(p-1)] = xp_i[:]
        xp = np.array(xp)

        # matrix operations, calculate the coefficient matrix.
        a, R_2, s_a0 = self.ordinary_least_squares(xp, dxf, True)

        # result
        rho = a[0]
        s_rho = s_a0

        # return a, R_2, s_a0
        return rho, s_rho

    def t_statistic(
        self,
        x,
        p,
    ):
        """
        the t statistic.
        Parameters
        ----------
        x: time series.
        p: the degree of adf.

        Returns
        -------

        """
        rho, std_rho = self.adf_least_squares_estimation(x,p)
        t = rho/std_rho

        return t

    def rho_standard_error(
        self,
    ):
        """

        Returns
        -------

        """


    def cal_acf(self, x):
        """acf, auto-correlation coefficient """
        ps_x = pd.Series(x)
        corr = ps_x.autocorr()
        return corr

    def mean_value_function(
        self,
        x,
        e
    ):
        """
        mean value function
        Parameters
        ----------
        x: observe value, series.
        e: error item
        Returns
        -------
        y_t: the observe value of time-step t.
        """
        p = x.shape[0]  # degree
        # mean = np.mean(x)
        fi = [0]*p  # coefficient of regression
        y_t = 0
        for i in range(p):
            y_i = fi[i] * x[i]
            y_t = y_t + y_i
        y_t = y_t + self.mean + e
        return y_t

    def std_function(
        self,
        w,
        e,
        std,
    ):
        """std function, garch."""
        q = e.shape[0]
        p = std.shape[0]
        a = [0]*q
        b = [0]*p
        sum_e = 0
        sum_std = 0
        for i in range(q):
            e_i = a[i] * pow(e[i], 2)
            sum_e = sum_e + e_i
        for i in range(p):
            std_i = b[i] * pow(std[i], 2)
            sum_std = sum_std + std_i
        std_t = w + sum_e + sum_std
        return std_t

    def cal_pacf(
        self,
        x,
    ):
        """
        pacf, partial auto-correlation function. a series consisted by partial auto-correlation coefficient.
        Parameters
        ----------
        x

        Returns
        -------

        """
        ps_x = pd.Series(x)
