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

    adf check

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
        self.t_critical_table = None  # critical table of t check statistic   Applied Time Series Analysis（4th edition） Yan Wang p228
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
        correlation coefficient, R^2.
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
        m: int = None,
    ):
        """
        autocorrelation function, acf.
        随机过程  8.2 模型的识别  p190  式（8.11）AR(p)自相关函数
        Parameters
        ----------
        x
        m

        Returns
        -------

        """
        if m is None:
            n_x = len(x)
            if n_x < 50:  # hydrology statistics p318
                m = int(n_x / 1.3) - 1
            else:
                m = int(n_x / 1.3)
                if m < 10:
                    m = n_x - 10
        p = list(range(0, m+1))
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
        partial autocorrelation function, pacf.
        随机过程  8.2 模型的识别  p198  式（8.25）AR(p)偏相关函数
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
                m = int(n_x / 1.3) - 1
            else:
                m = int(n_x / 1.3)
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
        r_k_ = r_k[1:]
        r_k_ = np.transpose(r_k_)
        try:
            R_1 = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        pacf_k = np.matmul(R_1, r_k_)
        # add 0 degree
        pacf = np.zeros(k+1)
        pacf[0] = 1
        pacf[1:] = pacf_k[:]

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

        if b_s_a:  # Econometric chapter 3
            n_y = len(Y)
            e = Y - y_
            e = np.absolute(e)
            e_t = np.transpose(e)
            var_e = np.matmul(e_t, e)
            var_e = var_e / n_y
            std_e = np.sqrt(var_e)
            x = A[:, 0]             # Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge chapter 3
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
        dx(t) = rho*x(t-1) + b_1*dx(t-1) + b_2*dx(t-2) + ... + b_(p-1)*dx(t-(p-1)) + e(t)   Applied Time Series Analysis p228 formula(6.7)
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
        rho, std_rho = self.adf_least_squares_estimation(x, p)
        t = rho/std_rho

        return t


    def generate_t_critical_table(
        self,
        case,
        n_sample,
        significance_level,
    ):
        """
        Time Series Analysis  James D.Hamilton  p882 table B.6
        Returns
        -------

        """
        # table B.6
        case_names = ["case 1", "case 2", "case 4"]
        T = [25, 50, 100, 250, 500, "∞"]
        p = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99]
        case_1 = [
            [-2.66, -2.26, -1.95, -1.6, 0.92, 1.33, 1.7, 2.16],
            [-2.62, -2.25, -1.95, -1.61, 0.91, 1.31, 1.66, 2.08],
            [-2.6, -2.24, -1.95, -1.61, 0.90, 1.29, 1.64, 2.03],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.29, 1.63, 2.01],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00],
        ]
        case_2 = [
            [-3.75, -3.33, -3.00, -2.63, -0.37, 0.00, 0.34, 0.72],
            [-3.58, -3.22, -2.93, -2.60, -0.40, -0.03, 0.29, 0.66],
            [-3.51, -3.17, -2.89, -2.58, -0.42, -0.05, 0.26, 0.63],
            [-3.46, -3.14, -2.88, -2.57, -0.42, -0.06, 0.24, 0.62],
            [-3.44, -3.13, -2.87, -2.57, -0.43, -0.07, 0.24, 0.61],
            [-3.43, -3.12, -2.86, -2.57, -0.44, -0.07, 0.23, 0.60],
        ]
        case_4 = [
            [-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15],
            [-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24],
            [-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28],
            [-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31],
            [-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32],
            [-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33],
        ]
        data = np.array([case_1, case_2, case_4])
        # sample_size_T = T
        # columns = p
        # index = pd.MultiIndex.from_product([case_names, sample_size_T])
        # table_B_6 = pd.DataFrame(data=data, index=index, columns=columns)
        # table_B_6.columns = p
        # table_B_6.index.name = "sample size T"
        # table_B_6.set_index("sample size T", inplace=True)
        case_i = case_names.index(case)
        sl_i = p.index(significance_level)
        for i in range(1, len(T)-1):
            if n_sample <= T[0]:
                n_sample = T[0]
                break
            if n_sample > T[-2]:
                n_sample = T[-1]
                break
            if (n_sample > T[i-1]) and (n_sample <= T[i]):
                n_sample = T[i]
                break
        sample_i = T.index(n_sample)
        t_critical = data[case_i, sample_i, sl_i]

        return t_critical

    def get_t_critical(
        self,
        case,
        n_sample,
        significance_level,
    ):
        """

        Returns
        -------

        """
        t_critical = self.generate_t_critical_table(case, n_sample, significance_level)
        return t_critical

    def adf_test(
        self,
        x,
        p,
        case,
        significance_level,
    ):
        """
        Augmented Dickey-Fuller Test
        ADF test.  unit root.
        Time Series Analysis  James D.Hamilton p625
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
        # assumption
        H0 = True
        H1 = False
        # t_statistic
        t_statistic = self.t_statistic(x, p)
        # t_critical
        n_x = len(x)
        t_critical = self.get_t_critical(case, n_x, significance_level)
        # test
        if t_statistic < t_critical:
            b_stability = H0
        else:
            b_stability = H1

        return b_stability

    def integrate_one_degree(
        self,
        x,
    ):
        """
        one degree integration model
        Parameters
        ----------
        x

        Returns
        -------

        """
        n_x = len(x)
        dx = np.zeros(n_x)
        tx = np.zeros(n_x)  # trend
        tx[0] = x[0]
        for i in range(0, n_x):
            if i == 0:
                dx_i = x[i] - 0
                tx_i = 0
                dx[i] = dx_i
                tx[i] = tx_i
            else:
                dx_i = x[i] - x[i-1]
                tx_i = x[i-1]
                dx[i] = dx_i
                tx[i] = tx_i

        return dx, tx

    def integrate_d_degree(
        self,
        x,
        d,
    ):
        """
        d degree integration model
        Parameters
        ----------
        x: time series
        d: degree of integrate

        Returns
        -------

        """
        n_x = len(x)
        tx = np.zeros(n_x)  # trend
        dx_i0 = x
        dx_i = None
        for i in range(d):
            dx_i, tx_i = self.integrate_one_degree(dx_i0)
            dx_i0 = dx_i[:]
            tx = tx + tx_i[:]
        dx = dx_i[:]

        return dx, tx

    def arma(
        self,
        x,
        e,
        phi,
        theta,
        p: int = 0,
        q: int = 0,
    ):
        """
        arma model, single step.
        Applied Time Series Analysis（4th edition） Yan Wang  p110
        p, q
        Parameters
        ----------
        x: time series
        e: time series
        phi: parameters of AR(p) model
        theat: parameters of MA(q) model
        p: degree of autoregression model
        q: degree of moving average model

        Returns
        -------

        """
        y_t = 0
        # ar
        ar = 0
        for i in range(p):
            ar_i = phi[i] * x[p-1-i]
            ar = ar + ar_i
        # ma
        ma = e[-1]
        for i in range(q):
            ma_i = - theta[i] * e[q-2-i]
            ma = ma + ma_i
        # arma
        y_t = y_t + ar + ma   #

        return y_t

    def arima(
        self,
        x,
        e,
        phi,
        theta,
        p: int = 0,
        d: int = 0,
        q: int = 0,
    ):
        """
        ARIMA model
        Parameters
        ----------
        x: time series
        e: time series
        p: degree of autoregression model
        d: degree of integration model
        q: degree of moving average model

        Returns
        -------

        """
        n_x = len(x)
        # integrate
        if d > 0:
            dx, tx = self.integrate_d_degree(x, d)
        else:
            dx = x  # integration
            tx = [0]*n_x  # trend
        # center
        mean_dx = np.mean(dx)
        dx = dx - mean_dx
        std_x = np.std(x)  # todo:
        # arma
        y_t = [0]*n_x
        start = max(p, q)
        y_t[:start] = dx[:start]
        for i in range(start, n_x):
            y_t[i] = self.arma(dx[i-p:i], e[i-q:i+1], phi, theta, p, q)
        # arma + i
        y_t = y_t + mean_dx
        y_t = y_t + tx

        return y_t

    def arma_least_squares_estimation(
        self,
        x,
        e,
        p: int = 0,
        q: int = 0,
    ):
        """
        least squares estimation for parameters of arma model.
        Parameters
        ----------
        x: time series
        e: time series
        p: degree of autoregression model
        q: degree of moving average model

        Returns
        -------

        """
        n_x = len(x)

        # construct matrix
        start = max(p, q)
        xf = x[start:]
        ef = e[start:]
        xf = np.array(xf) - np.array(ef)
        xf = np.transpose(xf)
        xp = []
        for i in range(start, n_x):
            ar_i = x[i-p:i]
            ar_i.reverse()
            ma_i = e[i-q:i]
            ma_i.reverse()
            xp_i = ar_i + ma_i
            xp.append(xp_i)

        # matrix operations, calculate the coefficient matrix.
        a, R_2 = self.ordinary_least_squares(xp, xf)

        phi = a[:p]
        theat = -a[p:]

        return phi, theat, R_2


    def x_residual(
        self,
        x,
        e,
        p,
        d,
        q,
    ):
        """
        residual of ARIMA model.
        Parameters
        ----------
        x

        Returns
        -------

        """
        phi, theta, R_2 = self.arma_least_squares_estimation(x, e, p, q)
        y_t = self.arima(x, e, phi, theta, q, d, p)

        x_residual = x - y_t

        return x_residual

    def LB_statistic(
        self,
        residual,
        m,
    ):
        """
        LB statistic of ARIMA model.
        Parameters
        ----------
        residual

        Returns
        -------

        """
        n_residual = len(residual)
        acf = self.autocorrelation_function(residual, m)
        acf = np.power(acf, 2)
        for i in range(m):
            acf[i] = acf[i] / (n_residual - (i + 1))
        # LB(Ljung-Box) statistic
        LB = n_residual * (n_residual + 2) * np.sum(acf)

        return LB

    def get_chi_critical(
        self,
        t,
        significance_level,
        n_sample,
    ):
        """

        Parameters
        ----------
        t
        significance_level
        n_sample

        Returns
        -------

        """
        T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        p = [0.01, 0.05, 0.10]
        data = [
            [-2.66, -2.26, -1.95, -1.6, 0.92, 1.33, 1.7, 2.16],
            [-2.62, -2.25, -1.95, -1.61, 0.91, 1.31, 1.66, 2.08],
            [-2.6, -2.24, -1.95, -1.61, 0.90, 1.29, 1.64, 2.03],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.29, 1.63, 2.01],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00],
            [-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00],
            [-3.75, -3.33, -3.00, -2.63, -0.37, 0.00, 0.34, 0.72],
            [-3.58, -3.22, -2.93, -2.60, -0.40, -0.03, 0.29, 0.66],
            [-3.51, -3.17, -2.89, -2.58, -0.42, -0.05, 0.26, 0.63],
            [-3.46, -3.14, -2.88, -2.57, -0.42, -0.06, 0.24, 0.62],
            [-3.44, -3.13, -2.87, -2.57, -0.43, -0.07, 0.24, 0.61],
            [-3.43, -3.12, -2.86, -2.57, -0.44, -0.07, 0.23, 0.60],
            [-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15],
            [-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24],
            [-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28],
            [-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31],
            [-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32],
            [-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33],
        ]
        data = np.array(data)
        t_i = T.index(t)
        sl_i = p.index(significance_level)
        for i in range(1, len(T)-1):
            if n_sample <= T[0]:
                n_sample = T[0]
                break
            if n_sample > T[-2]:
                n_sample = T[-1]
                break
            if (n_sample > T[i-1]) and (n_sample <= T[i]):
                n_sample = T[i]
                break
        sample_i = T.index(n_sample)
        t_critical = data[t_i, sample_i, sl_i]

        return t_critical

    def test_arima(
        self,
        residual,
        m,
    ):
        """
        significance test of ARIMA model.  chi-square test
        Parameters
        ----------
        residual
        m

        Returns
        -------

        """
        n_residual = len(residual)
        LB = self.LB_statistic(residual, m)
        t = m
        significance_level = 0.05
        n_sample = n_residual
        chi_critical = self.get_chi_critical(t, significance_level, n_sample)

        # assumption
        H0 = True
        H1 = False

        if LB < chi_critical:
            b_ = H0
        else:
            b_ = H1

        return b_





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
        t = 10
        a = 0
        b = 0
        r = 0
        d_xt_ = 0
        e_t = 0
        d_xt = a + b * t + r * x[t-1] + d_xt_ + e_t

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
