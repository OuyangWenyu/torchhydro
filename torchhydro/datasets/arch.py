"""
nothing but English.
"""
import mpmath.libmp
import numpy as np
from typing import Optional


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

    adf check

    AIC degree

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
        # self.x = x
        self.e = None  # error
        # self.length = len(x)
        # self.mean = np.mean(x)
        self.p = None  # degree of autoregression
        self.q = None  # degree of moving average
        self.d = None  # degree of integrate
        self.p0 = 0.05  # significance level of p_check
        self.t_critical_table = None  # critical table of t check statistic   Applied Time Series Analysis（4th edition） Yan Wang p228
        self.fi = None
        self.sigma = None


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

    def cov(
        self,
        x,
        y: Optional = None,
        mean_x: Optional = None,
        mean_y: Optional = None,
    ):
        """
        covariance and variance
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
        n_x = len(x)
        cov = 0
        for i in range(n_x):
            x_i = x[i] - mean_x
            y_i = y[i] - mean_y
            xy_i = x_i * y_i
            cov = cov + xy_i
        try:
            cov = cov / n_x
        except ZeroDivisionError:
            raise ZeroDivisionError('division by zero')
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
        var_x = self.cov(x, mean_x=mean_x)
        std_x = pow(var_x, 0.5)
        var_y = self.cov(y, mean_x=mean_y)
        std_y = pow(var_y, 0.5)
        rho_xy = cov_xy / (std_x * std_y)
        return rho_xy

    def autocorrelation_coefficient(
        self,
        x,
        p,
        var_x: Optional = None,
    ):
        """
        biased acf.  Applied Time Series Analysis（4th edition） Yan Wang  p66
        Parameters
        ----------
        x: time series
        p: degree
        var_x: variance

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
            x_ = x[:n_x-p]
        y_ = x[p:]
        cov_xy = self.cov(x_, y_, mean_x, mean_x)
        if var_x is None:
            var_x = self.cov(x=x, mean_x=mean_x)
        rho_p_xx = (cov_xy / var_x) * ((n_x - p) / n_x)

        return rho_p_xx

    def autocorrelation_function(
        self,
        x,
        m: int = None,
    ):
        """
        autocorrelation function, acf.
        随机过程  8.2 模型的识别  p190  式（8.11）AR(p)自相关函数
        Time Series Analysis with Applications in R (second edition) Jonathan D.Cryer, Kung-Sil Chan p77
        Parameters
        ----------
        x: time series
        m: degree

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
        mean_x = np.mean(x)
        var_x = self.cov(x, mean_x=mean_x)
        acf = [0]*(m+1)
        for i in range(m+1):
            acf[i] = self.autocorrelation_coefficient(x, p[i], var_x)
        return acf

    def partial_autocorrelation_coefficient(
        self,
        r,
    ):
        """
        partial auto-correlation coefficient.
        随机过程  8.2 模型的识别  p198  式（8.25）AR(p)偏相关函数
        Time series Analysis: Forecasting and Control, 5th Edition, George E.P.Box etc. p51
        Parameters
        ----------
        r

        Returns
        -------

        """
        k = len(r) - 1
        # D
        D = np.zeros((k, k))
        for i in range(k):
            kk = 0
            for j in range(k):
                if i < j:
                    D[i, j] = r[j]
                    kk = kk + 1
                else:
                    D[i, j] = r[i-j]
                    kk = kk + 1
        # Dk
        r_k_ = np.array(r[1:])
        Dk = D[:, :k-1]
        Dk = np.column_stack((Dk, r_k_))
        D = np.linalg.det(D)
        Dk = np.linalg.det(Dk)
        pacc_k = Dk / D

        return pacc_k

    def partial_autocorrelation_function(
        self,
        x,
        k: int = None,
    ):
        """
        partial auto-correlation function, pacf.
        Time series Analysis: Forecasting and Control, 5th Edition, George E.P.Box etc. p52
        Parameters
        ----------
        x: time series.
        k: degree of partial auto-correlation function.

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

        pacf = np.zeros(k + 1)
        pacf[0] = 1
        for i in range(1, k+1):
            r_i = r_k[:i+1]
            pacf[i] = self.partial_autocorrelation_coefficient(r_i)

        return pacf

    def ar_least_squares_estimation(
        self,
        x,
        p: int = 2,
        b_constant: bool = False,
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
        for i in range(n_x-p):
            xp_i = x[i:i+p]
            xp_i.reverse()
            xp.append(xp_i)

        if b_constant:
            xp_constant = np.ones(n_x-p)
            xp = np.array(xp)
            xp = np.insert(xp, xp_constant, 0, axis=1)

        # matrix operations, calculate the coefficient matrix.
        a, R_2 = self.ordinary_least_squares(xp, xf)

        return a, R_2

    def ordinary_least_squares(
        self,
        A,
        Y,
        b_s_a: Optional = False,
        b_se_beta: Optional = False,
    ):
        """
        ordinary least squares, ols.
        minimize the square summation of residual error -> parameters of model.
        numerical analysis page 67-71.
        a = (A' * A)_1 * A' * Y
        Parameters
        ----------
        A: matrix
        Y: matrix
        b_s_a: bool,
        b_se_beta

        Returns
        -------

        """
        # matrix operations, calculate the coefficient matrix.
        A = np.array(A)
        At = np.transpose(A)
        B = np.matmul(At, A)
        try:
            B_1 = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        a = np.matmul(B_1, At)
        try:
            a = np.matmul(a, Y)  # parameters
        except ValueError:
            raise ValueError("matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 22 is different from 23)")

        # R_2, coefficient of determination
        y_ = np.matmul(A, a)
        R = self.correlation_coefficient(Y, y_)
        R_2 = pow(R, 2)

        b_s = (b_s_a or b_se_beta)
        if b_s:
            n_y = len(Y)
            e = Y - y_
            e = np.absolute(e)
            e_t = np.transpose(e)
            var_e = np.matmul(e_t, e)
            var_e = var_e / n_y
            std_e = np.sqrt(var_e)
            if b_s_a:  # Econometric chapter 3
                x = A[:, 0]             # Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge chapter 3 P79-80
                var_x = self.cov(x)
                std_x = np.sqrt(var_x)
                x_A = A[:, 1:]
                _, x_R_2 = self.ordinary_least_squares(x_A, x)
                se_a0 = std_e / (np.sqrt(n_y) * std_x * np.sqrt(1 - x_R_2))  # standard error of a[0]
                return a, R_2, se_a0
            if b_se_beta:
                n_A = A.shape[1]
                se_beta = [0]*n_A
                for i in range(n_A):
                    x_i = A[:, i]       # Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge chapter 3 P79-80
                    var_xi = self.cov(x_i)
                    std_xi = np.sqrt(var_xi)
                    xi_A = np.delete(A, i, axis=1)
                    _, xi_R_2 = self.ordinary_least_squares(xi_A, x_i)
                    se_beta[i] = std_e / (np.sqrt(n_y) * std_xi * np.sqrt(1 - xi_R_2))  # standard error of a[0]
                return a, R_2, se_beta

        return a, R_2

    def ar_maximum_likelihood_estimation(
        self,
    ):
        """
        Time Series Analysis  James D.Hamilton  chapter 5 p134
        Returns
        -------

        """



    def adf_least_squares_estimation(
        self,
        x,
        p,
    ):
        """
        least squares estimation of Augmented Dickey-Fuller Tested.
        minimize the square summation of residual error -> parameters of adf model.
        Applied Time Series Analysis p228 formula(6.7)
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

    def tau_statistic(
        self,
        x,
        p,
    ):
        """
        the tau statistic.
        Parameters
        ----------
        x: time series.
        p: the degree of adf.

        Returns
        -------

        """
        rho, std_rho = self.adf_least_squares_estimation(x, p)
        tau = rho/std_rho

        return tau

    def get_tau_critical(
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

        # locate
        case_i = case_names.index(case)
        if significance_level in p:
            sl_i = p.index(significance_level)
        else:
            raise ValueError('Significance level = ' + str(significance_level) + 'not in Significance level array.')
        sample_i = None
        sample_within = None
        if (n_sample >= T[0]) and (n_sample <= T[-2]):
            if n_sample in T:
                sample_i = T.index(n_sample)
            else:
                n_T = len(T)
                for i in range(n_T-1):
                    if (n_sample > T[i]) and (n_sample < T[i+1]):
                        sample_i = [i, i+1]
                        sample_within = [T[i], T[i+1]]
                        break
        elif n_sample > T[-2]:
            sample_i = -1
        else:
            raise ValueError('sample volume n_sample = ' + str(n_sample) + 'out of range.')

        # querying
        if type(sample_i) is list:
            critical_0 = data[case_i, sample_i[0], sl_i]
            critical_1 = data[case_i, sample_i[1], sl_i]
            tau_critical = (critical_1 - critical_0) / (sample_within[1] - sample_within[0]) * (n_sample - sample_within[0]) + critical_0  # linear interpolation
        else:
            tau_critical = data[case_i, sample_i, sl_i]

        return tau_critical

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
        # tau_statistic
        tau_statistic = self.tau_statistic(x, p)
        # t_critical
        n_x = len(x)
        t_critical = self.get_tau_critical(case, n_x, significance_level)
        # test
        if tau_statistic < t_critical:
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
        x: time series

        Returns
        -------

        """
        n_x = len(x)
        dx = np.zeros(n_x-1)
        tx = np.zeros(n_x)  # trend
        tx[0] = x[0]
        for i in range(0, n_x):
            if i == 0:
                tx_i = x[i]
                tx[i] = tx_i
            else:
                dx_i = x[i] - x[i-1]
                tx_i = x[i-1]
                dx[i-1] = dx_i
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
            tx[i:] = tx[i:] + tx_i[:]
        dx = dx_i[:]

        return dx, tx

    def integration(
        self,
        x,
        d,
    ):
        """

        Parameters
        ----------
        x: time series
        d: degree of integrate

        Returns
        -------

        """
        n_x = len(x)
        # integrate
        if d > 0:
            dx, tx = self.integrate_d_degree(x, d)
            dx = dx.tolist()
        else:
            dx = x  # integration
            tx = [0]*n_x  # trend
        # center
        # mean_dx = np.mean(dx)
        # dx_c = (dx - mean_dx).tolist()

        # return dx_c, mean_dx, tx

        return dx, tx

    def ar_one_step(
        self,
        x,
        phi,
    ):
        """
        AR(p) model
        Parameters
        ----------
        x: time series
        phi: parameters of AR(p) model
        p: degree of autoregression model

        Returns
        -------

        """
        # ar
        xt = np.transpose(x)
        ar = np.matmul(phi, xt)
        return ar

    def ma_one_step(
        self,
        e,
        theta,
    ):
        """
        MA(q) model
        Parameters
        ----------
        e: time series
        theta: parameters of MA(p) model
        q: degree of moving average model

        Returns
        -------

        """
        # MA
        et = e[1:]
        et = np.transpose(et)
        ma0 = e[0]
        ma = np.matmul(theta, et)
        ma = ma0 - ma
        return ma

    def arma(
        self,
        x,
        e: Optional = None,
        b_constant: bool = False,
        phi: Optional = None,
        theta: Optional = None,
        p: int = 0,
        q: int = 0,
    ):
        """
        ARMA model, multi-step.
        Applied Time Series Analysis（4th edition） Yan Wang  p110
        p, q
        Parameters
        ----------
        x: time series
        e: time series
        b_constant: constant item or not in AR(P) model.
        phi: parameters of AR(p) model
        theat: parameters of MA(q) model
        p: degree of autoregression model
        q: degree of moving average model

        Returns
        -------

        """
        # check parameters
        if p > 0:
            if (x is None) or (phi is None):
                raise ValueError("x and phi must be provided both.")
        if q > 0:
            if (e is None) or (theta is None):
                raise ValueError("e and theta must be provided both.")

        n_x = len(x)
        y = np.zeros(n_x)
        start = max(p, q)
        y[:start] = x[:start]
        if p > 0:
            ar = np.zeros(n_x-start)
            for i in range(start, n_x):
                x_i = x[i-p:i]
                if b_constant:
                    x_i.append(1)  # constant item
                x_i.reverse()
                ar[i-start] = self.ar_one_step(x_i, phi)
            y[start:] = ar[:]
        if q > 0:
            # y[:start] = y[:start]  # + e[:start]
            ma = np.zeros(n_x - start)
            for i in range(start, n_x):
                e_i = e[i-q:i+1]
                e_i.reverse()
                ma[i-start] = self.ma_one_step(e_i, theta)
            y[start:] = y[start:] + ma[:]

        return y

    def Q_statistic(
        self,
        x,
        m: int = None,
    ):
        """
        Q statistic
        Applied Time Series Analysis（4th edition） Yan Wang p30
        Parameters
        ----------
        x: time series
        m: degree

        Returns
        -------

        """
        n_x = len(x)
        acf_ = self.autocorrelation_function(x, m)
        acf = acf_[1:]
        acf = np.power(acf, 2)
        Q = n_x * np.sum(acf)
        return Q

    def white_noise_test(
        self,
        x,
        m,
        significance_level,
    ):
        """
        white noise test
        Applied Time Series Analysis（4th edition） Yan Wang p30    概率论与数理统计 p201

        Parameters
        ----------
        x: time series
        m: free degree
        significance_level: significance level
        Returns
        -------

        """
        n_x = len(x)
        if n_x > 100:  # todo:
            Q_statistic = self.Q_statistic(x, m)
        else:
            Q_statistic, acf = self.LB_statistic(x, m)
        # significance_level_ = 1 - significance_level
        Q_critical = self.get_chi_critical(m, significance_level)    # 概率论与数理统计 p201

        # assumption
        H0 = True
        H1 = False

        if Q_statistic < Q_critical:   # Applied Time Series Analysis（4th edition） Yan Wang p30
            b_white_noise = H0
        else:
            b_white_noise = H1

        return b_white_noise

    def white_noise_test_degree(
        self,
        x,
        m_list,
        significance_level,
    ):
        """
        white noise test with multiple free degree.
        Parameters
        ----------
        x
        m_list: free degree list
        significance_level

        Returns
        -------

        """
        n_degree = len(m_list)



    def arima(
        self,
        x,
        e: Optional = None,
        phi: Optional = None,
        theta: Optional = None,
        p: int = 0,
        d: int = 0,
        q: int = 0,
        # dx_c: Optional = None,
        # mean_dx: Optional = None,
        dx: Optional = None,
        tx: Optional = None,
    ):
        """
        ARIMA model
        Parameters
        ----------
        x: time series
        e: time series
        phi: parameters of AR(p) model
        theta: parameters of MA(q) model
        p: degree of autoregression model
        d: degree of integration model
        q: degree of moving average model
        dx_c: the integration result of time series x by integration model with d degree.
        mean_dx: the mean of integration result dx.
        tx: the trend item of integration result of time series x by integration model with d degree.

        Returns
        -------

        """
        # parameter check
        if p > 0:
            if (x is None) or (phi is None):
                raise ValueError("x and phi must be provided both.")
        if q > 0:
            if (x is None) or (e is None) or (theta is None):
                raise ValueError("x, e and theta must be provided all.")

        n_x = len(x)
        # integrate
        # if (dx_c is None) or (mean_dx is None) or (tx is None):
        if (dx is None) or (tx is None):
            if d > 0:
                # dx_c, mean_dx, tx = self.integration(x, d)
                dx, tx = self.integration(x, d)
            else:
                # dx_c = x
                # mean_dx = 0
                dx = x
                tx = np.zeros(n_x)

        # arma
        y = np.zeros(n_x)
        if q > 0:
            e_ = e[d:]
            if p > 0:
                # y_ = self.arma(x=dx_c, e=e_, phi=phi, theta=theta, p=p, q=q)  # arma
                y_ = self.arma(x=dx, e=e_, phi=phi, theta=theta, p=p, q=q)
            else:
                # y_ = self.arma(x=dx_c, e=e_, theta=theta, q=q)  # ma
                y_ = self.arma(x=dx, e=e_, theta=theta, q=q)  # ma
        elif p > 0:
            # y_ = self.arma(x=dx_c, phi=phi, p=p)  # ar
            y_ = self.arma(x=dx, phi=phi, p=p)
        else:
            # return dx_c, mean_dx, tx  # i
            return dx, tx

        # arma + i
        # y[d:] = y[d:] + mean_dx
        y[d:] = y[d:] + y_[:]
        y = y + tx

        return y

    def arma_least_squares_estimation(
        self,
        x: Optional = None,
        e: Optional = None,
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
        # parameter check
        if p > 0:
            if x is None:
                raise ValueError("x must be provided.")
        if q > 0:
            if e is None:
                raise ValueError("e must be provided.")
        if (x is None) and (e is None):
            raise ValueError("Either x or e must be provided.")

        n_x = len(x)

        # construct matrix
        start = max(p, q)
        xf = x[start:]
        if q > 0:
            ef = e[start:]
            xf = np.array(xf) - np.array(ef)
        xf = np.transpose(xf)
        xp = []
        for i in range(start, n_x):
            xp_i = []
            if p > 0:
                ar_i = x[i-p:i]
                ar_i.reverse()
                xp_i = ar_i
            if q > 0:
                ma_i = e[i-q:i]
                ma_i.reverse()
                xp_i = xp_i + ma_i
            xp.append(xp_i[:])

        # matrix operations, calculate the coefficient matrix.
        # a, R_2, a_diagonal = self.ordinary_least_squares(xp, xf, b_a_diagonal=True)
        a, R_2, se_beta = self.ordinary_least_squares(xp, xf, b_se_beta=True)

        # allot parameters
        phi = []
        theta = []
        if p > 0:
            phi = a[:p]
            if q > 0:
                theta = -a[p:]
        else:
            theta = -a[:]

        return phi, theta, R_2, se_beta

    def x_residual(
        self,
        x,
        e: Optional = None,
        p: int = 0,
        d: int = 0,
        q: int = 0,
    ):
        """
        residual of ARIMA model.
        Parameters
        ----------
        x: time series
        e: white noise series
        p: degree of autoregression model
        d: degree of integration model
        q: degree of moving average model

        Returns
        -------

        """
        if d > 0:
            # dx, mean_dx, tx = self.integration(x, d)
            dx, tx = self.integration(x, d)
            if q > 0:
                if e is None:
                    raise ValueError("e must be provided.")
                phi, theta, R_2, se_beta = self.arma_least_squares_estimation(x=dx, e=e[d:], p=p, q=q)
            else:
                phi, theta, R_2, se_beta = self.arma_least_squares_estimation(x=dx, p=p)
            # y_t = self.arima(x, e, phi, theta, p, d, q, dx, mean_dx, tx)
            y_t = self.arima(x, e, phi, theta, p, d, q, dx, tx)
        else:
            if q > 0:
                if e is None:
                    raise ValueError("e must be provided.")
                phi, theta, R_2, se_beta = self.arma_least_squares_estimation(x=x, e=e, p=p, q=q)
            else:
                phi, theta, R_2, se_beta = self.arma_least_squares_estimation(x=x, p=p)
            y_t = self.arima(x, e, phi, theta, p, d, q)

        x_residual = np.array(x) - np.array(y_t)

        return x_residual, y_t, R_2, phi, theta, se_beta

    def LB_statistic(
        self,
        residual,
        m: Optional = None,
    ):
        """
        LB statistic of ARIMA model.
        Applied Time Series Analysis（4th edition） Yan Wang p77
        Time Series Analysis with Applications in R (second edition) Jonathan D.Cryer, Kung-Sil Chan p131
        Parameters
        ----------
        residual
        m

        Returns
        -------

        """
        n_residual = len(residual)
        acf0 = self.autocorrelation_function(residual, m)
        acf = acf0[1:]  # remove p=0
        acf_ = np.power(acf, 2)
        for i in range(m):
            acf_i = acf_[i] / (n_residual - (i + 1))
            acf_[i] = acf_i
        sum_acf = np.sum(acf_)
        # LB(Ljung-Box) statistic
        LB = n_residual * (n_residual + 2) * sum_acf

        return LB, acf

    def get_chi_critical(
        self,
        m,
        significance_level,
    ):
        """
        Time Series Analysis  James D.Hamilton  p872 table B.2
        Parameters
        ----------
        m
        significance_level
        n_sample

        Returns
        -------

        """
        # table
        m_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 40, 50, 60, 70, 80, 90, 100,]
        p = [0.995, 0.99, 0.975, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
        data = [
            [0.00004, 0.0002, 0.001, 0.004, 0.016, 0.102, 0.455, 1.32, 2.71, 3.84, 5.02, 6.63, 7.88, 10.8],
            [0.01, 0.02, 0.051, 0.103, 0.211, 0.575, 1.39, 2.77, 4.61, 5.99, 7.38, 9.21, 10.6, 13.8],
            [0.072, 0.115, 0.216, 0.352, 0.584, 1.21, 2.37, 4.11, 6.25, 7.81, 9.35, 11.3, 12.8, 16.3],
            [0.207, 0.297, 0.484, 0.711, 1.06, 1.92, 3.36, 5.39, 7.78, 9.49, 11.1, 13.3, 14.9, 18.5],
            [0.412, 0.554, 0.831, 1.15, 1.61, 2.67, 4.35, 6.63, 9.24, 11.1, 12.8, 15.1, 16.7, 20.5],
            [0.676, 0.872, 1.24, 1.64, 2.2, 3.45, 5.35, 7.84, 10.6, 12.6, 14.4, 16.8, 18.5, 22.5],
            [0.989, 1.24, 1.69, 2.17, 2.83, 4.25, 6.35, 9.04, 12, 14.1, 16, 18.5, 20.3, 24.3],
            [1.34, 1.65, 2.18, 2.73, 3.49, 5.07, 7.34, 10.2, 13.4, 15.5, 17.5, 20.1, 22, 26.1],
            [1.73, 2.09, 2.7, 3.33, 4.17, 5.9, 8.34, 11.4, 14.7, 16.9, 19, 21.7, 23.6, 27.9],
            [2.16, 2.56, 3.25, 3.94, 4.87, 6.74, 9.34, 12.5, 16, 18.3, 20.5, 23.2, 25.2, 29.6],
            [2.6, 3.05, 3.82, 4.57, 5.58, 7.58, 10.3, 13.7, 17.3, 19.7, 21.9, 24.7, 26.8, 31.3],
            [3.07, 3.57, 4.4, 5.23, 6.3, 8.44, 11.3, 14.8, 18.5, 21, 23.3, 26.2, 28.3, 32.9],
            [3.57, 4.11, 5.01, 5.89, 7.04, 9.3, 12.3, 16, 19.8, 22.4, 24.7, 27.7, 29.8, 34.5],
            [4.07, 4.66, 5.63, 6.57, 7.79, 10.2, 13.3, 17.1, 21.1, 23.7, 26.1, 29.1, 31.3, 36.1],
            [4.6, 5.23, 6.26, 7.26, 8.55, 11, 14.3, 18.2, 22.3, 25, 27.5, 30.6, 32.8, 37.7],
            [5.14, 5.81, 6.91, 7.96, 9.31, 11.9, 15.3, 19.4, 23.5, 26.3, 28.8, 32, 34.3, 39.3],
            [5.7, 6.41, 7.56, 8.67, 10.1, 12.8, 16.3, 20.5, 24.8, 27.6, 30.2, 33.4, 35.7, 40.8],
            [6.26, 7.01, 8.23, 9.39, 10.9, 13.7, 17.3, 21.6, 26, 28.9, 31.5, 34.8, 37.2, 42.3],
            [6.84, 7.63, 8.91, 10.1, 11.7, 14.6, 18.3, 22.7, 27.2, 30.1, 32.9, 36.2, 38.6, 43.8],
            [7.43, 8.26, 9.59, 10.9, 12.4, 15.5, 19.3, 23.8, 28.4, 31.4, 34.2, 37.6, 40, 45.3],
            [8.03, 8.9, 10.3, 11.6, 13.2, 16.3, 20.3, 24.9, 29.6, 32.7, 35.5, 38.9, 41.4, 46.8],
            [8.64, 9.54, 11, 12.3, 14, 17.2, 21.3, 26, 30.8, 33.9, 36.8, 40.3, 42.8, 48.3],
            [9.26, 10.2, 11.7, 13.1, 14.8, 18.1, 22.3, 27.1, 32, 35.2, 38.1, 41.6, 44.2, 49.7],
            [9.89, 10.9, 12.4, 13.8, 15.7, 19, 23.3, 28.2, 33.2, 36.4, 39.4, 43, 45.6, 51.2],
            [10.5, 11.5, 13.1, 14.6, 16.5, 19.9, 24.3, 29.3, 34.4, 37.7, 40.6, 44.3, 46.9, 52.6],
            [11.2, 12.2, 13.8, 15.4, 17.3, 20.8, 25.3, 30.4, 35.6, 38.9, 41.9, 45.6, 48.3, 54.1],
            [11.8, 12.9, 14.6, 16.2, 18.1, 21.7, 26.3, 31.5, 36.7, 40.1, 43.2, 47, 49.6, 55.5],
            [12.5, 13.6, 15.3, 16.9, 18.9, 22.7, 27.3, 32.6, 37.9, 41.3, 44.5, 48.3, 51, 56.9],
            [13.1, 14.3, 16, 17.7, 19.8, 23.6, 28.3, 33.7, 39.1, 42.6, 45.7, 49.6, 52.3, 58.3],
            [13.8, 15, 16.8, 18.5, 20.6, 24.5, 29.3, 34.8, 40.3, 43.8, 47, 50.9, 53.7, 59.7],
            [20.7, 22.2, 24.4, 26.5, 29.1, 33.7, 39.3, 45.6, 51.8, 55.8, 59.3, 63.7, 66.8, 73.4],
            [28, 29.7, 32.4, 34.8, 37.7, 42.9, 49.3, 56.3, 63.2, 67.5, 71.4, 76.2, 79.5, 86.7],
            [35.5, 37.5, 40.5, 43.2, 46.5, 52.3, 59.3, 67, 74.4, 79.1, 83.3, 88.4, 92, 99.6],
            [43.3, 45.4, 48.8, 51.7, 55.3, 61.7, 69.3, 77.6, 85.5, 90.5, 95, 100, 104, 112],
            [51.2, 53.5, 57.2, 60.4, 64.3, 71.1, 79.3, 88.1, 96.6, 102, 107, 112, 116, 125],
            [59.2, 61.8, 65.6, 69.1, 73.3, 80.6, 89.3, 98.6, 108, 113, 118, 124, 128, 137],
            [67.3, 70.1, 74.2, 77.9, 82.4, 90.1, 99.3, 109, 118, 124, 130, 136, 140, 149],
        ]
        data = np.array(data)

        # locate
        m_i = None
        m_within = None
        if (m >= m_index[0]) and (m <= m_index[-1]):
            if m in m_index:
                m_i = m_index.index(m)
            else:
                m_30 = m_index.index(30)
                n_m_index = len(m_index)
                for i in range(m_30, n_m_index-1):
                    if (m > m_index[i]) and (m < m_index[i+1]):
                        m_i = [i, i+1]
                        m_within = [m_index[i], m_index[i+1]]
                        break
        else:
            raise ValueError('Index m = ' + str(m) + 'out of range.')

        if significance_level in p:
            sl_i = p.index(significance_level)
        else:
            raise ValueError('Significance level = ' + str(significance_level) + 'not in Significance level array.')

        # querying
        if type(m_i) is list:
            critical_0 = data[m_i[0], sl_i]
            critical_1 = data[m_i[1], sl_i]
            chi_critical = (critical_1 - critical_0) / (m_within[1] - m_within[0]) * (m - m_within[0]) + critical_0  # linear interpolation
        else:
            chi_critical = data[m_i, sl_i]

        return chi_critical

    def test_arima(
        self,
        residual,
        m,
        significance_level,
    ):
        """
        significance test of ARIMA model.  chi-square test

        Parameters
        ----------
        residual
        m
        significance_level

        Returns
        -------

        """
        n_residula = len(residual)
        if m > n_residula:  # todo:
            raise ValueError("degree m = " + str(m) + " out of series length.")
        LB, acf = self.LB_statistic(residual, m)
        # significance_level_ = 1 - significance_level
        chi_critical = self.get_chi_critical(m, significance_level)       # 概率论与数理统计 p201

        # assumption
        H0 = True
        H1 = False

        if LB < chi_critical:    #Applied Time Series Analysis（4th edition） Yan Wang p30
            b_significant = H0
        else:
            b_significant = H1

        return b_significant

    def t_statistic(
        self,
        residual,
        phi,
        theta,
        # a_diagonal,
        se_beta,
    ):
        """
        significance test for parameters of ARIMA model.    t test
        Applied Time Series Analysis（4th edition） Yan Wang p78
        Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge  p96-109
        Parameters
        ----------
        phi
        theta
        a_diagonal
        Returns
        -------

        """
        beta = phi + theta
        # m = len(beta)
        # n_residual = len(residual)
        # t = np.sqrt(n_residual-m)
        # residual_2 = np.power(residual, 2)
        # sum_residual_2 = np.sum(residual_2)
        # t = t / np.sqrt(sum_residual_2)
        # a_ = np.sqrt(a_diagonal)
        # t = t / a_
        # t_statistic = t * beta
        t_statistic = np.array(beta) / np.array(se_beta)

        return t_statistic

    def get_t_critical(
        self,
        m,
        significance_level,
    ):
        """
        Time Series Analysis  James D.Hamilton  p880 table B.4
        Returns
        -------

        """
        # table
        m_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 40, 60, 120, "∞"]
        p = [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
        data = [
            [1, 3.078, 6.314, 12.706, 31.821, 63.657, 318.31],
            [0.816, 1.886, 2.92, 4.303, 6.965, 9.925, 22.326],
            [0.765, 1.638, 2.353, 3.182, 4.541, 5.841, 10.213],
            [0.741, 1.533, 2.132, 2.776, 3.747, 4.604, 7.173],
            [0.727, 1.476, 2.015, 2.571, 3.365, 4.032, 5.893],
            [0.718, 1.44, 1.943, 2.447, 3.143, 3.707, 5.208],
            [0.711, 1.415, 1.895, 2.365, 2.998, 3.499, 4.785],
            [0.706, 1.397, 1.86, 2.306, 2.896, 3.355, 4.501],
            [0.703, 1.383, 1.833, 2.262, 2.821, 3.25, 4.297],
            [0.7, 1.372, 1.812, 2.228, 2.764, 3.169, 4.144],
            [0.697, 1.363, 1.796, 2.201, 2.718, 3.106, 4.025],
            [0.695, 1.356, 1.782, 2.179, 2.681, 3.055, 3.93],
            [0.694, 1.35, 1.771, 2.16, 2.65, 3.012, 3.852],
            [0.692, 1.345, 1.761, 2.145, 2.624, 2.977, 3.787],
            [0.691, 1.341, 1.753, 2.131, 2.602, 2.947, 3.733],
            [0.69, 1.337, 1.746, 2.12, 2.583, 2.921, 3.686],
            [0.689, 1.333, 1.74, 2.11, 2.567, 2.898, 3.646],
            [0.688, 1.33, 1.734, 2.101, 2.552, 2.878, 3.61],
            [0.688, 1.328, 1.729, 2.093, 2.539, 2.861, 3.579],
            [0.687, 1.325, 1.725, 2.086, 2.528, 2.845, 3.552],
            [0.686, 1.323, 1.721, 2.08, 2.518, 2.831, 3.527],
            [0.686, 1.321, 1.717, 2.074, 2.508, 2.819, 3.505],
            [0.685, 1.319, 1.714, 2.069, 2.5, 2.807, 3.485],
            [0.685, 1.318, 1.711, 2.064, 2.492, 2.797, 3.467],
            [0.684, 1.316, 1.708, 2.06, 2.485, 2.787, 3.45],
            [0.684, 1.315, 1.706, 2.056, 2.479, 2.779, 3.435],
            [0.684, 1.314, 1.703, 2.052, 2.473, 2.771, 3.421],
            [0.683, 1.313, 1.701, 2.048, 2.467, 2.763, 3.408],
            [0.683, 1.311, 1.699, 2.045, 2.462, 2.756, 3.396],
            [0.683, 1.31, 1.697, 2.042, 2.457, 2.75, 3.385],
            [0.681, 1.303, 1.684, 2.021, 2.423, 2.704, 3.307],
            [0.679, 1.296, 1.671, 2, 2.39, 2.66, 3.232],
            [0.677, 1.289, 1.658, 1.98, 2.358, 2.617, 3.16],
            [0.674, 1.282, 1.645, 1.96, 2.326, 2.576, 3.09],
        ]
        data = np.array(data)

        # locate
        m_i = None
        m_within = None
        n_m_index = len(m_index)
        if (m >= m_index[0]) and (m <= m_index[-2]):
            if m in m_index:
                m_i = m_index.index(m)
            else:
                m_30 = m_index.index(30)
                for i in range(m_30, n_m_index - 2):
                    if (m > m_index[i]) and (m < m_index[i + 1]):
                        m_i = [i, i + 1]
                        m_within = [m_index[i], m_index[i + 1]]
                        break
        elif m > m_index[-2]:
            m_i = n_m_index - 1
        else:
            raise ValueError('Index m = ' + str(m) + 'out of range.')

        if significance_level in p:
            sl_i = p.index(significance_level)
        else:
            raise ValueError('Significance level = ' + str(significance_level) + 'not in Significance level array.')

        # querying
        if type(m_i) is list:
            critical_0 = data[m_i[0], sl_i]
            critical_1 = data[m_i[1], sl_i]
            t_critical = (critical_1 - critical_0) / (m_within[1] - m_within[0]) * (m - m_within[0]) + critical_0  # linear interpolation
        else:
            t_critical = data[m_i, sl_i]

        return t_critical

    def test_parameters(
        self,
        residual,
        phi,
        theta,
        se_beta,
        m,
        significance_level,
    ):
        """
        significance test for parameters of ARIMA model.    t test
        Applied Time Series Analysis（4th edition） Yan Wang p78
        Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge  p96-109
        Parameters
        ----------
        phi
        theta
        se_beta

        Returns
        -------

        """
        n_resudual = len(residual)
        t_statistic = self.t_statistic(residual, phi, theta, se_beta)
        t_statistic = np.absolute(t_statistic)
        # significance_level_ = 1 - significance_level
        t_critical = self.get_t_critical(n_resudual-m, significance_level)

        # assumption
        H0 = False
        H1 = True

        b_significant = []
        for i in range(len(t_statistic)):
            if t_statistic[i] >= t_critical:
                b_i = H1
            else:
                b_i = H0
            b_significant.append(b_i)

        return b_significant

    def arma_one_step(
        self,
        x: Optional[list] = None,
        e: Optional[list] = None,
        phi: Optional[list] = None,
        theta: Optional[list] = None,
    ):
        """

        Parameters
        ----------
        x
        e
        phi
        theta

        Returns
        -------

        """
        if (phi is None) and (theta is None):
            raise ValueError('Either phi or theta must be provided.')

        x_infer = 0
        if phi is not None:
            ar = self.ar_one_step(x, phi)
            x_infer = ar
        if theta is not None:
            ma = self.ma_one_step(e, theta)
            x_infer = x_infer + ma

        return x_infer

    def reverse_integrate_one_degree_one_step(
        self,
        x_infer_t,
        x_t_,
    ):
        """

        Parameters
        ----------
        x_infer_t: infer item of arma model at t.
        x_t_: trend of time series at t, = x(t-1).

        Returns
        -------

        """
        x_infer = x_infer_t + x_t_

        return x_infer

    def reverse_integrate_d_degree_one_step(
        self,
        d,
        x_infer_t,
        x_t,
    ):
        """

        Parameters
        ----------
        d
        x_infer_t
        x_t_

        Returns
        -------

        """
        n_x_t = len(x_t)
        if n_x_t != d:
            raise ValueError("The length of x_t_ need to be equal to d.")

        tx = []  # trend
        tx.append(x_t)
        for i in range(d-1):
            tx_i = [0]*(d-i-1)
            x_t_i = tx[i]
            for j in range(d-1-i):
                tx_i[j] = x_t_i[j+1] - x_t_i[j]
            tx.append(tx_i[:])

        x_infer = x_infer_t
        for i in range(d-1, -1, -1):
                x_infer = self.reverse_integrate_one_degree_one_step(x_infer, tx[i][-1])

        return x_infer

    def arima_infer(
        self,
        t: int,
        b_constant: bool = False,
        x: Optional = None,
        e: Optional = None,
        phi: Optional[list] = None,
        theta: Optional[list] = None,
        p: Optional[int] = 0,
        d: Optional[int] = 0,
        q: Optional[int] = 0,
    ):
        """

        Parameters
        ----------
        t
        x
        e
        phi
        theta
        p
        d
        q
        Returns
        -------

        """
        if ((d > 0) or (p > 0)) and (x is None):
            raise ValueError("x need be provided when either d or p > 0.")
        if q > 0 and (e is None):
            raise ValueError("e need be provided when q > 0.")
        n_x = 0
        if x is not None:
            n_x = len(x)
        pd = p + d
        if n_x < pd:
            raise ValueError("the length of x need be larger then p+d.")

        dx = []  # trend
        x_t = x[-pd:]
        dx.append(x_t)
        xx = dx[:]
        d_x = [dx[-1]]
        if d > 0:
            for i in range(pd + d - 1):
                tx_i = [0] * (pd + d - i - 1)
                x_t_i = dx[i]
                for j in range(pd + d - 1 - i):
                    tx_i[j] = x_t_i[j + 1] - x_t_i[j]
                dx.append(tx_i[:])
                d_x.append(tx_i[-1])
            xx = dx[-1][:]

        x_infer = [0]*t
        if q > 0:
            for i in range(t):
                e_i = e[i - q:i]
                e_i.reverse()
                if p > 0:
                    if i == 0:
                        x_i = xx[:]
                    elif (i > 1) and (i < pd):
                        x_i = xx[-(pd-i):]
                        x_i = x_i + x_infer[:i]
                    else:
                        x_i = x_infer[i-1-p:i-1]
                    if b_constant:
                        x_i.append(1)
                    x_i.reverse()
                    x_infer[i] = self.arma_one_step(x_i, e_i, phi, theta)
                else:
                    x_infer[i] = self.arma_one_step(e_i, theta=theta)
        elif p > 0:
            for i in range(t):
                if i == 0:
                    x_i = xx[:]
                elif (i > 0) and (i < (p + q)):
                    x_i = xx[-(pd-i):]
                    x_i = x_i + x_infer[:i]
                else:
                    x_i = x_infer[i-1-p:i-1]
                if b_constant:
                    x_i.append(1)
                x_i.reverse()
                x_infer[i] = self.arma_one_step(x=x_i, phi=phi)
        if d > 0:
            for i in range(t):
                if i == 0:
                    tx = x[-d:]
                elif i > 0 and i < d:
                    tx = x[-d+i:]
                    tx = tx + x_infer[i-(d-i):i]
                else:
                    tx = x_infer[i-d:i]
                x_infer[i] = x_infer[i] + self.reverse_integrate_d_degree_one_step(d, x_infer[i], tx)

        return x_infer

    def LM_statistic(
        self,
        residual_2,
        q,
        e_2,
    ):
        """
        LM statistic,  Applied Time Series Analysis（4th edition） Yan Wang p147
        Lagrange multiplier test, LM test.
        Parameters
        ----------
        residual_2
        q
        e_2

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        sst = residual_2[q:]
        sst = np.sum(sst)   # T-q-1
        sse = e_2[q:]
        sse = np.sum(sse)  # T-2q-1
        ssr = sst - sse
        lm = (ssr/q) / (sse / (n_residual_2 - 2*q - 1))  # q-1

        return lm

    def F_statistic(
        self,
        R_2,
        k,
        n
    ):
        """
        F statistic
        Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge  p217
        Parameters
        ----------
        R_2
        k
        n

        Returns
        -------

        """
        F_statistic = (R_2 / k) / ((1 - R_2) / (n - k - 1))
        return F_statistic

    def BPtest_LM_statistic(
        self,
        R_2,
        n,
    ):
        """
        BP test LM
        Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge  p217
        Parameters
        ----------
        R_2
        n

        Returns
        -------

        """
        bpLM = n * R_2
        return bpLM

    def get_F_critical(
        self,
        fd_n,
        fd_d,
        significance_level,
    ):
        """
        F test critical table
        Time Series Analysis  James D.Hamilton table B.3 F distribution P874
        Parameters
        ----------
        fd_n
        fd_d
        significance_level: only two significance level, 0.05 and 0.01.

        Returns
        -------

        """
        fd_numerator = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 80, 100, 125, 150, 200,
                        400, 1000, "∞"]
        fd_denominator = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 30, 40, 50, 75, 100, 200, 500, "∞"]
        p = [0.05, 0.01]
        data = [
            [[161, 200, 216, 225, 230, 234, 237, 239, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 253,
             254, 254, 254],
            [4052, 4999, 5403, 5625, 5764, 5859, 5928, 5981, 6022, 6056, 6082, 6106, 6142, 6169, 6203, 6234, 6258, 6286,
             6302, 6323, 6334, 6352, 6361, 6366]],
            [[18.51, 19, 19.16, 19.25, 19.3, 19.33, 19.36, 19.37, 19.38, 19.39, 19.4, 19.41, 19.42, 19.43, 19.44, 19.45,
             19.46, 19.47, 19.47, 19.48, 19.49, 19.49, 19.5, 19.5],
            [98.49, 99, 99.17, 99.25, 99.3, 99.33, 99.34, 99.36, 99.38, 99.4, 99.41, 99.42, 99.43, 99.44, 99.45, 99.46,
             99.47, 99.48, 99.48, 99.49, 99.49, 99.49, 99.5, 99.5]],
            [[10.13, 9.55, 9.28, 9.12, 9.01, 8.94, 8.88, 8.84, 8.81, 8.78, 8.76, 8.74, 8.71, 8.69, 8.66, 8.64, 8.62, 8.6,
             8.58, 8.57, 8.56, 8.54, 8.54, 8.53],
            [34.12, 30.82, 29.46, 28.71, 28.24, 27.91, 27.67, 27.49, 27.34, 27.23, 27.13, 27.05, 26.92, 26.83, 26.69,
             26.6, 26.5, 26.41, 26.35, 26.27, 26.23, 26.18, 26.14, 26.12]],
            [[7.71, 6.94, 6.59, 6.39, 6.26, 6.16, 6.09, 6.04, 6, 5.96, 5.93, 5.91, 5.87, 5.84, 5.8, 5.77, 5.74, 5.71,
             5.7, 5.68, 5.66, 5.65, 5.64, 5.63],
            [21.2, 18, 16.69, 15.98, 15.52, 15.21, 14.98, 14.8, 14.66, 14.54, 14.45, 14.37, 14.24, 14.15, 14.02, 13.93,
             13.83, 13.74, 13.69, 13.61, 13.57, 13.52, 13.48, 13.46]],
            [[6.61, 5.79, 5.41, 5.19, 5.05, 4.95, 4.88, 4.82, 4.78, 4.74, 4.7, 4.68, 4.64, 4.6, 4.56, 4.53, 4.5, 4.46,
             4.44, 4.42, 4.4, 4.38, 4.37, 4.36],
            [16.26, 13.27, 12.06, 11.39, 10.97, 10.67, 10.45, 10.27, 10.15, 10.05, 9.96, 9.89, 9.77, 9.68, 9.55, 9.47,
             9.38, 9.29, 9.24, 9.17, 9.13, 9.07, 9.04, 9.02]],
            [[5.99, 5.14, 4.76, 4.53, 4.39, 4.28, 4.21, 4.15, 4.1, 4.06, 4.03, 4, 3.96, 3.92, 3.87, 3.84, 3.81, 3.77,
             3.75, 3.72, 3.71, 3.69, 3.68, 3.67],
            [13.74, 10.92, 9.78, 9.15, 8.75, 8.47, 8.26, 8.1, 7.98, 7.87, 7.79, 7.72, 7.6, 7.52, 7.39, 7.31, 7.23, 7.14,
             7.09, 7.02, 6.99, 6.94, 6.9, 6.88]],
            [[5.59, 4.74, 4.35, 4.12, 3.97, 3.87, 3.79, 3.73, 3.68, 3.63, 3.6, 3.57, 3.52, 3.49, 3.44, 3.41, 3.38, 3.34,
             3.32, 3.29, 3.28, 3.25, 3.24, 3.23],
            [12.25, 9.55, 8.45, 7.85, 7.46, 7.19, 7, 6.84, 6.71, 6.62, 6.54, 6.47, 6.35, 6.27, 6.15, 6.07, 5.98, 5.9,
             5.85, 5.78, 5.75, 5.7, 5.67, 5.65]],
            [[5.32, 4.46, 4.07, 3.84, 3.69, 3.58, 3.5, 3.44, 3.39, 3.34, 3.31, 3.28, 3.23, 3.2, 3.15, 3.12, 3.08, 3.05,
             3.03, 3, 2.98, 2.96, 2.94, 2.93],
            [11.26, 8.65, 7.59, 7.01, 6.63, 6.37, 6.19, 6.03, 5.91, 5.82, 5.74, 5.67, 5.56, 5.48, 5.36, 5.28, 5.2, 5.11,
             5.06, 5, 4.96, 4.91, 4.88, 4.86]],
            [[5.12, 4.26, 3.86, 3.63, 3.48, 3.37, 3.29, 3.23, 3.18, 3.13, 3.1, 3.07, 3.02, 2.98, 2.93, 2.9, 2.86, 2.82,
             2.8, 2.77, 2.76, 2.73, 2.72, 2.71],
            [10.56, 8.02, 6.99, 6.42, 6.06, 5.8, 5.62, 5.47, 5.35, 5.26, 5.18, 5.11, 5, 4.92, 4.8, 4.73, 4.64, 4.56,
             4.51, 4.45, 4.41, 4.36, 4.33, 4.31]],
            [[4.96, 4.1, 3.71, 3.48, 3.33, 3.22, 3.14, 3.07, 3.02, 2.97, 2.94, 2.91, 2.86, 2.82, 2.77, 2.74, 2.7, 2.67,
             2.64, 2.61, 2.59, 2.56, 2.55, 2.54],
            [10.04, 7.56, 6.55, 5.99, 5.64, 5.39, 5.21, 5.06, 4.95, 4.85, 4.78, 4.71, 4.6, 4.52, 4.41, 4.33, 4.25, 4.17,
             4.12, 4.05, 4.01, 3.96, 3.93, 3.91]],
            [[4.84, 3.98, 3.59, 3.36, 3.2, 3.09, 3.01, 2.95, 2.9, 2.86, 2.82, 2.79, 2.74, 2.7, 2.65, 2.61, 2.57, 2.53,
             2.5, 2.47, 2.45, 2.42, 2.41, 2.4],
            [9.65, 7.2, 6.22, 5.67, 5.32, 5.07, 4.88, 4.74, 4.63, 4.54, 4.46, 4.4, 4.29, 4.21, 4.1, 4.02, 3.94, 3.86,
             3.8, 3.74, 3.7, 3.66, 3.62, 3.6]],
            [[4.75, 3.88, 3.49, 3.26, 3.11, 3, 2.92, 2.85, 2.8, 2.76, 2.72, 2.69, 2.64, 2.6, 2.54, 2.5, 2.46, 2.42, 2.4,
             2.36, 2.35, 2.32, 2.31, 2.3],
            [9.33, 6.93, 5.95, 5.41, 5.06, 4.82, 4.65, 4.5, 4.39, 4.3, 4.22, 4.16, 4.05, 3.93, 3.86, 3.78, 3.7, 3.61,
             3.56, 3.49, 3.46, 3.41, 3.38, 3.36]],
            [[4.67, 3.8, 3.41, 3.18, 3.02, 2.92, 2.84, 2.77, 2.72, 2.67, 2.63, 2.6, 2.55, 2.51, 2.46, 2.42, 2.38, 2.34,
             2.32, 2.28, 2.26, 2.24, 2.22, 2.21],
            [9.07, 6.7, 5.74, 5.2, 4.86, 4.62, 4.44, 4.3, 4.19, 4.1, 4.02, 3.96, 3.85, 3.78, 3.67, 3.59, 3.51, 3.42,
             3.37, 3.3, 3.27, 3.21, 3.18, 3.16]],
            [[4.6, 3.74, 3.34, 3.11, 2.96, 2.85, 2.77, 2.7, 2.65, 2.6, 2.56, 2.53, 2.48, 2.44, 2.39, 2.35, 2.31, 2.27,
             2.24, 2.21, 2.19, 2.16, 2.14, 2.13],
            [8.86, 6.51, 5.56, 5.03, 4.69, 4.46, 4.28, 4.14, 4.03, 3.94, 3.86, 3.8, 3.7, 3.62, 3.51, 3.43, 3.34, 3.26,
             3.21, 3.14, 3.11, 3.06, 3.02, 3]],
            [[4.54, 3.68, 3.29, 3.06, 2.9, 2.79, 2.7, 2.64, 2.59, 2.55, 2.51, 2.48, 2.43, 2.39, 2.33, 2.29, 2.25, 2.21,
             2.18, 2.15, 2.12, 2.1, 2.08, 2.07],
            [8.68, 6.36, 5.42, 4.89, 4.56, 4.32, 4.14, 4, 3.89, 3.8, 3.73, 3.67, 3.56, 3.48, 3.36, 3.29, 3.2, 3.12,
             3.07, 3, 2.97, 2.92, 2.89, 2.87]],
            [[4.49, 3.63, 3.24, 3.01, 2.85, 2.74, 2.66, 2.59, 2.54, 2.49, 2.45, 2.42, 2.37, 2.33, 2.28, 2.24, 2.2, 2.16,
             2.13, 2.09, 2.07, 2.04, 2.02, 2.01],
            [8.53, 6.23, 5.29, 4.77, 4.44, 4.2, 4.03, 3.89, 3.78, 3.69, 3.61, 3.55, 3.45, 3.37, 3.25, 3.18, 3.1, 3.01,
             2.96, 2.89, 2.86, 2.8, 2.77, 2.75]],
            [[4.45, 3.59, 3.2, 2.96, 2.81, 2.7, 2.62, 2.55, 2.5, 2.45, 2.41, 2.38, 2.33, 2.29, 2.23, 2.19, 2.15, 2.11,
             2.08, 2.04, 2.02, 1.99, 1.97, 1.96],
            [8.4, 6.11, 5.18, 4.67, 4.34, 4.1, 3.93, 3.79, 3.68, 3.59, 3.52, 3.45, 3.35, 3.27, 3.16, 3.08, 3, 2.92,
             2.86, 2.79, 2.76, 2.7, 2.67, 2.65]],
            [[4.41, 3.55, 3.16, 2.93, 2.77, 2.66, 2.58, 2.51, 2.46, 2.41, 2.37, 2.34, 2.29, 2.25, 2.19, 2.15, 2.11, 2.07,
             2.04, 2, 1.98, 1.95, 1.93, 1.92],
            [8.28, 6.01, 5.09, 4.58, 4.25, 4.01, 3.85, 3.71, 3.6, 3.51, 3.44, 3.37, 3.27, 3.19, 3.07, 3, 2.91, 2.83,
             2.78, 2.71, 2.68, 2.62, 2.59, 2.57]],
            [[4.38, 3.52, 3.13, 2.9, 2.74, 2.63, 2.55, 2.48, 2.43, 2.38, 2.34, 2.31, 2.26, 2.21, 2.15, 2.11, 2.07, 2.02,
             2, 1.96, 1.94, 1.91, 1.9, 1.88],
            [8.18, 5.93, 5.01, 4.5, 4.17, 3.94, 3.77, 3.63, 3.52, 3.43, 3.36, 3.3, 3.19, 3.12, 3, 2.92, 2.84, 2.76, 2.7,
             2.63, 2.6, 2.54, 2.51, 2.49]],
            [[4.35, 3.49, 3.1, 2.87, 2.71, 2.6, 2.52, 2.45, 2.4, 2.35, 2.31, 2.28, 2.23, 2.18, 2.12, 2.08, 2.04, 1.99,
             1.96, 1.92, 1.9, 1.87, 1.85, 1.84],
            [8.1, 5.85, 4.94, 4.43, 4.1, 3.87, 3.71, 3.56, 3.45, 3.37, 3.3, 3.23, 3.13, 3.05, 2.94, 2.86, 2.77, 2.69,
             2.63, 2.56, 2.53, 2.47, 2.44, 2.42]],
            [[4.32, 3.47, 3.07, 2.84, 2.68, 2.57, 2.49, 2.42, 2.37, 2.32, 2.28, 2.25, 2.2, 2.15, 2.09, 2.05, 2, 1.96,
             1.93, 1.89, 1.87, 1.84, 1.82, 1.81],
            [8.02, 5.78, 4.87, 4.37, 4.04, 3.81, 3.65, 3.51, 3.4, 3.31, 3.24, 3.17, 3.07, 2.99, 2.88, 2.8, 2.72, 2.63,
             2.58, 2.51, 2.47, 2.42, 2.38, 2.36]],
            [[4.3, 3.44, 3.05, 2.82, 2.66, 2.55, 2.47, 2.4, 2.35, 2.3, 2.26, 2.23, 2.18, 2.13, 2.07, 2.03, 1.98, 1.93,
             1.91, 1.87, 1.84, 1.81, 1.8, 1.78],
            [7.94, 5.72, 4.82, 4.31, 3.99, 3.76, 3.59, 3.45, 3.35, 3.26, 3.18, 3.12, 3.02, 2.94, 2.83, 2.75, 2.67, 2.58,
             2.53, 2.46, 2.42, 2.37, 2.33, 2.31]],
            [[4.28, 3.42, 3.03, 2.8, 2.64, 2.53, 2.45, 2.38, 2.32, 2.28, 2.24, 2.2, 2.14, 2.1, 2.04, 2, 1.96, 1.91, 1.88,
             1.84, 1.82, 1.79, 1.77, 1.76],
            [7.88, 5.66, 4.76, 4.26, 3.94, 3.71, 3.54, 3.41, 3.3, 3.21, 3.14, 3.07, 2.97, 2.89, 2.78, 2.7, 2.62, 2.53,
             2.48, 2.41, 2.37, 2.32, 2.28, 2.26]],
            [[4.26, 3.4, 3.01, 2.78, 2.62, 2.51, 2.43, 2.36, 2.3, 2.26, 2.22, 2.18, 2.13, 2.09, 2.02, 1.98, 1.94, 1.89,
             1.86, 1.82, 1.8, 1.76, 1.74, 1.73],
            [7.82, 5.61, 4.72, 4.22, 3.9, 3.67, 3.5, 3.36, 3.25, 3.17, 3.09, 3.03, 2.93, 2.85, 2.74, 2.66, 2.58, 2.49,
             2.44, 2.36, 2.33, 2.27, 2.23, 2.21]],
            [[4.24, 3.38, 2.99, 2.76, 2.6, 2.49, 2.41, 2.34, 2.28, 2.24, 2.2, 2.16, 2.11, 2.06, 2, 1.96, 1.92, 1.87,
             1.84, 1.8, 1.77, 1.74, 1.72, 1.71],
            [7.77, 5.57, 4.68, 4.18, 3.86, 3.63, 3.46, 3.32, 3.21, 3.13, 3.05, 2.99, 2.89, 2.81, 2.7, 2.62, 2.54, 2.45,
             2.4, 2.32, 2.29, 2.23, 2.19, 2.17]],
            [[4.22, 3.37, 2.98, 2.74, 2.59, 2.47, 2.39, 2.32, 2.27, 2.22, 2.18, 2.15, 2.1, 2.05, 1.99, 1.95, 1.9, 1.85,
             1.82, 1.78, 1.76, 1.72, 1.7, 1.69],
            [7.72, 5.53, 4.64, 4.14, 3.82, 3.59, 3.42, 3.29, 3.17, 3.09, 3.02, 2.96, 2.86, 2.77, 2.66, 2.58, 2.5, 2.41,
             2.36, 2.28, 2.25, 2.19, 2.15, 2.13]],
            [[4.21, 3.35, 2.96, 2.73, 2.57, 2.46, 2.37, 2.3, 2.25, 2.2, 2.16, 2.13, 2.08, 2.03, 1.97, 1.93, 1.88, 1.84,
             1.8, 1.76, 1.74, 1.71, 1.68, 1.67],
            [7.68, 5.49, 4.6, 4.11, 3.79, 3.56, 3.39, 3.26, 3.14, 3.06, 2.98, 2.93, 2.83, 2.74, 2.63, 2.55, 2.47, 2.38,
             2.33, 2.25, 2.21, 2.16, 2.12, 2.1]],
            [[4.2, 3.34, 2.95, 2.71, 2.56, 2.44, 2.36, 2.29, 2.24, 2.19, 2.15, 2.12, 2.06, 2.02, 1.96, 1.91, 1.87, 1.81,
             1.78, 1.75, 1.72, 1.69, 1.67, 1.65],
            [7.64, 5.45, 4.57, 4.07, 3.76, 3.53, 3.36, 3.23, 3.11, 3.03, 2.95, 2.9, 2.8, 2.71, 2.6, 2.52, 2.44, 2.35,
             2.3, 2.22, 2.18, 2.13, 2.09, 2.06]],
            [[4.18, 3.33, 2.93, 2.7, 2.54, 2.43, 2.35, 2.28, 2.22, 2.18, 2.14, 2.1, 2.05, 2, 1.94, 1.9, 1.85, 1.8, 1.77,
             1.73, 1.71, 1.68, 1.65, 1.64],
            [7.6, 5.42, 4.54, 4.04, 3.73, 3.5, 3.33, 3.2, 3.08, 3, 2.92, 2.87, 2.77, 2.68, 2.57, 2.49, 2.41, 2.32, 2.27,
             2.19, 2.15, 2.1, 2.06, 2.03]],
            [[4.17, 3.32, 2.92, 2.69, 2.53, 2.42, 2.34, 2.27, 2.21, 2.16, 2.12, 2.09, 2.04, 1.99, 1.93, 1.89, 1.84, 1.79,
             1.76, 1.72, 1.69, 1.66, 1.64, 1.62],
            [7.56, 5.39, 4.51, 4.02, 3.7, 3.47, 3.3, 3.17, 3.06, 2.98, 2.9, 2.84, 2.74, 2.66, 2.55, 2.47, 2.38, 2.29,
             2.24, 2.16, 2.13, 2.07, 2.03, 2.01]],
            [[4.15, 3.3, 2.9, 2.67, 2.51, 2.4, 2.32, 2.25, 2.19, 2.14, 2.1, 2.07, 2.02, 1.97, 1.91, 1.86, 1.82, 1.76,
             1.74, 1.69, 1.67, 1.64, 1.61, 1.59],
            [7.5, 5.34, 4.46, 3.97, 3.66, 3.42, 3.25, 3.12, 3.01, 2.94, 2.86, 2.8, 2.7, 2.62, 2.51, 2.42, 2.34, 2.25,
             2.2, 2.12, 2.08, 2.02, 1.98, 1.96]],
            [[4.13, 3.28, 2.88, 2.65, 2.49, 2.38, 2.3, 2.23, 2.17, 2.12, 2.08, 2.05, 2, 1.95, 1.89, 1.84, 1.8, 1.74,
             1.71, 1.67, 1.64, 1.61, 1.59, 1.57],
            [7.44, 5.29, 4.42, 3.93, 3.61, 3.38, 3.21, 3.08, 2.97, 2.89, 2.82, 2.76, 2.66, 2.58, 2.47, 2.38, 2.3, 2.21,
             2.15, 2.08, 2.04, 1.98, 1.94, 1.91]],
            [[4.11, 3.26, 2.86, 2.63, 2.48, 2.36, 2.28, 2.21, 2.15, 2.1, 2.06, 2.03, 1.98, 1.93, 1.87, 1.82, 1.78, 1.72,
             1.69, 1.65, 1.62, 1.59, 1.56, 1.55],
            [7.39, 5.25, 4.38, 3.89, 3.58, 3.35, 3.18, 3.04, 2.94, 2.86, 2.78, 2.72, 2.62, 2.54, 2.43, 2.35, 2.26, 2.17,
             2.12, 2.04, 2, 1.94, 1.9, 1.87]],
            [[4.1, 3.25, 2.85, 2.62, 2.46, 2.35, 2.26, 2.19, 2.14, 2.09, 2.05, 2.02, 1.96, 1.92, 1.85, 1.8, 1.76, 1.71,
             1.67, 1.63, 1.6, 1.57, 1.54, 1.53],
            [7.35, 5.21, 4.34, 3.86, 3.54, 3.32, 3.15, 3.02, 2.91, 2.82, 2.75, 2.69, 2.59, 2.51, 2.4, 2.32, 2.22, 2.14,
             2.08, 2, 1.97, 1.9, 1.86, 1.84]],
            [[4.08, 3.23, 2.84, 2.61, 2.45, 2.34, 2.25, 2.18, 2.12, 2.07, 2.04, 2, 1.95, 1.9, 1.84, 1.79, 1.74, 1.69,
             1.66, 1.61, 1.59, 1.55, 1.53, 1.51],
            [7.31, 5.18, 4.31, 3.83, 3.51, 3.29, 3.12, 2.99, 2.88, 2.8, 2.73, 2.66, 2.56, 2.49, 2.37, 2.29, 2.2, 2.11,
             2.05, 1.97, 1.94, 1.88, 1.84, 1.81]],
            [[4.07, 3.22, 2.83, 2.59, 2.44, 2.32, 2.24, 2.17, 2.11, 2.06, 2.02, 1.99, 1.94, 1.89, 1.82, 1.78, 1.73, 1.68,
             1.64, 1.6, 1.57, 1.54, 1.51, 1.49],
            [7.27, 5.15, 4.29, 3.8, 3.49, 3.26, 3.1, 2.96, 2.86, 2.77, 2.7, 2.64, 2.54, 2.46, 2.35, 2.26, 2.17, 2.08,
             2.02, 1.94, 1.91, 1.85, 1.8, 1.78]],
            [[4.06, 3.21, 2.82, 2.58, 2.43, 2.31, 2.23, 2.16, 2.1, 2.05, 2.01, 1.98, 1.92, 1.88, 1.81, 1.76, 1.72, 1.66,
             1.63, 1.58, 1.56, 1.52, 1.5, 1.48],
            [7.24, 5.12, 4.26, 3.78, 3.46, 3.24, 3.07, 2.94, 2.84, 2.75, 2.68, 2.62, 2.52, 2.44, 2.32, 2.24, 2.15, 2.06,
             2, 1.92, 1.88, 1.82, 1.78, 1.75]],
            [[4.05, 3.2, 2.81, 2.57, 2.42, 2.3, 2.22, 2.14, 2.09, 2.04, 2, 1.97, 1.91, 1.87, 1.8, 1.75, 1.71, 1.65, 1.62,
             1.57, 1.54, 1.51, 1.48, 1.46],
            [7.21, 5.1, 4.24, 3.76, 3.44, 3.22, 3.05, 2.92, 2.82, 2.73, 2.66, 2.6, 2.5, 2.42, 2.3, 2.22, 2.13, 2.04,
             1.98, 1.9, 1.86, 1.8, 1.76, 1.72]],
            [[4.04, 3.19, 2.8, 2.56, 2.41, 2.3, 2.21, 2.14, 2.08, 2.03, 1.99, 1.96, 1.9, 1.86, 1.79, 1.74, 1.7, 1.64,
             1.61, 1.56, 1.53, 1.5, 1.47, 1.45],
            [7.19, 5.08, 4.22, 3.74, 3.42, 3.2, 3.04, 2.9, 2.8, 2.71, 2.64, 2.58, 2.48, 2.4, 2.28, 2.2, 2.11, 2.02,
             1.96, 1.88, 1.84, 1.78, 1.73, 1.7]],
            [[4.03, 3.18, 2.79, 2.56, 2.4, 2.29, 2.2, 2.13, 2.07, 2.02, 1.98, 1.95, 1.9, 1.85, 1.78, 1.74, 1.69, 1.63,
             1.6, 1.55, 1.52, 1.48, 1.46, 1.44],
            [7.17, 5.06, 4.2, 3.72, 3.41, 3.18, 3.02, 2.88, 2.78, 2.7, 2.62, 2.56, 2.46, 2.39, 2.26, 2.18, 2.1, 2, 1.94,
             1.86, 1.82, 1.76, 1.71, 1.68]],
            [[4.02, 3.17, 2.78, 2.54, 2.38, 2.27, 2.18, 2.11, 2.05, 2, 1.97, 1.93, 1.88, 1.83, 1.76, 1.72, 1.67, 1.61,
             1.58, 1.52, 1.5, 1.46, 1.43, 1.41],
            [7.12, 5.01, 4.16, 3.68, 3.37, 3.15, 2.98, 2.85, 2.75, 2.66, 2.59, 2.53, 2.43, 2.35, 2.23, 2.15, 2.06, 1.96,
             1.9, 1.82, 1.78, 1.71, 1.66, 1.64]],
            [[4, 3.15, 2.76, 2.52, 2.37, 2.25, 2.17, 2.1, 2.04, 1.99, 1.95, 1.92, 1.86, 1.81, 1.75, 1.7, 1.65, 1.59,
             1.56, 1.5, 1.48, 1.44, 1.41, 1.39],
            [7.08, 4.98, 4.13, 3.65, 3.34, 3.12, 2.95, 2.82, 2.72, 2.63, 2.56, 2.5, 2.4, 2.32, 2.2, 2.12, 2.03, 1.93,
             1.87, 1.79, 1.74, 1.68, 1.63, 1.6]],
            [[3.99, 3.14, 2.75, 2.51, 2.36, 2.24, 2.15, 2.08, 2.02, 1.98, 1.94, 1.9, 1.85, 1.8, 1.73, 1.68, 1.63, 1.57,
             1.54, 1.49, 1.46, 1.42, 1.39, 1.37],
            [7.04, 4.95, 4.1, 3.62, 3.31, 3.09, 2.93, 2.79, 2.7, 2.61, 2.54, 2.47, 2.37, 2.3, 2.18, 2.09, 2, 1.9, 1.84,
             1.76, 1.71, 1.64, 1.6, 1.56]],
            [[3.98, 3.13, 2.74, 2.5, 2.35, 2.23, 2.14, 2.07, 2.01, 1.97, 1.93, 1.89, 1.81, 1.79, 1.72, 1.67, 1.62, 1.56,
             1.53, 1.47, 1.45, 1.4, 1.37, 1.35],
            [7.01, 4.92, 4.08, 3.6, 3.29, 3.07, 2.91, 2.77, 2.67, 2.59, 2.51, 2.45, 2.35, 2.28, 2.15, 2.07, 1.98, 1.88,
             1.82, 1.74, 1.69, 1.62, 1.56, 1.53]],
            [[3.96, 3.11, 2.72, 2.48, 2.33, 2.21, 2.12, 2.05, 1.99, 1.95, 1.91, 1.88, 1.82, 1.77, 1.7, 1.65, 1.6, 1.54,
             1.51, 1.45, 1.42, 1.38, 1.35, 1.32],
            [6.96, 4.88, 4.04, 3.56, 3.25, 3.04, 2.87, 2.74, 2.64, 2.55, 2.48, 2.41, 2.32, 2.24, 2.11, 2.03, 1.94, 1.84,
             1.78, 1.7, 1.65, 1.57, 1.52, 1.49]],
            [[3.94, 3.09, 2.7, 2.46, 2.3, 2.19, 2.1, 2.03, 1.97, 1.92, 1.88, 1.85, 1.79, 1.75, 1.68, 1.63, 1.57, 1.51,
             1.48, 1.42, 1.39, 1.34, 1.3, 1.28],
            [6.9, 4.82, 3.98, 3.51, 3.2, 2.99, 2.82, 2.69, 2.59, 2.51, 2.43, 2.36, 2.26, 2.19, 2.06, 1.98, 1.89, 1.79,
             1.73, 1.64, 1.59, 1.51, 1.46, 1.43]],
            [[3.92, 3.07, 2.68, 2.44, 2.29, 2.17, 2.08, 2.01, 1.95, 1.9, 1.86, 1.83, 1.77, 1.72, 1.65, 1.6, 1.55, 1.49,
             1.45, 1.39, 1.36, 1.31, 1.27, 1.25],
            [6.84, 4.78, 3.94, 3.47, 3.17, 2.95, 2.79, 2.65, 2.56, 2.47, 2.4, 2.33, 2.23, 2.15, 2.03, 1.94, 1.85, 1.75,
             1.68, 1.59, 1.54, 1.46, 1.4, 1.37]],
            [[3.91, 3.06, 2.67, 2.43, 2.27, 2.16, 2.07, 2, 1.94, 1.89, 1.85, 1.82, 1.76, 1.71, 1.64, 1.59, 1.54, 1.47,
             1.44, 1.37, 1.34, 1.29, 1.25, 1.22],
            [6.81, 4.75, 3.91, 3.44, 3.14, 2.92, 2.76, 2.62, 2.53, 2.44, 2.37, 2.3, 2.2, 2.12, 2, 1.91, 1.83, 1.72,
             1.66, 1.56, 1.51, 1.43, 1.37, 1.33]],
            [[3.89, 3.04, 2.65, 2.41, 2.26, 2.14, 2.05, 1.98, 1.92, 1.87, 1.83, 1.8, 1.71, 1.69, 1.62, 1.57, 1.52, 1.45,
             1.42, 1.35, 1.32, 1.26, 1.22, 1.19],
            [6.76, 4.71, 3.88, 3.41, 3.11, 2.9, 2.73, 2.6, 2.5, 2.41, 2.34, 2.28, 2.17, 2.09, 1.97, 1.88, 1.79, 1.69,
             1.62, 1.53, 1.48, 1.39, 1.33, 1.28]],
            [[3.86, 3.02, 2.62, 2.39, 2.23, 2.12, 2.03, 1.96, 1.9, 1.85, 1.81, 1.78, 1.72, 1.67, 1.6, 1.54, 1.49, 1.42,
             1.38, 1.32, 1.28, 1.22, 1.16, 1.13],
            [6.7, 4.66, 3.83, 3.36, 3.06, 2.85, 2.69, 2.55, 2.46, 2.37, 2.29, 2.23, 2.12, 2.04, 1.92, 1.84, 1.74, 1.64,
             1.57, 1.47, 1.42, 1.32, 1.24, 1.19]],
            [[3.85, 3, 2.61, 2.38, 2.22, 2.1, 2.02, 1.95, 1.89, 1.84, 1.8, 1.76, 1.7, 1.65, 1.58, 1.53, 1.47, 1.41, 1.36,
             1.3, 1.26, 1.19, 1.13, 1.08],
            [6.66, 4.62, 3.8, 3.34, 3.04, 2.82, 2.66, 2.53, 2.43, 2.34, 2.26, 2.2, 2.09, 2.01, 1.89, 1.81, 1.71, 1.61,
             1.54, 1.44, 1.38, 1.28, 1.19, 1.11]],
            [[3.84, 2.99, 2.6, 2.37, 2.21, 2.09, 2.01, 1.94, 1.88, 1.83, 1.79, 1.75, 1.69, 1.64, 1.57, 1.52, 1.46, 1.4,
             1.35, 1.28, 1.24, 1.17, 1.11, 1],
            [6.64, 4.6, 3.78, 3.32, 3.02, 2.8, 2.64, 2.51, 2.41, 2.32, 2.24, 2.18, 2.07, 1.99, 1.87, 1.79, 1.69, 1.59,
             1.52, 1.41, 1.36, 1.25, 1.15, 1]],
        ]
        data = np.array(data)

        # locate
        fd_name = [fd_numerator, fd_denominator]
        fd_ = [30, 12]
        n_fd_ = 2
        fd_v = [fd_n, fd_d]
        fd_k = [0]*n_fd_
        fd_within = []
        for k in range(n_fd_):
            n_i = None
            n_within = None
            n_n_index = len(fd_name[k])
            if (fd_v[k] >= fd_name[k][0]) and (fd_v[k] <= fd_name[k][-2]):
                if fd_v[k] in fd_name[k]:
                    n_i = fd_name[k].index(fd_v[k])
                else:
                    fd = fd_name[k].index(fd_[k])
                    for i in range(fd, n_n_index - 2):
                        if (fd_v[k] > fd_name[k][i]) and (fd_v[k] < fd_name[k][i + 1]):
                            n_i = [i, i + 1]
                            n_within = [fd_name[k][i], fd_name[k][i + 1]]
                            break
            elif fd_v[k] > fd_name[k][-2]:
                n_i = n_n_index - 1
            else:
                raise ValueError('Index m = ' + str(fd_v[k]) + 'out of range.')
            fd_k[k] = n_i
            if n_within is not None:
                fd_within.append(n_within[:])
            else:
                fd_within.append(None)

        if significance_level in p:
            sl_i = p.index(significance_level)
        else:
            raise ValueError('Significance level = ' + str(significance_level) + ' not in Significance level array.')

        # querying
        if type(fd_k[0]) is list:
            if type(fd_k[1]) is list:
                critical_00 = data[fd_k[0][0], sl_i, fd_k[1][0]]
                critical_01 = data[fd_k[0][1], sl_i, fd_k[1][0]]
                critical_10 = data[fd_k[0][0], sl_i, fd_k[1][1]]
                critical_11 = data[fd_k[0][1], sl_i, fd_k[1][1]]
                F_critical0 = (critical_01 - critical_00) / (fd_within[0][1] - fd_within[0][0]) * (fd_n - fd_within[0][0]) + critical_00  # linear interpolation
                F_critical1 = (critical_11 - critical_10) / (fd_within[0][1] - fd_within[0][0]) * (fd_n - fd_within[0][0]) + critical_10
                F_critical = (F_critical1 - F_critical0) / (fd_within[1][1] - fd_within[1][0]) * (fd_d - fd_within[1][0]) + F_critical0
            else:
                critical_0 = data[fd_k[0][0], sl_i, fd_k[1]]
                critical_1 = data[fd_k[0][1], sl_i, fd_k[1]]
                F_critical = (critical_1 - critical_0) / (fd_within[0][1] - fd_within[0][0]) * (fd_n - fd_within[0][0]) + critical_0
        else:
            if type(fd_k[1]) is list:
                critical_0 = data[fd_k[0], sl_i, fd_k[1][0]]
                critical_1 = data[fd_k[0], sl_i, fd_k[1][1]]
                F_critical = (critical_1 - critical_0) / (fd_within[1][1] - fd_within[1][0]) * (fd_d - fd_within[1][0]) + critical_0
            else:
                F_critical = data[fd_k[0], sl_i, fd_k[1]]

        return F_critical

    def p_statistic(
        self,
        x,
    ):
        """
        p statistic
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
        p_statistic = np.abs(z) < 0.01
        return p_statistic

    def arch_test(
        self,
        residual,
        q,
        significance_level,
    ):
        """
        Applied Time Series Analysis（4th edition） Yan Wang p145-148
        Introductory Econometrics: A Modern Approach (6th edition) Jeffrey M. Wooldridge p216-219

        Parameters
        ----------
        residual
        q

        Returns
        -------

        """
        residual_2 = np.power(residual, 2)
        residual_2 = residual_2.tolist()
        # Portmanteau Q test
        Q_statistic, acf = self.LB_statistic(residual_2, q)
        chi_critical_Q = self.get_chi_critical(q-1, significance_level)

        # LM test  # todo:
        a, R_2 = self.ar_least_squares_estimation(residual_2, q, True)
        residual_2_fit = self.arma(x=residual_2, e=None, phi=a, theta=None, p=q, q=0, b_constant=True)  # ar model
        e = residual_2 - residual_2_fit
        e_2 = np.power(e, 2)
        LM_statistic = self.LM_statistic(residual_2, q, e_2)
        chi_critical_LM = self.get_chi_critical(q-1, significance_level)

        # F test
        n_residual_2 = len(residual_2)
        F_statistic = self.F_statistic(R_2, q, n_residual_2)
        F_critical = self.get_F_critical(n_residual_2-q-1, q, significance_level)

        # bpLM test
        bpLM_statistic = self.BPtest_LM_statistic(R_2, n_residual_2)
        chi_critical_bpLM = self.get_chi_critical(q, significance_level)

        # assumption
        H0 = False  # std1=std2=...=stdk=0, no ARCH.
        H1 = True   # exist stdi!=0 (1<i<k), be ARCH.

        # test
        if Q_statistic <= chi_critical_Q:
            b_arch_Q = H0
        else:
            b_arch_Q = H1

        if LM_statistic <= chi_critical_LM:
            b_arch_LM = H0
        else:
            b_arch_LM = H1

        if F_statistic < F_critical:
            b_arch_F = H0
        else:
            b_arch_F = H1

        if bpLM_statistic <= chi_critical_bpLM:
            b_arch_bpLM = H0
        else:
            b_arch_bpLM = H1

        return b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM

    def delta_2_one_step(
        self,
        residual_2,
        alpha,
    ):
        """
        delta_2, one step.   single step
        GARCH Models  Francq & Zakoian 2010 P128  formula (6.3)
        Parameters
        ----------
        residual_2:
        e:
        omega:
        q: degree of arch model.
        alpha:

        Returns
        -------

        """
        # residual_2_t = np.transpose(residual_2)
        delta_2_i = np.matmul(residual_2, alpha)

        return delta_2_i

    def epsilon_2(
        self,
        residual_2,
        alpha,
        q,
    ):
        """
        GARCH Models  Francq & Zakoian 2010 P128  formula (6.3)
        Parameters
        ----------
        residual_2
        e
        alpha
        q

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        epsilon_2 = np.zeros(n_residual_2)
        epsilon_2[:q] = np.sqrt(residual_2[:q])

        alpha_t = np.transpose(alpha)

        for i in range(q, n_residual_2):
            residual_2_i = residual_2[i-q:i]
            residual_2_i.append(1)  # omega
            residual_2_i.reverse()
            epsilon_2[i] = self.delta_2_one_step(residual_2_i, alpha_t)

        return epsilon_2

    def arch_ordinary_least_squares_estimation(
        self,
        residual_2,
        q,
    ):
        """
        GARCH Models Structure, Statistical Inference and Financial Applications  Christian Francq & Jean-Michel Zakoian P128
        Parameters
        ----------
        residual_2
        q

        Returns
        -------
        a: theta0
        delta_2: delta0_2
        """
        n_residual_2 = len(residual_2)

        # construct matrix
        xf = residual_2[q:]
        xf = np.array(xf)
        xf = np.transpose(xf)

        xp = []
        for i in range(q, n_residual_2):
            xp_i = residual_2[i-q:i]
            xp_i.append(1)  # omega
            xp_i.reverse()
            xp.append(xp_i)
        a, R_2 = self.ordinary_least_squares(xp, xf)

        y_ = self.epsilon_2(residual_2, a, q)
        delta_2 = np.sum(residual_2 - y_) / (n_residual_2-q-1)

        return a, R_2, delta_2

    def generalized_least_squares(
        self,
        A,
        Y,
        Omega,
        b_y: bool = False,
    ):
        """
        generalized least squares,
        GARCH Models  Francq & Zakoian 2010 P132
        Parameters
        ----------
        A
        Y
        Omega

        Returns
        -------

        """
        # matrix operations, calculate the coefficient matrix.
        A = np.array(A)
        At = np.transpose(A)
        B = np.matmul(At, Omega)
        B = np.matmul(B, A)
        try:
            B_1 = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        a = np.matmul(B_1, At)
        a = np.matmul(a, Omega)
        try:
            a = np.matmul(a, Y)  # parameters
        except ValueError:
            raise ValueError("matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 22 is different from 23)")

        # R_2, coefficient of determination
        y_ = np.matmul(A, a)
        R = self.correlation_coefficient(Y, y_)
        R_2 = pow(R, 2)

        if b_y:
            return a, R_2, y_

        return a, R_2

    def Omega_i(
        self,
        Zi,
        theta,
    ):
        """
        sigma_i(-4)
        GARCH Models  Francq & Zakoian 2010 P132
        Parameters
        ----------
        residual_2
        theta

        Returns
        -------

        """
        theta = np.transpose(theta)
        sigma_2 = np.matmul(Zi, theta)
        # sigma = np.sqrt(sigma_2)
        sigma_n4 = pow(sigma_2, -2)

        return sigma_n4


    def Omega(
        self,
        residual_2,
        theta,
    ):
        """
        Omega, based on ARCH(1).
        GARCH Models  Francq & Zakoian 2010 P132
        Parameters
        ----------
        residual_2
        theta
        q

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        Omega = [0]*(n_residual_2-1)
        for i in range(1, n_residual_2):
            x_i = [1] + [residual_2[i]]
            Omega[i-1] = self.Omega_i(x_i, theta)

        return Omega

    def arch_feasible_generalized_least_squares_estimation(
        self,
        residual_2,
        q,
        Omega,
        b_y: bool = False
    ):
        """
        fgls
        GARCH Models Structure, Statistical Inference and Financial Applications  Christian Francq & Jean-Michel Zakoian P132
        Parameters
        ----------
        residual_2
        q
        Omega

        Returns
        -------

        """
        Omega_diagonal = np.diag(Omega[q - 1:])
        n_residual_2 = len(residual_2)

        # construct matrix
        xf = residual_2[q:]
        xf = np.transpose(xf)

        xp = []
        for i in range(q, n_residual_2):
            xp_i = residual_2[i-q:i]
            xp_i.append(1)  # omega
            xp_i.reverse()
            xp.append(xp_i)

        if b_y:
            a, R_2, y = self.generalized_least_squares(xp, xf, Omega_diagonal, b_y=b_y)
            return a, R_2, y
        else:
            a, R_2 = self.generalized_least_squares(xp, xf, Omega_diagonal)
            return a, R_2

    def arch_constrained_ordinary_least_squares(
        self,
        residual_2,
        q,
        Omega,
        q_n,
        b_y: bool = False,
    ):
        """
        constrained ordinary least squares
        GARCH Models  Francq & Zakoian 2010 p135-137
        Parameters
        ----------
        residual_2
        q

        Returns
        -------

        """
        Omega_diagonal = np.diag(Omega[q - 1:])
        n_residual_2 = len(residual_2)

        # construct matrix
        xf = residual_2[q:]
        xf = np.transpose(xf)

        xp = []
        for i in range(q, n_residual_2):
            xp_i = residual_2[i-q:i]
            xp_i.append(1)  # omega
            xp_i.reverse()
            xp.append(xp_i)

        xp = np.array(xp)
        xp = np.delete(xp, q_n, axis=1)

        if b_y:
            a, R_2, y = self.generalized_least_squares(xp, xf, Omega_diagonal, b_y=b_y)
            a = np.insert(a, q_n, 0)
            return a, R_2, y
        else:
            a, R_2 = self.generalized_least_squares(xp, xf, Omega_diagonal)
            a = np.insert(a, q_n, 0)
            return a, R_2

        # a, R_2 = self.ordinary_least_squares(xp, xf)
        # a = np.insert(a, q_n, 0)
        # return a, R_2

    def initial_values(
        self,
        alpha,
        beta,
    ):
        """
        initial values
        GARCH Models  Francq & Zakoian 2010 p142 formula (7.5)
        Parameters
        ----------
        alpha
        beta

        Returns
        -------

        """
        omega = alpha[0]
        ini_values = omega / (1- sum(alpha[1:]) - sum(beta))

        if ini_values < 0:
            ini_values = omega

        return ini_values

    def conditional_quasi_likelihood(
        self,
        residual_2,
        delta_2,
        alpha,
        beta,
    ):
        """
        conditional quasi likelihood
        GARCH Models  Francq & Zakoian 2010 p142-143
        Parameters
        ----------
        residual_2
        q
        alpha

        Returns
        -------

        """
        theta = alpha + beta
        theta = np.transpose(theta)
        x = residual_2 + delta_2
        delta_2 = np.matmul(x, theta)

        return delta_2

    def likelihood_equations(
        self,
        residual_2,
        delta_2,
        eta_2,
    ):
        """
        likelihood equations
        GARCH Models  Francq & Zakoian 2010 p143
        Parameters
        ----------
        residual_2
        delta_2
        eta_2

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        l = np.array(residual_2) - np.array(delta_2)
        delta_n4 = np.power(delta_2, -2)
        l = l * delta_n4
        l = np.average(l)

        alpha = []  # todo:
        beta = []
        p = 3
        q = 2
        alpha_eta = alpha * eta_2  # GARCH Models  Francq & Zakoian 2010 p144
        beta_eta = beta * eta_2

        unit_diag_q = np.identity(q-1)
        A0_t1 = np.zeros((q, q))
        A0_t1[0, :] = alpha_eta
        A0_t1[1:,:(q-1)] = unit_diag_q
        A0_t2 = np.zeros((p, p))
        A0_t2[0, :] = beta_eta
        A0_t3 = np.zeros((q, q))
        A0_t3[0, :] = alpha
        unit_diag_p = np.identity(p - 1)
        A0_t4 = np.zeros((p, p))
        A0_t4[0, :] = beta
        A0_t4[1:, :(p - 1)] = unit_diag_p
        A0_t1 = np.concatenate((A0_t1, A0_t2), axis=1)
        A0_t3 = np.concatenate((A0_t3, A0_t4), axis=1)
        A0_t = np.concatenate((A0_t1, A0_t3), axis=0)

        return l

    def arch_asymptotic_variance(
        self,
        residual_2,
        alpha,
    ):
        """
        GARCH Models  Francq & Zakoian 2010  p147
        Parameters
        ----------
        residual_2
        alpha

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        sum_f = 0
        for i in range(1, n_residual_2):
            x_i = residual_2[i]
            delta_2_i = alpha[0] + alpha[1] * residual_2[i-1]
            f = x_i - delta_2_i
            delta_2_i_2 = np.power(delta_2_i, 2)
            f = f / delta_2_i_2
            sum_f = sum_f + f
        average_f = sum_f / n_residual_2

        return average_f

    def matrix_J(
        self,
        residual_2,
        alpha,
    ):
        """
        GARCH Models  Francq & Zakoian 2010  p147
        Parameters
        ----------
        residual_2
        alpha

        Returns
        -------

        """
        n_residual_2 = len(residual_2)

        matrixj = []
        for i in range(1, n_residual_2):
            x_i = residual_2[i]
            delta_2_i = alpha[0] + alpha[1] * residual_2[i-1]
            x_i_2 = np.power(x_i, 2)
            delta_2_i_2 = np.power(delta_2_i, 2)
            j_i = np.zeros((2, 2))
            j_i[0, 0] = 1 / delta_2_i
            j_i[0, 1] = x_i / delta_2_i
            j_i[1, 0] = x_i / delta_2_i
            j_i[1, 1] = x_i_2 / delta_2_i_2
            matrixj.append(j_i)

        return matrixj

    def arma_garch(
        self,
        x,
        eta,
        P,
        Q,
        p,
        q,
        a,
        b,
        alpha,
        beta,
        e0,
        h0,
    ):
        """
        GARCH Models  Francq & Zakoian 2010  p150
        Parameters
        ----------
        x
        P
        Q
        p
        q
        a
        b
        alpha
        beta

        Returns
        -------

        """
        n_e0 = len(e0)
        n_h0 = len(h0)
        start = max(P, Q, p, q)
        if n_e0 < start:
            raise ValueError("the length of e0 cannot be less than max(P, Q, p, q).")
        if n_h0 < start:
            raise ValueError("the length of h0 cannot be less than max(P, Q, p, q).")
        n_x = len(x)
        c0 = np.mean(x)
        a = np.transpose(a)
        b = np.transpose(b)
        alpha = np.transpose(alpha)
        beta = np.transpose(beta)


        e = [0]*n_x
        h = [0]*n_x
        xx = [0]*n_x
        for i in range(start, n_x):
            e_ = e[i-q:i]
            e_.append(1)
            e_.reverse()
            h_ = h[i-p:i]
            h_.reverse()
            h_i = np.matmul(e_, alpha) + np.matmul(h_, beta)
            e_i = np.sqrt(h_i) * eta[i]
            h[i] = h_i
            e[i] = e_i
            x_i = x[i-P:i]
            x_i.reverse()
            x_i = np.array(x_i) - c0
            ei = e[i-1-Q:i-1]
            ei.reverse()
            xx_i = np.matmul(x_i, a) + e[i] - np.matmul(ei, b) + c0
            xx[i] = xx_i

        return h, e, xx

    def asymptotic_variance_a0_alpha0(
        self,
        a,
        alpha,
    ):
        """
        asymptotic variance
        GARCH Models  Francq & Zakoian 2010  p155
        Parameters
        ----------
        a
        alpha

        Returns
        -------

        """
        a0 = [0, -0.5, -0.9]
        alpha = [0, 0.1, 0.25, 0.5]
        av00_N = [[1.00, 0.00], [0.00, 0.67]]
        av01_N = [[1.14, 0.00], [0.00, 1.15]]
        av02_N = [[1.20, 0.00], [0.00, 1.82]]
        av03_N = [[1.08, 0.00], [0.00, 2.99]]
        av00_X = [[1.00, -0.54], [-0.54, 0.94]]
        av01_X = [[1.70, -1.63], [-1.63, 8.01]]
        av02_X = [[2.78, -1.51], [-1.51, 18.78]]
        av03_X = "-"
        av10_N = [[0.75, 0.00], [0.00, 0.67]]
        av11_N = [[0.82, 0.00], [0.00, 1.15]]
        av12_N = [[0.83, 0.00], [0.00, 1.82]]
        av13_N = [[0.72, 0.00], [0.00, 2.99]]
        av10_X = [[0.75, -0.40], [-0.40, 0.94]]
        av11_X = [[1.04, -0.99], [-0.99, 8.02]]
        av12_X = [[1.41, -0.78], [-0.78, 18.85]]
        av13_X = "-"
        av20_N = [[0.19, 0.00], [0.00, 0.67]]
        av21_N = [[0.19, 0.00], [0.00, 1.15]]
        av22_N = [[0.18, 0.00], [0.00, 1.82]]
        av23_N = [[0.13, 0.00], [0.00, 2.98]]
        av20_X = [[0.19, -0.10], [-0.10, 0.94]]
        av21_X = [[0.20, -0.19], [-0.19, 8.01]]
        av22_X = [[0.21, -0.12], [-0.12, 18.90]]
        av23_X = "-"
        av = np.empty((2, 2, 4))

        # set value
        av[0, 0, 0] = av00_N

        return av

    def delta_2(
        self,
        residual_2,
        alpha,
    ):
        """
        Time Series Analysis  James D.Hamilton P766
        Parameters
        ----------
        residual_2
        alpha

        Returns
        -------

        """
        n_residual_2 = len(residual_2)
        q = len(alpha) - 1

        alpha_t = np.transpose(alpha)
        delta_2 = []
        for i in range(q, n_residual_2):
            x_i = residual_2[i-q:i]
            x_i.append(1)
            x_i.reverse()
            delta_2_i = self.delta_2_one_step(x_i, alpha_t)
            delta_2.append(delta_2_i)

        return delta_2

    def log_likelihood_bc(
        self,
        residual_2,
        alpha
    ):
        """
        Time Series Analysis  James D.Hamilton P766
        Parameters
        ----------
        residual_2
        alpha

        Returns
        -------

        """
        q = len(alpha) -1
        alpha_ = np.transpose(alpha)
        delta_2 = self.delta_2(residual_2, alpha_)
        L_theta_b = np.log(delta_2) / 2
        L_theta_b = np.sum(L_theta_b)
        L_theta_c = np.array(residual_2[q:]) / np.array(delta_2)
        L_theta_c = np.sum(L_theta_c) / 2
        L_theta_bc = L_theta_b + L_theta_c

        return L_theta_bc

    def distance_theta_0_1(
        self,
        theta0,
        theta1,
    ):
        """
        Time Series Analysis  James D.Hamilton P155
        Parameters
        ----------
        theta0
        theta1

        Returns
        -------

        """
        d = np.array(theta1) - np.array(theta0)
        d_t = np.transpose(d)
        dist = np.matmul(d, d_t)

        return dist

    def gradient_theta(
        self,
        lambd,
        theta0,
        theta1,
    ):
        """
        gradient
        Time Series Analysis  James D.Hamilton P155
        Parameters
        ----------
        lambd
        theta0
        theta1

        Returns
        -------

        """
        grad = -2 * lambd * (np.array(theta1) - np.array(theta0))

        return grad

    def grid_search(
        self,
        residual_2,
        theta0,
        grad,  # todo:
    ):
        """
        grid searching
        Time Series Analysis  James D.Hamilton P157
        Parameters
        ----------
        theta0

        Returns
        -------

        """
        s = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
        n_s = len(s)

        L_theta_bc = [0] * n_s
        theta1 = []
        for i in range(n_s):
            theta1_i = np.array(theta0) + s[i] * grad
            theta1.append(theta1_i)
            L_theta_bc_i = self.log_likelihood_bc(residual_2, theta1_i)
            L_theta_bc[i] = L_theta_bc_i
        i_max = np.argmax(L_theta_bc)
        theta1_ = theta1[i_max]

        return theta1_

    def gradient_thetai(
        self,
        x,
        theta,
        d_theta,
        i_theta,
        p,
        q,
    ):
        """
        gradient of single parameter.
        Time Series Analysis  James D.Hamilton P156
        Parameters
        ----------
        theta: list of parameters.
        d_theta: the change range of single parameter.
        i_theta: the index of single parameter in parameter list.

        Returns
        -------

        """
        theta_i = theta[i_theta] + d_theta
        theta1 = theta
        theta1[i_theta] = theta_i
        phi0 = theta[:p]
        alpha0 = theta[p:]
        residual0 = self.x_residual_via_parameters(x=x, phi=phi0)
        residual0_2 = np.power(residual0, 2)
        L0 = self.log_likelihood_bc(residual0_2, alpha0)
        phi1 = theta1[:p]
        alpha1 = theta1[p:]
        residual1 = self.x_residual_via_parameters(x=x, phi=phi1)
        residual1_2 = np.power(residual1, 2)
        L1 = self.log_likelihood_bc(residual1_2, alpha1)

        grad = (L1 - L0) / d_theta

        return grad

    def x_residual_via_parameters(
        self,
        x,
        phi,
    ):
        """
        residual via parameters.
        Parameters
        ----------
        x
        alpha

        Returns
        -------

        """
        p = len(phi)
        y_t = self.arima(x=x, phi=phi, p=p)
        x_residual = np.array(x) - np.array(y_t)

        return x_residual

    def gradient(
        self,
        x,
        theta,
        d_theta,
        p,
        q,
    ):
        """
        gradient of all parameters.
        Time Series Analysis  James D.Hamilton P156
        Parameters
        ----------
        x
        theta
        d_theta
        p
        q

        Returns
        -------

        """
        n_theta = len(theta)
        if n_theta != p + q:
            raise ValueError("the length of theta do not equal to p+q, error!")

        gradient = [0]*n_theta
        for i in range(n_theta):
            gradient[i] = self.gradient_thetai(x, theta, d_theta, i, p, q)

        return gradient


    def mL_estimation(
        self,
        x,
        y,
        e,
    ):
        """
        maximum likelihood estimation, 概率论与数理统计 p152
        Time Series Analysis  James D.Hamilton p131
        pdf, likelihood function
        Parameters
        ----------
        x

        Returns
        -------

        """
        b0 = 0.5
        b1 = 0.2
        wt = 3
        pi = np.pi
        sum_lene = 0
        sum_y = 0
        sum_se = 0
        n_x = len(x)
        for i in range(n_x):
            p_e = pow(e[i], 2)
            ln_i = np.log(p_e)
            sum_lene = sum_lene + ln_i
            p_y = pow(y[i]-b0-b1*x[i], 2)
            sum_i = p_y / p_e
            se_i = p_y / p_e
            sum_se = sum_se + se_i

        # mle
        lnLt = -0.5*np.log(2*pi) - sum_lene / (2 * n_x)
        lnLt = -0.5*np.log(2*pi) - sum_lene / (2 * n_x) - sum_se / (2 * n_x)

        return lnLt

    def ml_function(
        self,
        residual_2,
        beta,
        sigma,
    ):
        """
        maximum likelihood function.  Time Series Analysis with Applications in R (second edition) Jonathan D.Cryer, Kung-Sil Chan p214
        Parameters
        ----------
        residual_2
        beta
        sigma

        Returns
        -------

        """

    def arch_one_degree_mle(
        self,
        residual_2,
        e,
        p=1,
    ):
        """

        Parameters
        ----------
        residual_2
        e_2
        p

        Returns
        -------

        """
        alpha = [0]*(p+1)
        ht = alpha[0] + alpha[1] * residual_2
        epsilon = np.sqrt(ht)
        epsilon = epsilon * e

        return epsilon

    def std_function(
        self,
        w,
        e,
        std,
    ):
        """std function"""
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

    def garch_one_step(
        self,
        h,
        residual_2,
        eta,
        alpha,
        e,
    ):
        """
        garch model.   single step   Applied Time Series Analysis（4th edition） Yan Wang p149
        Parameters
        ----------
        residual_2
        h
        eta
        alpha
        e

        Returns
        -------

        """
        residual_2_t = np.transpose(residual_2)
        h_t = np.matmul(eta, h)
        h_t = h_t + np.matmul(alpha, residual_2_t)
        epsilon_t = np.sqrt(h_t) * e

        return epsilon_t, h_t

    def garch(
        self,
        residual_2,
        e,
        eta,
        alpha,
        p,
        q,
    ):
        """
        Time Series Analysis  James D.Hamilton p771
        Parameters
        ----------
        residual_2:
        e:
        eta:
        alpha:
        q:
        omega:

        Returns
        -------

        """
        n_residual_2 = len(residual_2)

        epsilon = np.zeros(n_residual_2)
        h = [0]*(n_residual_2-1)
        epsilon[:q] = np.sqrt(residual_2[:q])

        for i in range(q, n_residual_2):
            h_i = h[i-p:i]
            h_i.reverse()
            residual_2_i = residual_2[i-q:i]
            residual_2_i.reverse()
            residual_2_i.append(1)  # omega
            epsilon[i], h[i] = self.garch_one_step(h_i, residual_2_i,  eta, alpha, e[i])

        return epsilon

    def garch_least_squares_estimation(
        self,
        h,
        residual_2,
        e,
        p,
        q,
    ):
        """
        Applied Time Series Analysis（4th edition） Yan Wang p149
        Parameters
        ----------
        residual_2
        e
        alpha
        q
        omega

        Returns
        -------

        """
        n_residual_2 = len(residual_2)

        # construct matrix
        xf = residual_2[q:]
        xf = np.array(xf)
        ef = e[q:]
        ef = np.array(ef)
        xf = xf / ef
        xf = np.transpose(xf)

        xp = []
        start = max(p, q)
        for i in range(start, n_residual_2):
            h_i = h[i-p:i]
            h_i.reverse()
            r_i = residual_2[i-q:i]
            r_i.reverse()
            r_i.append(1)  # omega
            xp_i = h_i + r_i
            xp.append(xp_i)
        a, R_2 = self.ordinary_least_squares(xp, xf)

        return a, R_2
