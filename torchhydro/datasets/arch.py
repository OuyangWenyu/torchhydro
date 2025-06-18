""" """
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
        x
        r
        k

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

        # matrix operations, calculate the coefficient matrix.
        a, R_2 = self.ordinary_least_squares(xp, xf)

        return a, R_2

    def ordinary_least_squares(
        self,
        A,
        Y,
        b_s_a: Optional = False,
        b_a_diagonal: Optional = False,
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
        try:
            a = np.matmul(a, Y)  # parameters
        except ValueError:
            raise ValueError("matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 22 is different from 23)")

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

        if b_a_diagonal:
            n_B = B_1.shape[0]
            a_diagonal = np.zeros(n_B)
            for i in range(n_B):
                a_diagonal[i] = B_1[i, i]
            return a, R_2, a_diagonal

        return a, R_2

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
            sample_i = T[-1]
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
        else:
            dx = x  # integration
            tx = [0]*n_x  # trend
        # center
        mean_dx = np.mean(dx)
        dx_c = (dx - mean_dx).tolist()

        return dx_c, mean_dx, tx

    def ar_one_step(
        self,
        x: Optional = None,
        phi: Optional = None,
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
        e: Optional = None,
        theta: Optional = None,
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
            Q_statistic = self.LB_statistic(x, m)
        Q_critical = self.get_chi_critical(m, significance_level)

        # assumption
        H0 = True
        H1 = False

        if Q_statistic > Q_critical:   # Applied Time Series Analysis（4th edition） Yan Wang p30
            b_white_noise = H0
        else:
            b_white_noise = H1

        return b_white_noise

    def arima(
        self,
        x,
        e: Optional = None,
        phi: Optional = None,
        theta: Optional = None,
        p: int = 0,
        d: int = 0,
        q: int = 0,
        dx_c: Optional = None,
        mean_dx: Optional = None,
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
        if (dx_c is None) or (mean_dx is None) or (tx is None):
            if d > 0:
                dx_c, mean_dx, tx = self.integration(x, d)
            else:
                dx_c = x
                mean_dx = 0
                tx = np.zeros(n_x)

        # arma
        y = np.zeros(n_x)
        if q > 0:
            e_ = e[d:]
            if p > 0:
                y_ = self.arma(x=dx_c, e=e_, phi=phi, theta=theta, p=p, q=q)  # arma
            else:
                y_ = self.arma(x=dx_c, e=e_, theta=theta, q=q)  # ma
        elif p > 0:
            y_ = self.arma(x=dx_c, phi=phi, p=p)  # ar
        else:
            return dx_c, mean_dx, tx  # i

        # arma + i
        y[d:] = y[d:] + mean_dx
        y[d:] = y[d:] + y_[:]
        y = y + tx

        return y

    def arma_least_squares_estimation(
        self,
        x: Optional,
        e: Optional,
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
        a, R_2, a_diagonal = self.ordinary_least_squares(xp, xf, b_a_diagonal=True)

        # allot parameters
        phi = []
        theta = []
        if p > 0:
            phi = a[:p]
            if q > 0:
                theta = -a[p:]
        else:
            theta = -a[:]

        return phi, theta, R_2, a_diagonal

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
        x: time series
        e: white noise series
        p: degree of autoregression model
        d: degree of integration model
        q: degree of moving average model

        Returns
        -------

        """
        if d > 0:
            dx, mean_dx, tx = self.integration(x, d)
            phi, theta, R_2, a_diagonal = self.arma_least_squares_estimation(dx, e[d:], p, q)
            y_t = self.arima(x, e, phi, theta, p, d, q, dx, mean_dx, tx)
        else:
            phi, theta, R_2, a_diagonal = self.arma_least_squares_estimation(x, e, p, q)
            y_t = self.arima(x, e, phi, theta, p, d, q)

        x_residual = np.array(x) - np.array(y_t)

        return x_residual, y_t, R_2, phi, theta, a_diagonal

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

        significance_level_ = 1 - significance_level
        if significance_level_ in p:
            sl_i = p.index(significance_level_)
        else:
            raise ValueError('Significance level = 1 - ' + str(significance_level) + 'not in Significance level array.')

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
        chi_critical = self.get_chi_critical(m, significance_level)

        # assumption
        H0 = True
        H1 = False

        if LB < chi_critical:
            b_significant = H0
        else:
            b_significant = H1

        return b_significant

    def t_statistic(
        self,
        residual,
        phi,
        theta,
        a_diagonal,
    ):
        """
        significance test for parameters of ARIMA model.    t test
        Applied Time Series Analysis（4th edition） Yan Wang p78
        Parameters
        ----------
        phi
        theta
        a_diagonal
        Returns
        -------

        """
        beta = phi + theta
        m = len(beta)
        n_residual = len(residual)
        t = np.sqrt(n_residual-m)
        residual_2 = np.power(residual, 2)
        sum_residual_2 = np.sum(residual_2)
        t = t / np.sqrt(sum_residual_2)
        a_ = np.sqrt(a_diagonal)
        t = t / a_
        t_statistic = t * beta

        return t_statistic

    def get_t_statistic(
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

        significance_level_ = significance_level  # 1 -
        if significance_level_ in p:
            sl_i = p.index(significance_level_)
        else:
            raise ValueError('1 - Significance level = ' + str(significance_level_) + 'not in Significance level array.')

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
        a_diagonal,
        m,
        significance_level,
    ):
        """
        significance test for parameters of ARIMA model.    t test
        Parameters
        ----------
        phi
        theta
        a_diagonal

        Returns
        -------

        """
        n_resudual = len(residual)
        t_statistic = self.t_statistic(residual, phi, theta, a_diagonal)
        t_statistic = np.absolute(t_statistic)
        t_critical = self.get_t_statistic(n_resudual-m, significance_level)

        # assumption
        H0 = True
        H1 = False

        b_significant = []
        for i in range(len(t_statistic)):
            if t_statistic[i] > t_critical:
                b_i = H0
            else:
                b_i = H1
            b_significant.append(b_i)

        return b_significant

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

    def arch_test(
        self,
        residual,
        q,
        significance_level,
    ):
        """

        Parameters
        ----------
        residual
        q

        Returns
        -------

        """
        residual_2 = np.power(residual, 2)
        # Portmanteau Q test
        Q_statistic = self.LB_statistic(residual_2, q)
        chi_critical_Q = self.get_chi_critical(q-1, significance_level)

        # LM test
        a, R_2 = self.ar_least_squares_estimation(residual_2, q)
        residual_2_fit = self.arma(x=residual_2, e=None, phi=a, theta=None, p=q, q=0)  # ar model
        e = residual_2 - residual_2_fit
        e_2 = np.power(e, 2)
        lm_statistic = self.LM_statistic(residual_2, q, e_2)
        chi_critical_lm = self.get_chi_critical(q-1, significance_level)

        # assumption
        H0 = True
        H1 = False

        # test
        if Q_statistic > chi_critical_Q:
            b_arch_Q = H0
        else:
            b_arch_Q = H1

        if lm_statistic > chi_critical_lm:
            b_arch_lm = H0
        else:
            b_arch_lm = H1

        return b_arch_Q, b_arch_lm

    def arch_one_step(
        self,
        residual_2,
        alpha,
        e,
    ):
        """
        arch model.   single step
        Applied Time Series Analysis（4th edition） Yan Wang p147
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
        residual_2_t = np.transpose(residual_2)
        h_t = np.matmul(alpha, residual_2_t)
        epsilon_t = np.sqrt(h_t) * e

        return epsilon_t

    def arch(
        self,
        residual_2,
        e,
        alpha,
        q,
    ):
        """
        Time Series Analysis  James D.Hamilton p762
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
        epsilon = np.zeros(n_residual_2)
        epsilon[:q] = np.sqrt(residual_2[:q])

        for i in range(q, n_residual_2):
            residual_2_i = residual_2[i-q:i]
            residual_2_i.reverse()
            residual_2_i.append(1)  # omega
            epsilon[i] = self.arch_one_step(residual_2_i, alpha, e[i])

        return epsilon

    def arch_least_squares_estimation(
        self,
        residual_2,
        e,
        q,
    ):
        """

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

        # construct matrix
        xf = residual_2[q:]
        xf = np.array(xf)
        ef = e[q:]
        ef = np.array(ef)
        xf = xf / ef
        xf = np.transpose(xf)

        xp = []
        for i in range(q, n_residual_2):
            xp_i = residual_2[i-q:i]
            xp_i.reverse()
            xp_i.append(1)  # omega
            xp.append(xp_i)
        a, R_2 = self.ordinary_least_squares(xp, xf)

        return a, R_2

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
