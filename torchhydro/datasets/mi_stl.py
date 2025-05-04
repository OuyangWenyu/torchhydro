
import numpy as np
import pandas as pd

class STL():
    """
    Seasonal-Trend decomposition using LOESS
    Loess   circle-subseries
    y_t = T_t + S_t + R_t
    todo: unify the data type.
    """
    def __init__(self, x):
    # def __init__(self):
        """
        initiate a STL model
        """
        self.x = x  # the original data
        # self.x = None
        self.frequency = 1  # the frequency of time series
        self.length = len(x)  # the length of time series
        # self.length = None
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.mode = "addition"
        self.parity = None  # the parity of frequency
        self.compound_season = None
        self.u = 0
        self.mutation = None
        self.cycle_subseries = None
        self.cycle_length = 365
        self.window_length = 5  # window width, span
        self.t_window = 15 # need to be odd
        self.t_degree = 0 # 1 or 2
        self.s_window = 5  # need to be odd
        self.s_degree = 1 # 1 or 2
        self.robust = True # True of False
        self.degree = 1 # 1 or 2, locally-linear or locally-quadratic

        self._get_parity()


    def _get_parity(self):
        """get the parity of frequency"""
        if self.frequency % 2 == 0:
            self.parity = "even"
        else:
            self.parity = "odd"

    def _get_t_range_odd(self, t):
        """get the range of time period t"""
        t_start = (self.frequency+self.length)/2
        t_end = self.length - (self.frequency-1)/2
        t_range = [t_start, t_end]
        return t_range

    def _get_t_range_even(self, t):
        """get the range of time period t"""
        t_start = self.frequency/2 + 1
        t_end = self.length - self.frequency/2
        t_range = [t_start, t_end]
        return t_range

    def _get_data(self, t_range):
        """get the data of specified time range"""
        data = self.x[t_range[0]:t_range[1], :, :]
        return data

    def _trend_t_odd(self, t):
        """
        get the trend of specified time period
        """
        t_range = self._get_t_range_odd(t)
        data = self._get_data(t_range)
        trend_t = np.sum(data, axis=2)/self.frequency

        return trend_t

    def _trend_t_even(self, t):
        """return trend item when frequency is even"""
        t_range = self._get_t_range_even(t)
        data = self._get_data(t_range)
        trend_t = (0.5 * (data[0, :, :] + data[-1, :, :]) + np.sum(data[1:-2, :, :], axis=2))/self.frequency

        return trend_t

    def _trend_odd(self):
        """get the trend of series, frequency is odd"""
        trend = []
        for i in range(self.length):
            trend_i = self._trend_t_odd(i)
            trend.append(trend_i)
        self.trend = np.array(trend)

    def _trend_even(self):
        """get the trend of series, frequency is even"""
        trend = []
        for i in range(self.length):
            trend_i = self._trend_t_even(i)
            trend.append(trend_i)
        self.trend = np.array(trend)

    def _compound_season(self):
        """get the compound season, detrending"""
        c_s = self.x - self.trend
        self.compound_season = c_s

    def _get_t_range_season(self, t, n):
        """get the range of time period"""
        t_start = 1
        t_end = self.frequency

        t_range = [t_start, t_end]
        return t_range

    def _get_c_s_data(self, t_range):
        data = self.compound_season[t_range[0]:t_range[1], :, :]
        return data

    def _seasonal_t(self, t):
        """the season of specified time period """
        n_range = self._get_n_range(t)
        t_range = self._get_t_range_season(t, n_range)  # todo:
        data = self._get_c_s_data(self._get_c_s_data(t_range))  # todo:
        season_t = np.sum(data, axis=2)/self.frequency
        # circle-subseries smoothing, Loess method

        return season_t

    def _season(self):
        """get the season of series"""
        season = []
        for i in range(self.length):
            season_i = self._seasonal_t(i)
            season.append(season_i)
        self.season = season

    def _cycle_subseries(self, x):
        """
        divide series into cycle subseries
        4 year date, (1990,1991,1992,1993). 1992 is leap year.
        cycle_length = 365
        reject 1992-02-29
        Returns
        -------

        """
        n_subseries = self.cycle_length
        len_subseries = int(self.length / n_subseries)  # 4
        subseries = [[]] * n_subseries
        subseries_i = [0] * len_subseries
        for i in range(n_subseries):
            for j in range(len_subseries):
                index = i + j * n_subseries
                # subseries_ij = self.x[index, :, :]
                subseries_ij = x[index]
                subseries_i[j] = subseries_ij
            subseries[i] = subseries_i[:]
        # self.cycle_subseries = subseries
        return subseries

    def _extend_subseries(self, subseries):
        """extend cycle subseries"""
        len_subseries = int(self.length/self.cycle_length)
        len_extend_subseries = len_subseries + 2
        extend_subseries = [[]] * self.cycle_length
        extend_subseries_i = [0] * len_extend_subseries
        for i in range(self.cycle_length):
            extend_subseries_i[0] = subseries[i][0]
            extend_subseries_i[1:len_extend_subseries-1] = subseries[i][:]
            extend_subseries_i[-1] = subseries[i][-1]
            extend_subseries[i] = extend_subseries_i[:]
        return extend_subseries

    def _de_extend_subseries(self, extend_subseries):
        """remove extend cycle subseries"""
        len_subseries = int(self.length/self.cycle_length)
        len_extend_subseries = len_subseries + 2
        subseries = [[]] * self.cycle_length
        subseries_i = []
        for i in range(self.cycle_length):
            subseries_i = extend_subseries[i][1:-1]
            subseries[i] = subseries_i[:]
        return subseries

    def _recover_series(self, subseries):
        """recover series from cycle subseries"""
        n_subseries = self.cycle_length
        len_subseries = int(self.length / n_subseries)
        series = []
        series_j = [0]*n_subseries
        for j in range(len_subseries):
            for i in range(n_subseries):
                series_ji = float(subseries[i][j])
                series_j[i] = series_ji
            series = series + series_j[:]
        return series

    def weight_function(self, u, d: int = 2,):
        """
        quadratic/cubic weight  function
        Parameters
        ----------
        u
        d, int, degree, 2 or 3.

        Returns
        -------

        """
        if np.absolute(u) < 1:
            return (1 - u ** d) ** d
        else:
            return 0

    def _neighborhood_weight(self, width):
        """calculate neighborhood weights within window"""
        degree = 2
        # length = int(self.window_length / 2)
        length = int(width / 2)
        weight = []
        for i in range(width):
            d_i = np.absolute((i + 1) - (length + 1)) / (length + 1)
            w_i = self.weight_function(d_i, degree)
            weight.append(w_i)
        return weight

    def _neighborhood_weight_x(self, xi, x):
        """"""
        width = len(x)
        weight = self._neighborhood_weight(width)
        v_xi = weight * np.absolute(xi - x)/((width - 1)/2)
        return v_xi

    def weight_least_squares(self, x, y, rho_weight):
        """
        polynomial regressive, least-squares, locally fit
        1 degree linear or 2 degree quadratic polynomial
        minimize the square summation of weight residual error -> parameters of polynomial -> estimate value
        least squares estimate
        numerical analysis page 67-71.
        degree = 1
        Parameters
        ----------
        x, independent variable
        y, dependent variable
        rho_weight, robustness weights
        Returns
        -------

        """
        length = len(x)
        x1 = [1]*length
        At = np.array([x1, x])
        A = np.transpose(At)
        Y = y
        weight = self._neighborhood_weight(length)
        weight = np.multiply(weight, rho_weight)
        W = np.diag(weight)
        B = np.matmul(At, W)
        B = np.matmul(B, A)
        try:
            B_1 = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        a = np.matmul(B_1, At)
        a = np.matmul(a, W)
        a = np.matmul(a, Y)

        i = int(length/2 + 1)
        yy = a[0] + a[1] * x[i]

        return yy

    def loess(
        self,
        width,
        x,
        rho_weight: list = None,
    ):
        """
        loess
        robustnes
        calculate loess curve for a series
        Parameters
        ----------
        width, window length
        x, series need to smoothing.
        Returns
        -------

        """
        length = len(x)
        if rho_weight is None:
            rho_w = [1]*length
        else:
            rho_w = rho_weight
        xx = list(range(width))
        start = int(width / 2)
        k = int(width / 2)
        result = [0] * length
        result[:start] = x[:start]
        for i in range(start, length-start):
            y = x[i-k:i+k+1]
            rw_i = rho_w[i-k:i+k+1]
            y_i = self.weight_least_squares(xx, y, rw_i)
            result[i] = y_i
        result[length - start:] = x[length - start:]
        return result

    def moving_average_smoothing(self, width, x):
        """moving average smoothing """
        length = len(x)
        start = int(width/2 + 1)
        k = int(width/2)
        result = [0]*length
        result[:start] = x[:start]
        for i in range(start, length-start+1):
            x_i = np.sum(x[i-k:i+k+1])/width
            result[i] = x_i
        result[length-start+1:] = x[length-start+1:]
        return result

    def inner_loop(
        self,
        y,
        trend,
        sub_rho_weight,
        rho_weight,
    ):
        """
        the inner loop
        Returns
        -------

        trend
        calculate seasonal item
        ni
        ns
        nl
        nt
        """
        ns = 5  # q  35
        nl = 3
        nt = 5
        k = 5
        # detrending
        y = np.array(y) - np.array(trend)
        # 2 cycle-subseries smoothing
        subseries = self._cycle_subseries(y)
        extend_subseries = self._extend_subseries(subseries)
        cycle = []
        for i in range(self.cycle_length):
            extend_subseries_i = extend_subseries[i]
            sub_rho_weight_i = sub_rho_weight[i]
            extend_subseries_i = self.loess(ns, extend_subseries_i, sub_rho_weight_i)  # q = ns, d = 1
            cycle.append(extend_subseries_i)

        # low-pass filtering of smoothed cycle-subseries
        lowf = []
        for i in range(self.cycle_length):
            cycle_i = cycle[i]
            lowf_i = self.moving_average_smoothing(k, cycle_i)
            lowf.append(lowf_i)
        for i in range(self.cycle_length):
            lowf_i = lowf[i]
            lowf_i = self.moving_average_smoothing(k, lowf_i)
            lowf[i] = lowf_i[:]
        for i in range(self.cycle_length):
            lowf_i = lowf[i]
            lowf_i = self.moving_average_smoothing(3, lowf_i)
            lowf[i] = lowf_i[:]
        for i in range(self.cycle_length):
            lowf_i = lowf[i]
            lowf_i = self.loess(nl, lowf_i)
            lowf[i] = lowf_i[:]

        # detrending if smoothed cycle-subseries
        cycle = np.array(cycle)
        lowf = np.array(lowf)
        extend_subseason = cycle - lowf
        subseason = self._de_extend_subseries(extend_subseason)
        season = self._recover_series(subseason)

        # deseasonalizing
        trend = y - season
        # 6 Trend Smoothing
        trend = self.loess(nt, trend, rho_weight)

        return trend, season

    def rho_weight(self, u):
        """robustness weights"""
        rho = [0]*self.length
        for i in range(self.length):
            rho[i] = self.weight_function(u[i], 2)
        return rho

    def extend_cycle_sub_rho_weight(self, rho_weight):
        """cycle sub-robustness weights for robustness cycle-subseries smoothing"""
        sub_rho_weight = self._cycle_subseries(rho_weight)
        extend_sub_rho_weight = self._extend_subseries(sub_rho_weight)
        return extend_sub_rho_weight

    def outer_loop(self):
        """
        the outer loop of stl
        Returns
        -------

        adjust robustness weights
        no
        """
        no = 3
        ni = 1
        trend = [0]*self.length
        season = [0]*self.length
        trend_ij0 = []
        season_ij0 = []
        rho_weight = [1]*self.length
        # outer loop
        for i in range(no):
            if i == 0:
                trend_i0 = trend
                season_i0 = season
            extend_sub_rho_weight = self.extend_cycle_sub_rho_weight(rho_weight)
            # inner loop
            for j in range(ni):
                if j == 0:
                    trend_ij0 = trend_i0[:]
                    # season_ij0 = season_i0[:]
                trend_i, season_i = self.inner_loop(self.x, trend_ij0, extend_sub_rho_weight, rho_weight)
                trend_ij0 = trend_i[:]
                season_ij0 = season_i[:]

            trend_i0 = trend_ij0[:]
            season_i0 = season_ij0[:]
            residuals = np.array(self.x) - np.array(trend_i0) - np.array(season_i0)
            abs_residuals = np.absolute(residuals)
            h = 6 * np.median(abs_residuals)
            abs_residuals_h = abs_residuals / h
            rho_weight = self.rho_weight(abs_residuals_h)

        return trend_i0, season_i0, residuals


    def season_post_smoothing(self, season):
        """post-smoothing of the seasonal"""
        ns = 7
        season = self.loess(ns, season)
        return season


