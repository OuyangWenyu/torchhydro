
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

    def _get_n_range(self, t):
        """
        get the number of time period when calculate season item at specified time period
        circle-subseries
        """
        t_start = 1
        t_end = self.frequency
        n = max(n, n <= self.length/self.frequency)  # todo；
        t_i = t + n * self.frequency
        n_range = 0

        return n_range

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

    def deseasonalizing(self):
        """detach season item"""


    def _cycle_subseries(self, x):
        """
        divide cycle subseries
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

    def _get_robustness_weights(self, data):
        """calculate robustness weights, """
        data = np.absolute(data, axis=2)
        h = np.median(data, axis=2) * 6
        u = data / h  # np.divide()
        r = np.where(
            u >= 1,
            0,
            (1 - u ** 2) ** 2
        )
        return r

    def _residuals(self):
        """get the residuals of series"""
        self.residuals = self.x - self.trend - self.season

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

    def _neighborhood_weight(self):
        """calculate neighborhood weights within window"""
        degree = 2
        length = int(self.window_length / 2)
        weigth = []
        for i in range(self.window_length):
            d_i = np.absolute((i + 1) - (length + 1)) / (length + 1)
            w_i = self.weight_function(d_i, degree)
            weigth.append(w_i)
        return weigth

    def _neighborhood_weight_x(self, xi, x):
        """"""
        weight = self._neighborhood_weight()
        v_xi = weight * np.absolute(xi - x)/((self.window_length - 1)/2)
        return v_xi

    def robustness_neighborhood_weights_x(self, v_xi):
        """robustness neighborhood weights"""
        # todo: ki
        ki = 0
        rho = 1/ki
        rw_xi = rho*v_xi
        return rw_xi

    def polynomial_regressive(self):
        """
        polynomial regressive, least-squares, locally fit
        1 degree linear or 2 degree quadratic polynomial
        minimize the square summation of weight residual error -> parameters of polynomial -> estimate value

        Returns
        -------

        """
        x = 0  # independence variable
        xi = 0
        v_i = self._neighborhood_weight_x(xi, x)
        d = self.degree  # degree
        q = 0
        a = 0
        c = 0
        g_x = a * x ** d + c

    def weight_least_squares(self, x, y):
        """
        least squares estimate
        numerical analysis page 67-71.
        degree = 1
        Parameters
        ----------
        x, independent variable
        y, dependent variable
        Returns
        -------

        """
        length = len(x)
        x1 = [1]*length
        At = np.array([x1, x])
        A = np.transpose(At)
        Y = y
        weight = self._neighborhood_weight()  # todo: use robustness_neighborhood_weights
        W = np.diag(weight)
        B = np.matmul(At, W)
        B = np.matmul(B, A)
        B_1 = np.linalg.inv(B)
        a = np.matmul(B_1, At)
        a = np.matmul(a, W)
        a = np.matmul(a, Y)

        i = int(length/2 + 1)
        yy = a[0] + a[1] * x[i]

        return yy

    def robustness_weights(self):
        """robustness weights"""
        length = self.length
        x = list(range(length))
        y = self.x  # todo
        y_ = self.weight_least_squares(x, y)
        error = y - y_  # residual error
        abs_error = np.absolute(error)
        s = np.median(abs_error)


    def loess(self, width, x):
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
        xx = list(range(width))
        start = int(width / 2 + 1)
        k = int(width / 2)
        result = [0] * length
        result[:start] = x[:start]
        for i in range(start, length-start+1):
            y = x[i-k:i+k+1]
            y_i = self.weight_least_squares(xx, y)
            result[i] = y_i
        result[length - start + 1:] = x[length - start + 1:]
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

    def inner_loop(self, y, trend):
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
        ns = 5
        nl = 3
        nt = 7
        k = 7

        y = np.array(y) - np.array(trend)

        subseries = self._cycle_subseries(y)
        cycle = []
        for i in range(self.cycle_length):
            subseries_i = subseries[i]
            extend_subseries_i = self.loess(ns, subseries_i)
            cycle.append(extend_subseries_i)

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
        # lowf = self.moving_average_smoothing(k, cycle)
        # lowf = self.moving_average_smoothing(k, lowf)
        # lowf = self.moving_average_smoothing(3, lowf)
        # lowf = self.loess(nl, lowf)

        season = cycle - lowf

        trend = y - season

        trend = self.loess(nt, trend)

        return trend, season

    def rho_weight(self, residuals):
        rho = [0]*self.length
        for i in range(self.length):
            rho[i] = self.weight_function(residuals[i], 2)
        return rho


    def outer_loop(self):
        """
        the outer loop of stl
        Returns
        -------

        adjust robustness weights
        no
        """
        no = 10
        ni = 1
        trend = [0]*self.length
        season = [0]*self.length
        trend_i0 = []
        season_i0 = []
        trend_ij0 = []
        season_ij0 = []
        for i in range(no):
            if i == 0:
                trend_i0 = trend
                season_i0 = season
            for j in range(ni):
                if j == 0:
                    trend_ij0 = trend_i0[:]
                    season_ij0 = season_i0[:]
                trend_i, season_i = self.inner_loop(trend_ij0)
                trend_ij0 = trend_i[:]
                season_ij0 = season_i[:]
            trend_i0 = trend_ij0[:]
            season_i0 = season_ij0[:]
            residuals = self.x - trend_i0 - season_i0
            abs_residuals = np.absolute(residuals)
            h = 6 * np.median(abs_residuals)
            abs_residuals_h = abs_residuals / h
            rho_weight = self.rho_weight(abs_residuals_h)


