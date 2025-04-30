
import numpy as np
import pandas as pd

class STL():
    """
    Seasonal-Trend decomposition using LOESS
    Loess   circle-subseries
    y_t = T_t + S_t + R_t
    """
    def __init__(self, x):
        """
        initiate a STL model
        """
        self.frequency = 1  # the frequency of time series
        self.length = x.size  # the length of time series
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.x = x  # the original data
        self.mode = "addition"
        self.parity = None  # the parity of frequency
        self.compound_season = None
        self.u = 0
        self.mutation = None
        self.cycle_subseries = None
        self.cycle_length = 4
        self.window_length = 5
        self.t_window = 15 # need to be odd
        self.t_degress = 0 # 1 or 2
        self.s_window = 5  # need to be odd
        self.s_degress = 1 # 1 or 2
        self.robust = True # True of False
        self.degress = 1 # 1 or 2, locally-linear or locally-quadratic


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


    def _neighborhood_weight(self):
        """calculate neighborhood weights within window"""
        length = int(self.window_length / 2)
        weigth = []
        for i in range(self.window_length):
            d_i = np.absolute((i + 1) - (length + 1)) / 2
            if d_i >= 1:
                w_i = 0
            else:
                w_i = (1 - d_i ** 3) ** 3
            weigth.append(w_i)
        return weigth

    def _neighborhood_weight_x(self, xi, x):
        weigth = self._neighborhood_weight()
        v_xi = weigth * np.absolute(xi - x)/((self.window_length - 1)/2)
        return v_xi


    def polynomial_regressive(self):
        """polynomial regressive, least-squares, locally fit"""
        x = 0  # independence variable
        xi = 0
        v_i = self._neighborhood_weight_x(xi, x)
        d = self.degress  # degree
        q = 0
        a = 0
        c = 0
        g_x = a * x ** d + c

    def _cycle_subseries(self):
        """divide cycle subseries"""
        n_subseries = int(self.length / self.cycle_length)
        subseries = [[]]*n_subseries
        for i in range(n_subseries):
            for j in range(self.cycle_length):
                index = j + i * self.cycle_length
                subseries_i = self.x[index, :, :]
                subseries[i].append(subseries_i)
        self.cycle_subseries = subseries

    def inner_loop(self):
        """
        the inner loop

        Returns
        -------

        """

    def outer_loop(self):
        """
        the outer loop of stl
        Returns
        -------

        """
