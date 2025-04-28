
import numpy as np
import pandas as pd



class STL():
    """
    Seasonal-Trend decomposition using LOESS
    """
    def __init__(self, x):
        """
        initiate a STL model
        """
        self.frequency = 6  # the frequency of time series
        self.length = 4 * 30  # the length of time series
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.x = x  # the original data
        self.mode = "addition"
        self.parity = None

    def _get_parity(self):
        """get the parity of frequency"""
        if self.frequency % 2 == 0:
            self.parity = 0  # even
        else:
            self.parity = 1  # odd

    def _get_t_range(self, t):
        """get the scope of time period t"""
        t_start = (self.frequency+self.length)/2
        t_end = 1 - (self.frequency-self.length)/2
        t_range = [t_start, t_end]
        return t_range

    def _get_data(self, t_range):
        """get the data of specified time range"""
        data = self.x[t_range[0]:t_range[1], :, :]
        return data

    def _trend_t(self, t):
        """
        get the trend of specified time period
        Parameters
        ----------
        x

        Returns
        -------

        """
        # length = x.shape[0]   # (time,basin,features)
        t_range = self._get_t_range(t)
        data = self._get_data(t_range)
        # if length != self.length:
        #     raise ValueError("Length mismatch")
        trend_t = np.sum(data, axis=2)/self.frequency

        return trend_t

    def _trend(self):
        """get the trend of series"""
        trend = []
        for i in range(self.length):
            trend_i = self._trend_t(i)
            trend.append(trend_i)
        self.trend = np.array(trend)

    def _compound_season(self):
        """get the compound season"""
        c_s = self.x - self.trend
        return c_s

    def _get_n_range(self, t):
        """get the number of time period"""
        t_start = 1
        t_end = self.frequency
        n = 0
        t_i = t + n * self.frequency

    def _seasonal_t(self, t):
        """the season of specified time period """


    def _season(self):
        """get the season of series"""
        c_s = self.x - self.trend


    def _residuals(self):
        """get the residuals of series"""
        self.residuals = self.x - self.trend - self.season
