
import numpy as np
import pandas as pd



class STL():
    """
    Seasonal-Trend decomposition using LOESS
    """
    def __init__(self):
        """
        initiate a STL model
        """
        self.f = 6  # the frequence of time series
        self.l = 4 * 30  # the length of time series
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.x = None  # the original data
        self.mode = "addition"

    def get_scope_t(self, t):
        """get the scope of time period t"""
        t_start = (self.f+self.l)/2
        t_end = 1 - (self.f-self.l)/2
        t_scope = [t_start, t_end]
        return t_scope



