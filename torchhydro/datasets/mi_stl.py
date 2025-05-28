"""
Author: Lili Yu
Date: 2025-05-10 18:00:00
LastEditTime: 2025-05-10 18:00:00
LastEditors: Lili Yu
Description:
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict

from hydroutils import hydro_time

from torchhydro.datasets.data_sources import data_sources_dict

class STL(object):
    """
    Seasonal-Trend decomposition using LOESS
    loess     circle-subseries    low pass filter
    y_t = T_t + S_t + R_t
    """
    def __init__(
        self,
        frequency,
        cycle_length,
        no: int = 1,
        ni: int = 1,
        ns: int = 33,  # todo:
        nl: int = 365,
        nt: int = 421,
        n_p: int = 365,
        degree: int = 1,
    ):
        """
        initiate a STL model
        """
        self.x = None  # the original data
        self.frequency = frequency  # the frequency of time series
        self.length = None  # the length of time series
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.mode = "addition"
        self.parity = None  # the parity of frequency
        self.mutation = None
        self.cycle_subseries = None
        self.cycle_length = cycle_length
        self.n_cycle = None
        self.extend_x = None
        self.no = no  # window width, span
        self.ni = ni  # need to be odd
        self.ns = ns  # odd >=7
        self.nl = nl  #
        self.nt = nt  # 1 or 2
        self.n_p = n_p  #
        self.degree = degree  # 1 or 2, locally-linear or locally-quadratic

        self._get_parity()

    def _get_parity(self):
        """get the parity of frequency"""
        if self.frequency % 2 == 0:
            self.parity = "even"
        else:
            self.parity = "odd"

    def reset_stl(
        self,
        x,
    ):
        self.x = x
        self.length = len(x)
        self.n_cycle = self.length / self.cycle_length
        self.extend_x = self._extend_original_series(self.x)
        self.cycle_subseries = None
        self.trend = None
        self.season = None
        self.residuals = None

        self._get_parity()

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
        len_subseries = int(self.length / n_subseries)
        subseries = [[]] * n_subseries
        subseries_i = [0] * len_subseries
        for i in range(n_subseries):
            for j in range(len_subseries):
                index = i + j * n_subseries
                subseries_ij = x[index]
                subseries_i[j] = subseries_ij
            subseries[i] = subseries_i[:]
        pd_subseries = pd.DataFrame({
            "subseries_" + str(i): subseries[i] for i in range(n_subseries)
        })
        pd_subseries.index.name = "time"
        # file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_subseries.csv"
        file_name = r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ystl/pet_subseries.csv"
        pd_subseries.to_csv(file_name, sep=" ")
        return subseries

    def _extend_subseries(self, subseries):
        """
        extend cycle subseries, extend one point in start and end of subseries separately.
        Parameters
        ----------
        subseries

        Returns
        -------

        """
        len_subseries = int(self.length/self.cycle_length)
        len_extend_subseries = len_subseries + 2
        extend_subseries = [[]] * self.cycle_length
        extend_subseries_i = [0] * len_extend_subseries
        for i in range(self.cycle_length):
            extend_subseries_i[0] = subseries[i][0]
            extend_subseries_i[1:len_extend_subseries-1] = subseries[i][:]
            extend_subseries_i[-1] = subseries[i][-1]
            extend_subseries[i] = extend_subseries_i[:]
        pd_extend_subseries = pd.DataFrame({
            "ext_subser_" + str(i): extend_subseries[i] for i in range(self.cycle_length)
        })
        pd_extend_subseries.index.name = "time"
        # file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_extend_subseries.csv"
        file_name = r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ystl/pet_extend_subseries.csv"
        pd_extend_subseries.to_csv(file_name, sep=" ")
        return extend_subseries

    def _de_extend_subseries(self, extend_subseries):
        """remove extend cycle subseries"""
        len_subseries = int(self.length/self.cycle_length)
        subseries = [[]] * self.cycle_length
        subseries_i = []
        for i in range(self.cycle_length):
            subseries_i = extend_subseries[i][1:-1]
            subseries[i] = subseries_i[:]
        return subseries

    def _recover_series(self, subseries):
        """
        recover series from extend cycle subseries,
        Parameters
        ----------
        subseries

        Returns
        -------

        """
        n_subseries = self.cycle_length
        len_subseries = int(self.length / n_subseries) + 2
        series = []
        series_i = [0]*n_subseries
        for i in range(len_subseries):
            for j in range(n_subseries):
                series_ji = float(subseries[j][i])
                series_i[j] = series_ji
            series = series + series_i[:]

        return series

    def _extend_original_series(self, series):
        """
        extend original series
        Parameters
        ----------
        series

        Returns
        -------

        """
        series_start = series[:self.cycle_length]
        series_end = series[-self.cycle_length:]
        extend_series = series_start + series + series_end
        return extend_series

    def weight_function(
        self,
        u: float,
        degree: int = 2,
    ):
        """
        quadratic/cubic weight  function
        Parameters
        ----------
        u: float
        degree, int, degree, 2 or 3.

        Returns
        -------
        weight
        """
        if np.absolute(u) < 1:
            weight = (1 - u ** degree) ** degree
        else:
            weight = 0

        return weight

    def _neighborhood_weight(
        self,
        width: int,
        i_focal: int,
        degree: int = 3,
    ):
        """
        calculate neighborhood weights within window
        Parameters
        ----------
        width: int, odd, window width.
        degree: int, 2 or 3, the degree of weight function.

        Returns
        -------
        weight: list, neighborhood weights
        """
        k = int(width / 2)
        weight = [0]*width
        # max distance
        if i_focal < k:
            q = (width - 1) - i_focal
        else:
            q = i_focal

        for i in range(width):
            d_i = abs(i - i_focal)  # focal point of the window
            u_i = d_i / q
            weight[i] = self.weight_function(u_i, degree)

        return weight

    def weight_least_squares_fit(
        self,
        x,
        y,
        i_focal: int,
        degree: int = 1,
        rho_weight: list = None,
    ):
        """
        polynomial regressive, least-squares, locally fit
        minimize the square summation of weight residual error -> parameters of polynomial -> estimate value
        least squares estimate
        numerical analysis page 67-71.
        degree = 1
        Parameters
        ----------
        x: independent variable
        y: dependent variable
        degree: int, the number of polynomial degree. 0 degree for constant, 1 degree for linear, 2 degree for quadratic
         polynomial.
        rho_weight, robustness weights.
        Returns
        -------

        """
        length = len(x)

        if rho_weight is None:
            rho_weight = [1]*length

        # construct matrix
        At = np.ones((1, length))
        for i in range(1, degree+1):
            at_i = np.power(x, i)
            At = np.concatenate([At, [at_i]], axis=0)
        A = np.transpose(At)
        Y = y
        neighbor_weight = self._neighborhood_weight(length, i_focal)
        try:
            weight = np.multiply(neighbor_weight, rho_weight)
        except ValueError:
            raise ValueError("operands could not be broadcast together with shapes (7,) (6,)")
        W = np.diag(weight)

        # matrix operations, calculate the coefficient matrix.
        B = np.matmul(At, W)
        B = np.matmul(B, A)
        try:
            B_1 = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Singular matrix")
        a = np.matmul(B_1, At)
        a = np.matmul(a, W)
        a = np.matmul(a, Y)

        # regressive, estimate value
        yy = 0
        for i in range(degree+1):
            yy = yy + a[i] * np.power(x[i_focal], i)

        return yy

    def loess(
        self,
        width: int,
        # xi,
        x,
        degree: int = 1,
        rho_weight: list = None,
    ):
        """
        loess, locally estimated scatterplot smoothing.
        calculate loess curve for a series
        Parameters
        ----------
        width, int, odd, window width.
        x, series need to smoothing.
        degree: int, the number of polynomial degree.
        rho_weight: list, robustness weights.
        Returns
        -------
        result, the smoothed series by loess filter.
        """
        length = len(x)

        if rho_weight is None:
            rho_w = [1]*length
        else:
            rho_w = rho_weight

        xx = list(range(width))
        k = int(width / 2)
        result = [0] * length
        for i in range(length):
            # start
            if i < k:
                m = k - i
                y1 = [x[i]]
                y2 = x[i+1:i+k+1]
                rw_i1 = [rho_w[i]]
                rw_i2 = rho_w[i+1:i+k+1]
                # xi1 = [xi[i]]
                # xi2 = xi[i+1:i+k+1]
                if m == k:
                    y0 = []
                    y3 = y2[:]
                    y3.reverse()
                    rw_i0 = []
                    rw_i3 = rw_i2[:]
                    rw_i3.reverse()
                    # xi0 = []
                    # xi3 = xi2[:]
                    # xi3.reverse()
                elif (m > 1) and (m < k):
                    y0 = x[:i]
                    y3 = y2[-m:]
                    y3.reverse()
                    rw_i0 = rho_w[:i]
                    rw_i3 = rw_i2[-m:]
                    rw_i3.reverse()
                    # xi0 = xi[:i]
                    # xi3 = xi2[-m:]
                    # xi3.reverse()
                else:
                    y0 = x[:i]
                    y3 = [y2[-m]]
                    rw_i0 = rho_w[:i]
                    rw_i3 = [rw_i2[-m]]
                    # xi0 = xi[:i]
                    # xi3 = xi2[-m:]
                y = y3 + y0 + y1 + y2
                rw_i = rw_i3 + rw_i0 + rw_i1 + rw_i2
                # xxi = xi3 + xi0 + xi1 + xi2
                # m = k - i
                # y1 = [x[i]]  # focal point of the window
                # y2 = x[i+1:i+k+1]
                # rw_i1 = [rho_w[i]]
                # rw_i2 = rho_w[i+1:i+k+1]
                # if m == k:
                #     y0 = []
                #     y3 = x[i+k+1:i+2*k+1]
                #     rw_i0 = []
                #     rw_i3 = rho_w[i+k+1:i+2*k+1]
                # elif (m > 1) and (m < k):
                #     y0 = x[:i]
                #     y3 = x[i+k+1:i+k+1+m]
                #     rw_i0 = x[:i]
                #     rw_i3 = rho_w[i+k+1:i+k+1+m]
                # else:
                #     y0 = x[:i]
                #     y3 = [x[i+k+m]]
                #     rw_i0 = rho_w[:i]
                #     rw_i3 = [rho_w[i+k+m]]
                # y = y0 + y1 + y2 + y3
                # rw_i = rw_i0 + rw_i1 + rw_i2 + rw_i3
                # i_focal = k - m
            # end
            elif i > (length-1) - k:
                m = k + i - (length-1)
                y1 = [x[i]]
                y0 = x[i-k:i]
                rw_i1 = [rho_w[i]]
                rw_i0 = rho_w[i-k:i]
                # xi1 = [xi[i]]
                # xi0 = xi[i-k:i]
                if m == k:
                    y2 = []
                    y3 = y0[:]
                    y3.reverse()
                    rw_i2 = []
                    rw_i3 = rw_i0[:]
                    rw_i3.reverse()
                    # xi2 = []
                    # xi3 = xi0[:]
                    # xi3.reverse()
                elif (m > 1) and (m < k):
                    y2 = x[i+1:]
                    y3 = y0[:m]
                    y3.reverse()
                    rw_i2 = rho_w[i+1:]
                    rw_i3 = rw_i0[:m]
                    rw_i3.reverse()
                    # xi2 = xi[i+1:]
                    # xi3 = xi0[:m]
                    # xi3.reverse()
                else:
                    y2 = x[i+1:]
                    y3 = [y0[m-1]]
                    rw_i2 = rho_w[i+1:]
                    rw_i3 = [rw_i0[m-1]]
                    # xi2 = xi[i+1:]
                    # xi3 = [xi0[m-1]]
                y = y0 + y1 + y2 + y3
                rw_i = rw_i0 + rw_i1 + rw_i2 + rw_i3
                # xxi = xi0 + xi1 + xi2 + xi3
                # m = k + i - (length-1)
                # y1 = [x[i]]
                # y0 = x[i-k:i]
                # rw_i1 = [rho_w[i]]
                # rw_i0 = rho_w[i-k:i]
                # if m == k:
                #     y2 = []
                #     y3 = x[i-2*k:i-k]
                #     rw_i2 = []
                #     rw_i3 = rho_w[i-2*k:i-k]
                # elif (m > 1) and (m < k):
                #     y2 = x[i+1:]
                #     y3 = x[i-k-m:i-k]
                #     rw_i2 = rho_w[i+1:]
                #     rw_i3 = rho_w[i-k-m:i-k]
                # else:
                #     y2 = x[i+1:]
                #     y3 = [x[i-k-m]]
                #     rw_i2 = rho_w[i+1:]
                #     rw_i3 = [rho_w[i-k-m]]
                # y = y3 + y0 + y1 + y2
                # rw_i = rw_i3 + rw_i0 + rw_i1 + rw_i2
                # i_focal = (width - 1) - (k - m)
            # middle
            else:
                y = x[i-k:i+k+1]
                rw_i = rho_w[i-k:i+k+1]
                # xxi = xi[i-k:i+k+1]
                # i_focal = k
            # fit
            i_focal = k
            y_i = self.weight_least_squares_fit(xx, y, i_focal, degree, rw_i)

            result[i] = y_i

            # print(xxi)
            # print(y)

        return result

    def _negative(self, x):
        negative_x = [- xi for xi in x]
        return negative_x

    def moving_average_smoothing(
        self,
        width,
        # xi,
        x,
        negative: bool = False,
    ):
        """
        moving average smoothing
        Parameters
        ----------
        width: int, window width.
        x: series need to smoothing.
        negative: bool, positive or negative.
        Returns
        -------
        result, the smoothed series by moving average filter.
        """
        length = len(x)
        k = int(width/2)
        result = [0]*length
        for i in range(length):
            # start
            if i < k:
                # m = k - i
                # xx1 = [x[i]]
                # xx2 = x[i + 1:i + k + 1]
                # # xi1 = [xi[i]]
                # # xi2 = xi[i + 1:i + k + 1]
                # if m == k:
                #     xx0 = []
                #     xx3 = xx2[:]
                #     xx3.reverse()
                #     # xi0 = []
                #     # xi3 = xi2[:]
                #     # xi3.reverse()
                # elif (m > 1) and (m < k):
                #     xx0 = x[:i]
                #     xx3 = xx2[-m:]
                #     xx3.reverse()
                #     # xi0 = xi[:i]
                #     # xi3 = xi2[-m:]
                #     # xi3.reverse()
                # else:
                #     xx0 = x[:i]
                #     xx3 = [xx2[-m]]
                #     # xi0 = xi[:i]
                #     # xi3 = [xi2[-m]]
                # if negative:
                #     xx3 = self._negative(xx3)
                # xx = xx3 + xx0 + xx1 + xx2
                # # xxi = xi3 + xi0 + xi1 + xi2
                xx = x[:i+k+1]
                n_xx = i+k+1
            # end
            elif i > (length-1) - k:
                # m = k + i - (length-1)
                # xx1 = [x[i]]
                # xx0 = x[i-k:i]
                # # xi1 = [xi[i]]
                # # xi0 = xi[i-k:i]
                # if m == k:
                #     xx2 = []
                #     xx3 = xx0[:]
                #     xx3.reverse()
                #     # xi2 = []
                #     # xi3 = xi0[:]
                #     # xi3.reverse()
                # elif (m > 1) and (m < k):
                #     xx2 = x[i+1:]
                #     xx3 = xx0[:m]
                #     xx3.reverse()
                #     # xi2 = xi[i+1:]
                #     # xi3 = xi0[:m]
                #     # xi3.reverse()
                # else:
                #     xx2 = x[i+1:]
                #     xx3 = [xx0[m-1]]
                #     # xi2 = xi[i+1:]
                #     # xi3 = [xi0[m-1]]
                # if negative:
                #     xx3 = self._negative(xx3)
                # xx = xx0 + xx1 + xx2 + xx3
                # # xxi = xi0 + xi1 + xi2 + xi3
                xx = x[i-k:]
                n_xx = length - i + k
            # middle
            else:
                xx = x[i-k:i+k+1]
                n_xx = width
                # xxi = xi[i-k:i+k+1]
            x_i = np.sum(xx)/n_xx
            result[i] = x_i
            # print(xxi)
            # print(xx)

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
        Parameters
        ----------
        y, the original series.
        trend, the trend item.
        sub_rho_weight, cycle sub-robustness weights, calculated by residuals.
        rho_weight, robustness weights, calculated by residuals.
        Returns
        -------

        trend
        calculate seasonal item
        ni, the number of passes through the inner loop, 1 or 2.
        no, the number of robustness iterations of the outer loop.
        n_p, the number of observations in each cycle of the seasonal component.
        ns, parameter of loess in step 2, the window width, >=7, odd.  the smoothing parameter for the seasonal component.
        nl, parameter of loess in step 3, the window width, min(>=n_p, odd).  the smoothing parameter for the low-pass filter.
        nt, parameter of loess in step 6, the window width, 1.5np<nt<2np, odd.  the smoothing parameter for the trend component.
        k,
        N, the total number of observations in whole series.

        low-pass, low-frequency power.
        if one have no idea about parameters, you could read this behind.
        n_p:
        It is entirely possible that a time series can have two or more periodic components. for example, a series
        measured daily might have weekly and yearly periodicities. In such a case one can use STL to successively
        estimate the components by proceeding from the shortest-period component to the longest-period component,
        estimating each component, subtracting it out, and estimating the next component from the residuals.
        no and ni:
        The STL robust estimation is needed when prior knowledge of the data or diagnostic checking indicates that
        non-Gaussian behavior in the time-series leads to extreme transient variation. Otherwise we can omit the
        robustness iterations and set no == 0.
        In many cases, ni = 1 is sufficient, but we recommend ni = 2 to provide near certainty of convergence.
        Suppose now that we need robustness iterations. We want to choose no large enough so that the robust estimates
        of the trend and seasonal components converge. taking ni = 1 is recommended, default.
        However, for the daily CO2 data in figure 1, convergence was slower and 10 iterations were required.
        ns:
        The choice of ns determines the variation in the data that makes up the seasonal component; the choice of the
        appropriate variation depends critically on the characteristics of the series. It should be emphasized that
        there is an intrinsic ambiguity in the definition of seasonal variation.
        in many applications the final decision must be based on knowledge about the mechanism generating the series and
        the goals of the analysis.
        The ambiguity is true of all seasonal decomposition procedures, not just STL. A lucid discussion of this point
        is given by Carlin and Dempster (1989).
        The additional variation in these seasonal values, compared with the seasonal values for ns = 35, appears to be
        noise and not meaningful seasonal variation because the cycle in the CO2 series is caused mainly by the seasonal
        cycle of foliage in the Northern Hemisphere, and one would expect a smooth evolution of this cycle over years.
        """
        # 1 detrending
        y = (np.array(y) - np.array(trend)).tolist()

        # 2 cycle-subseries smoothing
        subseries = self._cycle_subseries(y)
        cycle = []
        for i in range(self.cycle_length):
            subseries_i = subseries[i]
            sub_rho_weight_i = sub_rho_weight[i]
            extend_subseries_i = self.loess(self.ns, subseries_i, degree=1, rho_weight=sub_rho_weight_i)  # q = ns, d = 1
            cycle.append(extend_subseries_i)
        extend_cycle = self._extend_subseries(cycle)
        cycle_v = self._recover_series(extend_cycle)

        pd_cycle = pd.DataFrame({
            "cycle_" + str(i): cycle[i] for i in range(self.cycle_length)
        })
        pd_cycle.index.name = "time"
        # file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_cycle.csv"
        file_name = r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ystl/pet_cycle.csv"
        pd_cycle.to_csv(file_name, sep=" ")

        pd_series = pd.DataFrame({"pet_loess": cycle_v})
        pd_series.index.name = "time"
        # file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_loess_extend_cycle_v.csv"
        file_name = r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ystl/pet_loess_extend_cycle_v.csv"
        pd_series.to_csv(file_name, sep=" ")

        # 3 low-pass filtering of smoothed cycle-subseries
        lowf1 = self.moving_average_smoothing(self.n_p, cycle_v)  # n_p
        lowf2 = self.moving_average_smoothing(self.n_p, lowf1)
        lowf3 = self.moving_average_smoothing(self.n_p, lowf2)
        lowf4 = self.moving_average_smoothing(2 * self.n_p, lowf3)
        lowf5 = self.loess(self.nl, lowf4)
        pd_lowf = pd.DataFrame({"lowf1": lowf1, "lowf2": lowf2, "lowf3": lowf3, "lowf4": lowf4, "lowf5": lowf5})  # "lowf11": lowf11, "lowf12": lowf12,
        pd_lowf.index.name = "time"
        # file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_lowf.csv"
        file_name = r"/mnt/d/minio/waterism/datasets-origin/camels/camels_ystl/pet_lowf.csv"
        pd_lowf.to_csv(file_name, sep=" ")

        # 4 detrending of smoothed cycle-subseries
        cycle_v = np.array(cycle_v)
        lowf = np.array(lowf4)
        season = cycle_v - lowf

        # 5 deseasonalizing
        trend = np.array(self.extend_x) - season
        trend = trend.tolist()
        season = season.tolist()

        # 6 Trend Smoothing
        trend = self.loess(self.nt, trend, degree=1, rho_weight=rho_weight)

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
        # extend_sub_rho_weight = self._extend_subseries(sub_rho_weight)
        # return extend_sub_rho_weight
        return sub_rho_weight

    def outer_loop(self):
        """
        the outer loop of stl
        Returns
        -------
        the relationship of original data, trend and season in inner loop
        the role of loop

        more details about parameters value.

        robustness weights
        no, the number of outer loop, 0<=no<=10.
        """
        trend = [0]*self.length
        season = [0]*self.length
        trend_ij0 = []
        season_ij0 = []
        rho_weight = [1]*self.length
        extend_rho_weight = self._extend_original_series(rho_weight)

        # outer loop
        for i in range(self.no):
            if i == 0:
                trend_i0 = trend
                season_i0 = season
            extend_sub_rho_weight = self.extend_cycle_sub_rho_weight(rho_weight)

            # inner loop
            for j in range(self.ni):
                if j == 0:
                    trend_ij0 = trend_i0[:]
                    season_ij0 = season_i0[:]
                trend_i, season_i = self.inner_loop(self.x, trend_ij0, extend_sub_rho_weight, extend_rho_weight)
                if self.ni > 1:
                    t_a = np.max(np.absolute(np.array(trend_ij0)-np.array(trend_i)))
                    t_b = np.max(trend_ij0)
                    t_c = np.min(trend_ij0)
                    terminate_trend = np.divide(t_a, (t_b + t_c))
                    s_a = max(np.absolute(np.array(season_ij0)-np.array(season_i)))
                    s_b = max(season_ij0)
                    s_c = min(season_ij0)
                    terminate_season = s_a / (s_b + s_c)
                    if terminate_trend < 0.01 or terminate_season < 0.01:
                        print("terminate_trend < 0.01 or terminate_season < 0.01")
                        print("inner loop = " + str(j))
                        break
                    else:
                        trend_ij0 = trend_i[:]
                        season_ij0 = season_i[:]

            residuals = np.array(self.extend_x) - np.array(trend_i) - np.array(season_i)
            abs_residuals = np.absolute(residuals)
            h = 6 * np.median(abs_residuals)
            abs_residuals_h = abs_residuals / h
            extend_rho_weight = self.rho_weight(abs_residuals_h)  # todo:

            if self.no > 1:
                t_a = max(np.absolute(np.array(trend_i0) - np.array(trend_i)))
                t_b = max(trend_i0)
                t_c = min(trend_i0)
                terminate_trend = t_a / (t_b + t_c)
                s_a = max(np.absolute(np.array(season_i0) - np.array(season_i)))
                s_b = max(season_i0)
                s_c = min(season_i0)
                terminate_season = s_a / (s_b + s_c)
                if terminate_trend < 0.01 or terminate_season < 0.01:
                    print("terminate_trend < 0.01 or terminate_season < 0.01")
                    print("outer loop = " + str(i))
                    break
                else:
                    trend_i0 = trend_i[:]
                    season_i0 = season_i[:]

        return trend_i, season_i, residuals


    def season_post_smoothing(self, season):
        """post-smoothing of the seasonal"""
        ns = 11
        season = self.loess(ns, season, degree=2)
        return season

    def decomposition(self):
        """  """
        trend_, season_, residuals_ = self.outer_loop()
        post_season_ = self.season_post_smoothing(season_)
        post_residuals_ = np.array(self.extend_x) - trend_ - post_season_
        trend = trend_[self.cycle_length:-self.cycle_length]
        season = season_[self.cycle_length:-self.cycle_length]
        residuals = residuals_[self.cycle_length:-self.cycle_length]
        post_season = post_season_[self.cycle_length:-self.cycle_length]
        post_residuals = post_residuals_[self.cycle_length:-self.cycle_length]
        trend = np.around(trend, 2)
        season = np.around(season, 2)
        residuals = np.around(residuals, 2)
        post_season = np.around(post_season, 2)
        post_residuals = np.around(post_residuals, 2)
        self.trend = trend
        self.season = post_season
        self.residuals = post_residuals
        return trend, season, residuals, post_season, post_residuals

    def decompose(
        self,
        x,
    ):
        """
        decomposition function.
        Parameters
        ----------
        x
        Returns
        -------
        trend, season and residuals item.
        """
        self.reset_stl(x)
        self.decomposition()
        return self.trend, self.season, self.residuals

class MutualInformation(object):
    """mutual information"""
    def __init__(self):
        """
        probability
        joint probability
        """
        self.x = None
        self.y = None
        self.length = 0
        self.px = 0
        self.py = 0
        self.pxy = 0  # joint probability.
        self.mi = 0  # mutual information

    def reset_mutualinformation(self, x, y):
        """  """
        self.x = x
        self.y = y
        self.length = len(self.x)
        self.px = 0
        self.py = 0
        self.pxy = 0
        self.mi = 0

    def marginal_probability(
        self,
        x,
    ):
        """calculate the probability of a discrete variable"""
        incident, counts = np.unique(x, return_counts=True)
        frequency = np.divide(counts, self.length)
        distribution_low = np.array([incident, frequency])
        distribution_low = np.transpose(distribution_low)

        return distribution_low

    def joint_probability(
        self,
        x,
        y,
    ):
        """calculate the joint probability of two discrete variable"""
        xy = np.array([x, y])
        xy = np.transpose(xy)
        incident, counts = np.unique(xy, axis=0, return_counts=True)
        frequency = np.divide(counts, self.length)
        frequency = np.transpose([frequency])
        distribution_low = np.concatenate((incident, frequency), axis=1)

        return distribution_low

    def mutual_information(
        self,
        x = None,
        y = None,
    ):
        """calculate the mutual information of two discrete variables"""
        if (x is not None) and (y is not None):
            self.reset_mutualinformation(x, y)
        dl_x = self.marginal_probability(self.x)
        dl_y = self.marginal_probability(self.y)
        dl_xy = self.joint_probability(self.x, self.y)
        n_dl_xy = dl_xy.shape[0]
        mi = 0
        for i in range(n_dl_xy):
            x_i = dl_xy[i][0]
            y_i = dl_xy[i][1]
            pxy_i = dl_xy[i][2]
            px_ii = np.where(dl_x == x_i)
            px_i = dl_x[px_ii[0], 1]
            py_ii = np.where(dl_y == y_i)
            py_i = dl_y[py_ii[0], 1]
            mi_i = pxy_i * np.log(pxy_i / (px_i * py_i))
            mi = mi + mi_i[0]

        self.px = dl_x
        self.py = dl_y
        self.pxy = dl_xy
        self.mi = mi

        return dl_x, dl_y, dl_xy, mi

    def time_step(
        self,
        x,
        n
    ):
        """ """
        n_timestep = 0
        mi_ = [0]*n
        for i in range(1, n):
            x_ = x[i:]
            y = x[:-i]
            dl_x, dl_y, dl_xy, mi_i = self.mutual_information(x_, y)
            mi_[i] = float(mi_i)
            if i > 1:
                if mi_i > mi_i0:
                    n_timestep = i
                    break
            mi_i0 = mi_i

        # return n_timestep, mi_
        return n_timestep

class Decomposition(object):
    """
    decomposition class.
    """
    def __init__(
        self,
        data_cfgs: Dict,
    ):
        """
        initiate a decomposition object.
        """
        self.data_cfgs = data_cfgs
        self.t_range_train = data_cfgs["t_range_train"]
        self.t_range_valid = data_cfgs["t_range_valid"]
        self.t_range_test = data_cfgs["t_range_test"]
        self.time_range = self.marge_time_range(self.t_range_train, self.t_range_valid, self.t_range_test)
        self.basin = self.data_cfgs["object_ids"]
        self.n_basin = len(self.basin)
        self.x_origin = None
        self.y_origin = None
        self.c_origin = None
        self.y_decomposed = None
        self.time = None
        self.attrs = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.time_step_mi = 0
        self._read_xyc_specified_time(self.time_range)
        self.remove_leap_year_data()
        self.time_step_via_mi()


    def date_string2number(self, date_str):
        str_list = date_str.split("-")
        date_num = [int(s) for s in str_list]
        return date_num

    def marge_time_range(
        self,
        t_range_train: list = None,
        t_range_valid: list = None,
        t_range_test: list = None,
    ):
        """marge time range"""
        t_range_list = []
        if t_range_train is not None:
            t_range_list.append(t_range_train)
        if t_range_valid is not None:
            t_range_list.append(t_range_valid)
        if t_range_test is not None:
            t_range_list.append(t_range_test)
        # t_n = len(t_range_list)
        # for i in range(t_n - 1):
        #     if t_range_list[i][1] != t_range_list[i + 1][0]:
        #         raise ValueError("t_range_list and t_range_list must be equal")
        time_range = [t_range_list[0][0], t_range_list[-1][-1]]

        return time_range

    def pick_leap_year(
        self,
        time_range: list
    ):
        """
        pick out the leap years within a time range.
        """
        start_date = time_range[0]
        end_date = time_range[-1]
        start = self.date_string2number(start_date)
        end = self.date_string2number(end_date)
        year_start = start[0]
        month_start = start[1]
        year_end = end[0]
        if month_start > 2:
            year = list(range(year_start + 1, year_end + 1))
        else:
            year = list(range(year_start, year_end + 1))
        leap_year = []
        month_day = "-02-29"
        for i in range(len(year)):
            remainder = year[i] % 4
            if remainder == 0:
                year_month_day = str(year[i]) + month_day
                date = hydro_time.t2str(year_month_day)
                leap_year.append(date)
        return leap_year

    @property
    def data_source(self):
        source_name = self.data_cfgs["source_cfgs"]["source_name"]
        source_path = self.data_cfgs["source_cfgs"]["source_path"]
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        return data_sources_dict[source_name](source_path, **other_settings)

    def _read_xyc_specified_time(self, time_range):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        # x
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.basin,
            time_range,
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.basin,
            time_range,
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.basin,
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        if data_forcing_ds_ is not None:
            self.x_origin = data_forcing_ds_.copy(deep=True)
        if data_output_ds_ is not None:
            self.y_origin = data_output_ds_.copy(deep=True)
        if data_attr_ds is not None:
            self.c_origin = data_attr_ds.copy(deep=True)

    def remove_leap_year_data(self):
        """
        remove the data in February 29th per leap year.
        """
        leap_years = self.pick_leap_year(self.time_range)
        n_leap_years = len(leap_years)
        for i in range(n_leap_years):
            if self.x_origin is not None:
                self.x_origin = self.x_origin.drop_sel(time=leap_years[i])
            self.y_origin = self.y_origin.drop_sel(time=leap_years[i])
        self.time = self.y_origin.time
        self.attrs = self.y_origin.attrs

    def time_step_via_mi(self):
        """
        calculate time step of a time series via mutual information value.
        """
        n = 15
        time_step = 0
        mi = MutualInformation()
        x = self.y_origin.streamflow.values.tolist()
        for i in range(self.n_basin):
            xx = x[i]
            n_timestep_i = mi.time_step(xx, n)
            if i == 0:
                time_step = n_timestep_i
            else:
                if n_timestep_i < time_step:
                    time_step = n_timestep_i

        self.time_step_mi = time_step

    def stl_decomposition(self):
        """
        decompose time series via stl method.

        [time, basin, streamflow] -> [time, basin, trend|season|residuals]
        """
        trend = []
        season = []
        residuals = []
        stl = STL(frequency=1, cycle_length=365)
        for i in range(self.n_basin):
            data = self.y_origin.streamflow.values[i].tolist()
            trend_, season_, residuals_ = stl.decompose(data)
            trend.append(trend_[:])
            season.append(season_[:])
            residuals.append(residuals_[:])

        trend = np.array(trend)
        season = np.array(season)
        residuals = np.array(residuals)
        trend_DataArray = xr.DataArray(trend, dims=['basin', 'time'], coords={'basin': self.basin, 'time': self.time}, name = 'trend', attrs=self.attrs)
        season_DataArray = xr.DataArray(season, dims=['basin', 'time'], coords={'basin': self.basin, 'time': self.time}, name = 'season', attrs=self.attrs)
        residuals_DataArray = xr.DataArray(residuals, dims=['basin', 'time'], coords={'basin': self.basin, 'time': self.time}, name = 'residuals', attrs=self.attrs)
        self.y_decomposed = xr.Dataset({'trend': trend_DataArray,
                                'season': season_DataArray,
                                'residuals': residuals_DataArray})

        # split data into train, valid and test period
        self.split_period()

        # return rain, valid, test dataset and time step via mutual information.
        return self.train_data, self.valid_data, self.test_data, self.time_step_mi

    def split_period(self):
        """split data into train, valid and test period"""
        if self.t_range_train is not None:
            if self.x_origin is not None:
                x_origin_train = self.x_origin.sel(time=slice(self.t_range_train[0], self.t_range_train[1]))
            else:
                x_origin_train = None
            y_origin_train = self.y_origin.sel(time=slice(self.t_range_train[0], self.t_range_train[1]))
            c_origin_train = self.c_origin
            y_decomposed_train = self.y_decomposed.sel(time=slice(self.t_range_train[0], self.t_range_train[1]))
            self.train_data = [x_origin_train, y_origin_train, c_origin_train, y_decomposed_train]

            if self.t_range_valid is not None:
                if self.x_origin is not None:
                    x_origin_valid = self.x_origin.sel(time=slice(self.t_range_valid[0], self.t_range_valid[1]))
                else:
                    x_origin_valid = None
                y_origin_valid = self.y_origin.sel(time=slice(self.t_range_valid[0], self.t_range_valid[1]))
                c_origin_valid = self.c_origin
                y_decomposed_valid = self.y_decomposed.sel(time=slice(self.t_range_valid[0], self.t_range_valid[1]))
                self.valid_data = [x_origin_valid, y_origin_valid, c_origin_valid, y_decomposed_valid]

        if self.t_range_test is not None:
            if self.x_origin is not None:
                x_origin_test = self.x_origin.sel(time=slice(self.t_range_test[0], self.t_range_test[1]))
            else:
                x_origin_test = None
            y_origin_test = self.y_origin.sel(time=slice(self.t_range_test[0], self.t_range_test[1]))
            c_origin_test = self.c_origin
            y_decomposed_test = self.y_decomposed.sel(time=slice(self.t_range_test[0], self.t_range_test[1]))
            self.test_data = [x_origin_test, y_origin_test, c_origin_test, y_decomposed_test]


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

    def cal_statistics(self):
        """calculate statistics"""


    def cal_7_stat_inds(self, x):
        """
        Calculate seven statistics indices of a series: point number, mean value, standard deviation, min, percentile 15 50 85 and max.

        Parameters
        ----------
        x
            input data

        Returns
        -------
        list
            [mean, std, min, p15, p50, p85, max]
        """
        num_point = x.shape[0]
        mean = np.mean(x).astype(float)
        std = np.std(x).astype(float)
        min_ = np.min(x).astype(float)
        p15 = np.percentile(x, 15).astype(float)
        p50 = np.percentile(x, 50).astype(float)
        p85 = np.percentile(x, 85).astype(float)
        max_ = np.max(x).astype(float)

        if std < 0.001:
            std = 1
        return [num_point, mean, std, min_, p15, p50, p85, max_]

    def deficient_dataset(self):
        """generate deficient dataset."""


    def analysis_dataset(self):
        """ analysis dataset."""

    def cal_mse(self, x, y):
        """calculate mean squared error."""
        return np.mean((y - x) ** 2)

    def cal_spearman(self, x, y):
        """calculate spearman correlation."""
