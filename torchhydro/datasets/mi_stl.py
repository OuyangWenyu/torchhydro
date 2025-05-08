
import numpy as np
import pandas as pd

class STL():
    """
    Seasonal-Trend decomposition using LOESS
    Loess     circle-subseries     low pass filter
    y_t = T_t + S_t + R_t
    todo: unify the data type.
    """
    def __init__(self, x):
        """
        initiate a STL model
        """
        self.x = x  # the original data
        self.frequency = 1  # the frequency of time series
        self.length = len(x)  # the length of time series
        self.trend = None  # trend item
        self.season = None  # season item
        self.residuals = None  # residuals item
        self.mode = "addition"
        self.parity = None  # the parity of frequency
        self.mutation = None
        self.cycle_subseries = None
        self.cycle_length = 365
        self.no = 10  # window width, span
        self.ni = 1  # need to be odd
        self.ns = 9  # odd >=7
        self.nl = 365  # need to be odd
        self.nt = 421  # 1 or 2
        self.n_p = 365  # True of False
        self.degree = 1  # 1 or 2, locally-linear or locally-quadratic

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
        pd_subseries = pd.DataFrame({
            "subseries_" + str(i): subseries[i] for i in range(n_subseries)
        })
        pd_subseries.index.name = "time"
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_subseries.csv"
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
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_extend_subseries.csv"
        pd_extend_subseries.to_csv(file_name, sep=" ")
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
        for i in range(len_subseries):  # 18
            for j in range(n_subseries):  # 365
                series_ji = float(subseries[j][i])
                series_i[j] = series_ji
            series = series + series_i[:]
        pd_series = pd.DataFrame({"pet_loess": series})
        pd_series.index.name = "time"
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_loess_extend_cycle_v.csv"
        pd_series.to_csv(file_name, sep=" ")
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

    def _neighborhood_weight(
        self,
        width: int,
        degree: int = 2,
    ):
        """
        calculate neighborhood weights within window
        Parameters
        ----------
        width
        degree

        Returns
        -------

        """
        # todo: !
        k = int(width / 2)
        weight = []
        for i in range(width):
            d_i = np.absolute((i + 1) - (k + 1)) / (k + 1)
            w_i = self.weight_function(d_i, degree)
            weight.append(w_i)
        return weight

    def _neighborhood_weight_x(self, xi, x):
        """"""
        width = len(x)
        weight = self._neighborhood_weight(width)
        v_xi = weight * np.absolute(xi - x)/((width - 1)/2)
        return v_xi

    def weight_least_squares(
        self,
        x,
        y,
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
        weight = self._neighborhood_weight(length)
        try:
            weight = np.multiply(weight, rho_weight)
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
        ii = int(length/2 + 1)
        yy = 0
        for i in range(degree+1):
            yy = yy + a[i] * np.power(x[ii], i)

        return yy

    def loess(
        self,
        width: int,
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
        Returns
        -------

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
                if m == k:
                    y0 = []
                    y3 = y2[:]
                    y3.reverse()
                    rw_i0 = []
                    rw_i3 = rw_i2[:]
                    rw_i3.reverse()
                elif (m > 1) and (m < k):
                    y0 = x[:i]
                    y3 = y2[-m:]
                    y3.reverse()
                    rw_i0 = rho_w[:i]
                    rw_i3 = rw_i2[-m:]
                    rw_i3.reverse()
                else:
                    y0 = x[:i]
                    y3 = [y2[-m]]
                    rw_i0 = rho_w[:i]
                    rw_i3 = [rw_i2[-m]]
                y = y3 + y0 + y1 + y2
                rw_i = rw_i3 + rw_i0 + rw_i1 + rw_i2
            # end
            elif i > (length-1) - k:
                m = k + i - (length-1)
                y1 = [x[i]]
                y0 = x[i-k:i]
                rw_i1 = [rho_w[i]]
                rw_i0 = rho_w[i-k:i]
                if m == k:
                    y2 = []
                    y3 = y0[:]
                    y3.reverse()
                    rw_i2 = []
                    rw_i3 = rw_i0[:]
                    rw_i3.reverse()
                elif (m > 1) and (m < k):
                    y2 = x[i+1:]
                    y3 = y0[:m]
                    y3.reverse()
                    rw_i2 = rho_w[i+1:]
                    rw_i3 = rw_i0[:m]
                    rw_i3.reverse()
                else:
                    y2 = x[i+1:]
                    y3 = [y0[m-1]]
                    rw_i2 = rho_w[i+1:]
                    rw_i3 = [rw_i0[m-1]]
                y = y0 + y1 + y2 + y3
                rw_i = rw_i0 + rw_i1 + rw_i2 + rw_i3
            else:
                y = x[i-k:i+k+1]
                rw_i = rho_w[i-k:i+k+1]
            #
            y_i = self.weight_least_squares(xx, y, degree, rw_i)
            result[i] = y_i

        return result


    def moving_average_smoothing(
        self,
        width,
        x
    ):
        """
        moving average smoothing
        Parameters
        ----------
        width: int, window width.
        x: series need to smoothing.

        Returns
        -------

        """
        length = len(x)
        k = int(width/2)
        result = [0]*length
        for i in range(length):
            # start
            if i < k:
                m = k - i
                xx1 = [x[i]]
                xx2 = x[i + 1:i + k + 1]
                if m == k:
                    xx0 = []
                    xx3 = xx2[:]
                    xx3.reverse()
                elif (m > 1) and (m < k):
                    xx0 = x[:i]
                    xx3 = xx2[-m:]
                    xx3.reverse()
                else:
                    xx0 = x[:i]
                    xx3 = [xx2[-m]]
                xx = xx3 + xx0 + xx1 + xx2
            # end
            elif i > (length-1) - k:
                m = k + i - (length-1)
                xx1 = [x[i]]
                xx0 = x[i-k:i]
                if m == k:
                    xx2 = xx0[:]
                    xx2.reverse()
                    xx3 = []
                elif (m > 1) and (m < k):
                    xx2 = x[i+1:]
                    xx3 = xx0[:m]
                    xx3.reverse()
                else:
                    xx2 = x[i+1:]
                    xx3 = [xx0[m]]
                xx = xx0 + xx1 + xx2 + xx3
            # middle
            else:
                xx = x[i-k:i+k+1]
            x_i = np.sum(xx)/width
            result[i] = x_i

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
        # ns = 9  # q  35  odd >=7  < 15
        # nl = 365
        # nt = 421    # odd
        # n_p = 365

        k = 5  # todo

        # 1 detrending
        y = np.array(y) - np.array(trend)  # todo:

        # 2 cycle-subseries smoothing
        subseries = self._cycle_subseries(y)
        extend_subseries = self._extend_subseries(subseries)
        cycle = []
        for i in range(self.cycle_length):
            extend_subseries_i = extend_subseries[i]
            sub_rho_weight_i = sub_rho_weight[i]
            extend_subseries_i = self.loess(self.ns, extend_subseries_i, degree=1, rho_weight=sub_rho_weight_i)  # q = ns, d = 1
            cycle.append(extend_subseries_i)
        cycle_v = self._recover_series(cycle)
        pd_cycle = pd.DataFrame({
            "cycle_" + str(i): cycle[i] for i in range(self.cycle_length)
        })
        pd_cycle.index.name = "time"
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_cycle.csv"
        pd_cycle.to_csv(file_name, sep=" ")

        # 3 low-pass filtering of smoothed cycle-subseries
        lowf1 = self.moving_average_smoothing(self.n_p, cycle_v)  # n_p
        lowf2 = self.moving_average_smoothing(self.n_p, lowf1)
        lowf3 = self.moving_average_smoothing(3, lowf2)
        lowf4 = self.loess(self.nl, lowf3)
        pd_lowf = pd.DataFrame({"lowf1": lowf1, "lowf2": lowf2, "lowf3": lowf3, "lowf4": lowf4})
        pd_lowf.index.name = "time"
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_lowf.csv"
        pd_lowf.to_csv(file_name, sep=" ")

        # 4 detrending of smoothed cycle-subseries
        cycle_v = np.array(cycle_v)
        lowf = np.array(lowf4)
        season = cycle_v[self.cycle_length:-self.cycle_length] - lowf[self.cycle_length:-self.cycle_length]

        # 5 deseasonalizing
        trend = y - season
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
        extend_sub_rho_weight = self._extend_subseries(sub_rho_weight)
        return extend_sub_rho_weight

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
        no = 30
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
                    season_ij0 = season_i0[:]
                trend_i, season_i = self.inner_loop(self.x, trend_ij0, extend_sub_rho_weight, rho_weight)
                if ni > 1:
                    t_a = max(np.absolute(np.array(trend_ij0)-np.array(trend_i)))
                    t_b = max(trend_ij0)
                    t_c = min(trend_ij0)
                    terminate_trend = t_a / (t_b + t_c)
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

            residuals = np.array(self.x) - np.array(trend_i) - np.array(season_i)
            abs_residuals = np.absolute(residuals)
            h = 6 * np.median(abs_residuals)
            abs_residuals_h = abs_residuals / h
            rho_weight = self.rho_weight(abs_residuals_h)  # todo:

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
        ns = 7
        season = self.loess(ns, season, degree=2)
        return season

    def decomposition(self):
        """"""
        trend, season, residuals = self.outer_loop()
        post_season = self.season_post_smoothing(season)
        post_residuals = np.array(self.x) - trend - post_season
        decomposition = pd.DataFrame({"pet": self.x, "trend": trend, "season": season, "residuals": residuals, "post_season": post_season, "post_residuals": post_residuals})
        decomposition.index.name = "time"
        file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_decomposition.csv"
        decomposition.to_csv(file_name, sep=" ")
        self.trend = trend
        self.season = post_season
        self.residuals = post_residuals
        return decomposition
