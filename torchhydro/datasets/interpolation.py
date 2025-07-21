""" """

import numpy as np
import random
import time

from hydrodataset import CamelsCh, Camels

from torchhydro.datasets.arch import Arch

class Interpolation(object):
    """
    Time series interpolation class.

    """
    def __init__(self):
        self.arch = Arch()
        self.x = None
        self.x_dnan = None
        self.x_lose = None
        self.y = None
        self.n_x = None
        self.t_length = None
        self.datasource = Camels()
        self.gage_id = ["01013500",]
        # self.gage_id = ["01013500", "01022500", "01030500", "01187300"]  #["05087500",]     #["5011",]  # ["01013500",]
        # self.gage_id = self.datasource.gage
        # self.n_gage = self.datasource.n_gage
        self.time_range = ["1980-01-01", "2014-12-31"]      #["1984-01-01", "1987-12-31"]  # ["1980-01-01", "2014-12-31"]
        self.var_list = ["streamflow",]
        # self.var_list = self.datasource.get_relevant_cols()

        self.x = self.read_data()
        self.n_x = self.x.shape[0]
        self.t_length = self.x.shape[1]
        self.x_dnan = self.delete_nan()

    def read_data(self):
        data = self.datasource.read_ts_xrdataset(
            self.gage_id,
            self.time_range,
            self.var_list,
        )
        data = data.streamflow.values

        return data

    def delete_nan(self):
        """ Delete NaN values"""
        n_x = self.x.shape[0]
        x_dnan = []
        for i in range(n_x):
            x_i = self.x[i]
            x_i = x_i[~np.isnan(x_i)]
            x_dnan.append(x_i)

        return x_dnan

    def deletion_ratio(self):
        """
        Zhenghe Li P16
        Missing At Non-Random, MANR.
        Missing Completely At Random, MCAR.
        Missing Random, MAR.

        Returns
        -------

        """
        r0 = [0, 0.01]  # slight
        r1 = [0.01, 0.1]  # dram
        r2 = [0.1, 0.2]  # jot
        r3 = [0.2, 0.35]  # moderate  __
        r4 = [0.35, 0.45]  # stack
        r5 = [0.45, 0.99]  # serious

    def cal_lose_ratio(
        self,
    ):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        lose_ratio = []
        intact_series = []
        slight_lose = []
        dram_lose = []
        jot_lose = []
        moderate_lose = []
        stack_lose = []
        serious_lose = []
        for i in range(self.n_x):
            n_x_i = self.x_dnan[i].shape[0]
            lose_ratio_i = 1 - n_x_i / self.t_length
            lose_ratio.append(lose_ratio_i)
            if lose_ratio_i == 0.0:
                intact_series.append(i)
            elif lose_ratio_i < 0.01:
                slight_lose.append(i)
            elif (lose_ratio_i >= 0.01) and (lose_ratio_i < 0.1):
                dram_lose.append(i)
            elif (lose_ratio_i >= 0.1) and (lose_ratio_i < 0.2):
                jot_lose.append(i)
            elif (lose_ratio_i >= 0.2) and (lose_ratio_i < 0.35):
                moderate_lose.append(i)
            elif (lose_ratio_i >= 0.35) and (lose_ratio_i < 0.45):
                stack_lose.append(i)
            elif (lose_ratio_i >= 0.45):
                serious_lose.append(i)

        lose_type = {
            "intact_series": intact_series,
            "slight_lose": slight_lose,
            "dram_lose": dram_lose,
            "jot_lose": jot_lose,
            "moderate_lose": moderate_lose,
            "stack_lose": stack_lose,
            "serious_lose": serious_lose,
        }

        return lose_ratio, lose_type

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
        # num_point = x.shape[0]
        num_point = len(x)
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

    def statistics_indices(
        self,
    ):
        """

        Parameters
        ----------
        gage_id

        Returns
        -------

        """
        n_gage = len(self.gage_id)
        stat_inds = []
        for i in range(n_gage):
            x_i = self.x_dnan[i]
            stat_inds_i = self.cal_7_stat_inds(x_i)
            stat_inds.append(stat_inds_i)

        stat_inds = np.array(stat_inds)

        return stat_inds

    def lose_index(
        self,
        range,
        n,
    ):
        """

        Returns
        -------

        """
        random.seed(time.time()+1)
        index = np.random.randint(low=0, high=range, size=n, dtype=int)
        index = np.sort(index)

        return index

    def lose_series(
        self,
        x,
        n_x,
        ratio,
    ):
        """

        Parameters
        ----------
        n_x
        ratio

        Returns
        -------

        """
        lose_size = ratio * n_x
        lose_index = self.lose_index(n_x, lose_size)

        lose_x = x
        lose_x[lose_index] = np.NaN

        return lose_x

    def lose_set(
        self,
        x_set,
        ratio_list,
    ):
        """

        Parameters
        ----------
        x_set
        ratio_list: [0.05, 0.1, 0.15, 0.25, 0.3, 0.35]

        Returns
        -------

        """
        # ratio_list = [0.05, 0.1, 0.15, 0.25, 0.3, 0.35]
        n_ratio_list = len(ratio_list)
        n_x = x_set.shape[0]
        n_xi = x_set.shape[1]

        lose_set_x = [[]] * n_x
        for i in range(n_x):
            x_i = x_set[i]
            for j in range(n_ratio_list):
                lose_x_ij = self.lose_series(x_i, n_xi, ratio_list[j])
                lose_set_x[i].append(lose_x_ij[:])

        lose_set_x = np.array(lose_set_x)

        return lose_set_x


    def smooth_test(
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
        p = 3
        case = "case 1"
        significance_level = 0.05
        b_ = self.arch.adf_test(x, p, case, significance_level)

        return b_

    def degree_ar(
        self,
        x,
        phi = None,
    ):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        if phi is not None:
            y = self.arch.arima(x=x, phi=phi, p=len(phi))
            x = np.array(x) - y
            x = np.power(x, 2)
        acf = self.arch.autocorrelation_function(x)
        pacf = self.arch.partial_autocorrelation_function(x)

        return acf, pacf

    def model_degree(
        self,
        x,
        L,
    ):
        """

        Parameters
        ----------
        x
        L

        Returns
        -------

        """
        degree_aic, aic_min, phi_min, R_2_min, degree_bic, bic_min, degree_aic_c, aic_c_min = self.arch.arma_degree(x, L)

        return degree_aic, aic_min, phi_min, R_2_min, degree_bic, bic_min, degree_aic_c, aic_c_min

    def arima_parameter(
        self,
        x,
        p,
        q,
    ):
        """

        Parameters
        ----------
        x
        p
        q

        Returns
        -------

        """
        phi, theta,R_2, se_beta = self.arch.arma_least_squares_estimation(x=x, p=p, q=q)

        return phi, theta, R_2, se_beta


    def test_arima_model(
        self,
        x,
        phi,
        theta,
        se_beta,
        m,
        significance_level,
    ):
        """

        Parameters
        ----------
        x
        p

        Returns
        -------

        """
        residual, mean_residual, residual_center, residual_center_2 = self.arch.x_residual_via_parameters(x, phi, b_center=True)
        b_significant_arima = self.arch.test_arima(residual_center, m, significance_level)
        b_significant_para = self.arch.test_parameters(residual_center, phi, theta, se_beta, m, significance_level)

        return b_significant_arima, b_significant_para


    def degree_arch(
        self,
        x,
        phi,
    ):
        """

        Parameters
        ----------
        x
        phi

        Returns
        -------

        """
        residual, mean_residual, residual_center, residual_center_2 = self.arch.x_residual_via_parameters(x, phi, b_center=True)
        acf = self.arch.autocorrelation_function(residual_center_2)
        pacf = self.arch.partial_autocorrelation_function(residual_center_2)

        return acf, pacf

    def test_arch(
        self,
        x,
        phi,
        q,
        significance_level,
    ):
        """

        Parameters
        ----------
        x
        phi

        Returns
        -------

        """
        residual, mean_residual, residual_center, residual_center_2 = self.arch.x_residual_via_parameters(x, phi, b_center=True)
        b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = self.arch.arch_test(residual_center, q, significance_level)

        return b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM

    def arch_parameter(
        self,
        x,
        phi,
        p,
        q,
    ):
        """

        Parameters
        ----------
        x
        phi
        q

        Returns
        -------

        """
        residual, mean_residual, residual_center, residual_center_2 = self.arch.x_residual_via_parameters(x, phi, b_center=True)
        residual_center_2 = residual_center_2.tolist()
        a0, R_20, delta_20 = self.arch.arch_ordinary_least_squares_estimation(residual_center_2, 1)
        Omega = self.arch.Omega(residual_center_2, a0)
        a1, R_21, y1 = self.arch.arch_feasible_generalized_least_squares_estimation(residual_center_2, q, Omega, b_y=True)
        theta0 = a1[:]
        indices = np.where(a1 < 0)
        indices = indices[0]
        n_indices = indices.size
        a2 = None
        R_22 = None
        y2 = None
        if n_indices > 0:
            a2, R_22, y2 = self.arch.arch_constrained_ordinary_least_squares(residual_center_2, q, Omega, q_n=indices, b_y=True)
            theta0 = a2[:]
        theta0 = np.insert(theta0, 0, phi)
        b_arima = True
        theta1 = self.arch.gradient_ascent(x, theta0, p, q, b_arima)

        if n_indices > 0:
            return a0, R_20, delta_20, a1, R_21, y1, a2, R_22, y2, theta1

        return a0, R_20, delta_20, a1, R_21, y1, theta1

    def arch_model(
        self,
        x,
        theta,
        p,
        q,
        nse,
        rmse,
        max_error,
        max_loop,
    ):
        """

        Parameters
        ----------
        x
        theta
        p
        q

        Returns
        -------

        """
        n_theta = len(theta)
        if n_theta != p+q+1:
            raise ValueError("the length of theta need to be equal to p+q+1.")

        residual, mean_residual, residual_center, residual_center_2 = self.arch.x_residual_via_parameters(x, theta, b_center=True)
        result = self.arch.arima_arch_model(x, theta, p, q, nse, rmse, max_error, max_loop)

        return result

    def arch_interpolate(
        self,
        x,
        theta,
        p,
        q,
    ):
        """

        Parameters
        ----------
        x
        theta
        p
        q

        Returns
        -------

        """
        phi = theta[:p]
        alpha = theta[p:]

        (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_, nse, rmse,
         max_abs_error) = self.arch.arima_arch(x, theta, p, q)

        return (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_,
                nse, rmse, max_abs_error)



    def genetate_lose_time_series_single(
        self,
        x,
        n,
    ):
        """

        Parameters
        ----------
        x
        n

        Returns
        -------

        """
        n_x = len(x)
        x = np.array(x)
        lose_index = self.lose_index(n_x, n)
        lose_x = x[:]
        lose_x[lose_index] = "NaN"

        return lose_x

    def genetate_lose_time_series(
        self,
        x: np.ndarray,
        n,
    ):
        """

        Parameters
        ----------
        x
        n

        Returns
        -------

        """
        n_x = x.shape[0]
        n_xi = x.shape[1]

        lose_x = np.zeros((n_x, n_xi))
        for i in range(n_x):
            lose_x[i, :] = self.genetate_lose_time_series_single(x[i, :], n)

        return lose_x

    def mse(
        self,
        x,
        y,
    ):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        error = np.array(y) - np.array(x)
        error_2 = np.power(error, 2)
        mse = np.mean(error_2)

        return mse

    def correlation_coefficient_spearman(
        self,
        x,
        n,
    ):
        """
        Zhenghe Li P30
        Parameters
        ----------
        x
        n

        Returns
        -------

        """
        # n_x = x.shape[0]
        n_x = len(x)
        m = n_x - n
        d1 = x[:m]
        d2 = x[n:]
        d = np.array(d1) - np.array(d2)
        d_2 = np.power(d, 2)
        rho = np.sum(d_2)
        rho = 6 * rho / (m * (m * m - 1))
        rho = 1 - rho

        return rho

    def fourier_series(
        self,
        T,
    ):
        """

        Parameters
        ----------
        T

        Returns
        -------

        """


