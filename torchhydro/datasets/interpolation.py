""" """

import numpy as np
import random
import time

from torchhydro.datasets.arch import Arch

class Interpolation(object):
    """
    Time series interpolation class.

    """
    def __init__(self):
        self.arch = Arch()
        self.x = None
        self.y = None



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

    def read_data(self, path):

        x = np.load(path)

        return

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
    ):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        acf = self.arch.autocorrelation_function(x)
        pacf = self.arch.partial_autocorrelation_function(x)

        return acf, pacf

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


    def test_model(
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
        residual = self.arch.x_residual_via_parameters(x, phi)
        b_significant_arima = self.arch.test_arima(residual, m, significance_level)
        b_significant_para = self.arch.test_parameters(residual, phi, theta, se_beta, m, significance_level)

        return b_significant_arima, b_significant_para

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
        residual = self.arch.x_residual_via_parameters(x, phi)
        b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = self.arch.arch_test(residual, q, significance_level)

        return b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM

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
        residual = self.arch.x_residual_via_parameters(x, phi)
        residual_2 = np.power(residual, 2)
        acf = self.arch.autocorrelation_function(residual_2)
        pacf = self.arch.partial_autocorrelation_function(residual_2)

        return acf, pacf

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
        residual = self.arch.x_residual_via_parameters(x, phi)
        mean_residual, residual_center = self.arch.residual_center(residual)
        residual_2 = np.power(residual_center, 2)
        residual_2 = residual_2.tolist()
        a0, R_20, delta_20 = self.arch.arch_ordinary_least_squares_estimation(residual_2, 1)
        Omega = self.arch.Omega(residual_2, a0)
        a1, R_21, y1 = self.arch.arch_feasible_generalized_least_squares_estimation(residual_2, q, Omega, b_y=True)
        theta0 = a1[:]
        indices = np.where(a1 < 0)
        indices = indices[0]
        n_indices = indices.size
        a2 = None
        R_22 = None
        y2 = None
        if n_indices > 0:
            a2, R_22, y2 = self.arch.arch_constrained_ordinary_least_squares(residual_2, q, Omega, q_n=indices, b_y=True)
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

        residual = self.arch.x_residual_via_parameters(x, theta)
        mean_residual, residual_center = self.arch.residual_center(residual)
        residual_2 = np.power(residual_center, 2)
        (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_, nse, rmse,
         max_abs_error) = self.arch.arima_arch(x, theta, p, q)

        return (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_, nse,
                rmse, max_abs_error)

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
        x,
    ):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        n_x = len(x)
        x = np.array(x)
        indeces = np.where(x == -100)
        n_indeces = indeces[0].size

        lose_ratio = n_indeces / n_x

        return lose_ratio

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
        index = np.random.normal(loc=0, scale=1, size=n)

        return index
