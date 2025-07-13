""" """

import numpy as np

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
        residual_2 = np.power(residual, 2)
        residual_2 = residual_2.tolist()
        a0, R_20, delta_20 = self.arch.arch_ordinary_least_squares_estimation(residual_2, 1)
        Omega = self.arch.Omega(residual_2, a0)
        a1, R_21, y1 = self.arch.arch_feasible_generalized_least_squares_estimation(residual_2, q, Omega, b_y=True)
        theta0 = a1[:]
        indices = np.where(a1 < 0)
        indices = indices[0]
        n_indices = indices.size
        if n_indices > 0:
            a2, R_22, y2 = self.arch.arch_constrained_ordinary_least_squares(residual_2, q, Omega, q_n=indices, b_y=True)
            theta0 = a2[:]
        p = 2
        q = 3
        b_arima = False
        theta1 = self.arch.gradient_ascent(x, theta0, p, q, b_arima)

        return a0, R_20, delta_20, a1, R_21, y1, theta1

