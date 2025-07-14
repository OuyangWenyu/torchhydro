
import numpy as np

from hydrodataset import CamelsCh, Camels
from test_arch_data import camelsch_streamflow_8487, camelsch_streamflow_8183, camelsch_streamflow_8183_d1, camelsch_streamflow_81
from torchhydro.datasets.interpolation import Interpolation


class Camelsdata(object):
    def __init__(self):
        self.datasource = CamelsCh()
        self.basin = ["5011",]  # ["01013500",]
        self.time_range = ["1984-01-01", "1987-12-31"]  # ["1980-01-01", "2014-12-31"]
        self.var_list = ["streamflow"]
        self.pet_list = ["pet"]
        self.prcp_list = ["prcp"]
        self.streamflow = None
        self.prcp = None
        self.pet = None
        self.read_streamflow()
        # self.read_prcp()
        # self.read_pet()

    def read_streamflow(self):
        data = self.datasource.read_ts_xrdataset(
            self.basin,
            self.time_range,
            self.var_list,
        )
        data1 = data.streamflow.to_dataframe()
        # data2 = data.discharge_vol1.to_dataframe()
        # data3 = data.discharge_vol2.to_dataframe()
        # data4 = data.discharge_vol3.to_dataframe()
        # data1.drop(axis=0, index=("1000", "1992-02-29"), inplace=True)
        # data2.drop(axis=0, index=("1000", "1992-02-29"), inplace=True)
        # data3.drop(axis=0, index=("1000", "1992-02-29"), inplace=True)
        # data4.drop(axis=0, index=("1000", "1992-02-29"), inplace=True)
        data1 = data1.values[:, 0]
        # data2 = data2.values[:, 0]
        # data3 = data3.values[:, 0]
        # data4 = data4.values[:, 0]
        data_ = data1.tolist()  # + data2.tolist() + data3.tolist() + data4.tolist()
        self.streamflow = data_  # + data_

def test_readdata():
    camelsdata = Camelsdata()
    x = camelsdata.streamflow
    print(len(x))
# Camels  01013500
# 12784
# CamelsCh  5011
# 14610
# CamelsCh  5011  ["1984-01-01", "1987-12-31"]
# 1461

def test_smooth_test():
    inter = Interpolation()
    b_ = inter.smooth_test(camelsch_streamflow_8183_d1)
    print("b_ = " + str(b_))
# camelsch_streamflow_8183
# b_ = True
# acmelsch_streamflow_8183_d1
# b_ = True

def test_degree_ar():
    inter = Interpolation()
    acf, pacf = inter.degree_ar(camelsch_streamflow_81)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\acf.txt', acf)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\pacf.txt', pacf)
    print("n_acf = " + str(len(acf)))
    print("n_pacf = " + str(len(pacf)))
# camelsch_streamflow_8183
# n_acf = 731
# n_pacf = 731
# camelsch_streamflow_8183d1
# n_acf = 731
# n_pacf = 731
# camelsch_streamflow_81
# n_acf = 244
# n_pacf = 244

def test_arma_parameters():
    inter = Interpolation()
    x = camelsch_streamflow_81
    p = 4
    q = 0
    phi, theta, R_2, se_beta = inter.arima_parameter(x, p, q)
    print("phi = " + str(phi))
    print("theta = " + str(theta))
    print("R_2 = " + str(R_2))
    print("se_beta = " + str(se_beta))
# camelsch_streamflow_8183  p=3 q=0
# phi = [ 1.30134078 -0.67576837  0.26822102]
# theta = []
# R_2 = 0.7602774658956198
# se_beta = [0.030009693485665807, 0.04471833540272408, 0.030006814627895072]
# camelsch_streamflow_8183  p=4 q=0
# phi = [ 1.32965079 -0.7478719   0.40758963 -0.10746793]
# theta = []
# R_2 = 0.7685028215408174
# se_beta = [0.030322822779558105, 0.04826339272938611, 0.04826404860790467, 0.030321046375571228]
# camelsch_streamflow_8183_d1  p=3 q=0
# phi = [ 0.39064336 -0.35918506  0.04628432]
# theta = []
# R_2 = 0.19022723687736434
# se_beta = [0.02997564497817271, 0.03032822813836525, 0.029976815826624433]
# camelsch_streamflow_81  p=3 q=0
# phi = [ 1.2410429  -0.6771721   0.33453013]
# theta = []
# R_2 = 0.7420380099133933
# se_beta = [0.05097314149464376, 0.07408783460004402, 0.05097224111773126]
# camelsch_streamflow_81  p=4 q=0
# phi = [ 1.28817563 -0.77413387  0.5133167  -0.14537688]
# theta = []
# R_2 = 0.7601657853047998
# se_beta = [0.05147396046129712, 0.07907659933663176, 0.07908462323509598, 0.051473988192846154]

def test_test_model():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    phi = [1.30134078, -0.67576837, 0.26822102]
    theta = []
    se_beta = [0.030009693485665807, 0.04471833540272408, 0.030006814627895072]
    m = 6
    significance_level = 0.05
    b_significant_arima, b_significant_para = inter.test_model(x, phi, theta, se_beta, m, significance_level)
    print("b_significant_arima = " + str(b_significant_arima))
    print("b_significant_para = " + str(b_significant_para))
# b_significant_arima = False
# b_significant_para = [True, True, True]

def test_test_arch():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    phi = [1.30134078, -0.67576837, 0.26822102]
    q = 3
    significance_level = 0.05
    b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = inter.test_arch(x, phi, q, significance_level)
    print("b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = " + str([b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM]))
# camelsch_streamflow_8183  q=3
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]

def test_degree_arch():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    phi = [1.30134078, -0.67576837, 0.26822102]
    acf, pacf = inter.degree_arch(x, phi)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\acf.txt', acf)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\pacf.txt', pacf)
    print("n_acf = " + str(len(acf)))
    print("n_pacf = " + str(len(pacf)))
# n_acf = 731
# n_pacf = 731

def test_arch_parameter():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    phi = [1.30134078, -0.67576837, 0.26822102]
    p = 3
    q = 4
    a0, R_20, delta_20, a1, R_21, y1, a2, R_22, y2, theta1 = inter.arch_parameter(x, phi, p, q)
    print("a0 = " + str(a0))
    print("R_20 = " + str(R_20))
    print("delta_20 = " + str(delta_20))
    print("a1 = " + str(a1))
    print("R_21 = " + str(R_21))
    print("a2 = " + str(a2))
    print("R_22 = " + str(R_22))
    print("theta1 = " + str(theta1))
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y1.txt', y1)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y2.txt', y2)
# camelsch_streamflow_8183
# a0 = [7.19334832e+03 3.41621327e-01]
# R_20 = 0.11670513665147611
# delta_20 = -2.300613862270624e-11
# a1 = [ 7.83544295e+02  4.08663609e-02 -2.61010083e-03  2.04775580e-02
#   5.91862621e-03]
# R_21 = 0.15476001469339426
# camelsch_streamflow_8183  p=3 q=4
# a0 = [6.93353764e+03 3.50556876e-01]
# R_20 = 0.12289020305449845
# delta_20 = 0.21365016209392332
# a1 = [ 8.71910230e+02  3.81756924e-02 -5.53395332e-03  2.21789260e-02
#   6.50269592e-03]
# R_21 = 0.1605177614336352
# a2 = [8.68204977e+02 3.41215781e-02 0.00000000e+00 2.05339673e-02
#  6.24759671e-03]
# R_22 = 0.1638094288492594
# theta1 = [ 1.30134078e+00 -6.75768370e-01  2.68221020e-01  8.68204977e+02
#   3.41215781e-02  0.00000000e+00  2.05339673e-02  6.24759671e-03]


def test_arch_model():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    phi = [1.30134078, -0.67576837, 0.26822102]
    theta = [8.68204977e+02, 3.41215781e-02, 0.00000000e+00, 2.05339673e-02, 6.24759671e-03]
    p = 3
    q = 4
    (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_, nse,
     rmse, max_abs_error) = inter.arch_model(x, theta, p ,q)


def test_cal_lose_ratio():
    inter = Interpolation()
    x = camelsch_streamflow_8487
    lose_ratio = inter.cal_lose_ratio(x)
    print("lose_ratio = " + str(lose_ratio))
# lose_ratio = 0.05270362765229295

def test_lose_index():
    inter = Interpolation()
    range = 100
    n = 10
    index = inter.lose_index(range, n)
    print("index = " + str(index))
# index = [52 74 85 87 18 90 39 20 52 78]
