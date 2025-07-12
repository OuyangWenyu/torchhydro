
import numpy as np

from hydrodataset import CamelsCh
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

def test_degree():
    inter = Interpolation()
    acf, pacf = inter.degree(camelsch_streamflow_81)
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
