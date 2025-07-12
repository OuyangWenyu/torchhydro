
from hydrodataset import CamelsCh
from test_arch_data import camelsch_streamflow
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

def test_readdat():
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
    b_ = inter.smooth_test()
    print("b_ = " + str(b_))
