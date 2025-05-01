from hydrodataset import CamelsYstl
from torchhydro.datasets.mi_stl import STL


def test_dataset():
    camelsystl = CamelsYstl()
    basin = camelsystl.gage
    print(basin)


class ystl():
    def __init__(self):
        self.datasource = CamelsYstl()
        self.basin = ["1000",]
        self.time_range = ["1990-01-01","1994-01-01"]
        self.var_list = ["streamflow",]
        self.data = None
        self.read_data()

    def read_data(self):
        data = self.datasource.read_ts_xrdataset(
            self.basin,
            self.time_range,
            self.var_list,
        )
        self.data = data.streamflow.data[0].T
        # print(self.data)

def test_read_data():
    x = ystl().data
    print(x)
# [165.8 164.1 158.8 ...  73.2  71.1  71.3]

def test_cycle_subseries():
    x = ystl().data
    stl = STL(x)
    stl._cycle_subseries()
    print(len(stl.cycle_subseries))
# PASSED                      [100%]365

def test_weight_function():
    u = [1, 0.5, 0, 0.5, 1]
    n = 5
    w = []
    stl = STL()
    for i in range(n):
        w_i = stl.weight_function(u[i], 3)
        w.append(w_i)
    print(w)
# [0, 0.5625, 1, 0.5625, 0]
# [0, 0.669921875, 1, 0.669921875, 0]

def test_recover_series():
    x = ystl().data
    stl = STL()
    stl._cycle_subseries()
    print(len(stl.cycle_subseries))
    series = stl._recover_series()
    print(series[:10])

def test_moving_average_smoothing():
    x = ystl().data
    stl = STL()
    result = stl.moving_average_smoothing(7, x)
    print(result[:10])


