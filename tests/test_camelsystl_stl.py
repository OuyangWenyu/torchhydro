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
        self.var_list = ["streamflow", "discharge_vol1", "discharge_vol2", "discharge_vol3",]
        self.data = None
        # self.read_data()
        self.pet = None
        self.pet_list = ["pet"]
        self.read_pet()

    def read_data(self):
        data = self.datasource.read_ts_xrdataset(
            self.basin,
            self.time_range,
            self.var_list,
        )
        data1 = data.streamflow.to_dataframe()
        data2 = data.discharge_vol1.to_dataframe()
        data3 = data.discharge_vol2.to_dataframe()
        data4 = data.discharge_vol3.to_dataframe()
        data1.drop(axis=0, index=("1000","1992-02-29"), inplace=True)
        data2.drop(axis=0, index=("1000","1992-02-29"), inplace=True)
        data3.drop(axis=0, index=("1000","1992-02-29"), inplace=True)
        data4.drop(axis=0, index=("1000","1992-02-29"), inplace=True)
        data1 = data1.values[:, 0]
        data2 = data2.values[:, 0]
        data3 = data3.values[:, 0]
        data4 = data4.values[:, 0]
        self.data = data1.tolist() + data2.tolist() + data3.tolist() + data4.tolist()
        # print(self.data)

    def read_pet(self):
        data = self.datasource.read_ts_xrdataset(
            self.basin,
            self.time_range,
            self.pet_list
        )
        data = data.pet.to_dataframe()
        data.drop(axis=0, index=("1000", "1992-02-29"), inplace=True)
        data = data.values[:, 0]
        self.pet = data.tolist() + data.tolist() + data.tolist() + data.tolist()

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

def test_extend_subseries():
    x = ystl().data
    stl = STL(x)
    subseries = stl._cycle_subseries(x)
    extend_subseries = stl._extend_subseries(subseries)
    print(len(extend_subseries))
    print(len(extend_subseries[0]))
# 365
# 18

def test_de_extend_subseries():
    x = ystl().data
    stl = STL(x)
    subseries = stl._cycle_subseries(x)
    extend_subseries = stl._extend_subseries(subseries)
    de_extend_subseries = stl._de_extend_subseries(extend_subseries)
    print(len(de_extend_subseries))
    print(len(de_extend_subseries[0]))
# 365
# 16

def test_recover_series():
    x = ystl().data
    stl = STL(x)
    stl._cycle_subseries(x)
    print(len(stl.cycle_subseries))
    series = stl._recover_series(stl.cycle_subseries)
    print(series[:10])
# 365
# [165.8, 164.1, 158.8, 158.0, 156.2, 144.8, 137.6, 134.6, 130.3, 128.2]

def test_neighborhood_weight():
    x = ystl().data
    stl = STL(x)
    weight = stl._neighborhood_weight()
    print(weight)
# [0, np.float64(0.669921875), np.float64(1.0), np.float64(0.669921875), 0]
# [0.0, 0.669921875, 1.0, 0.669921875, 0.0]

def test_moving_average_smoothing():
    x = ystl().data
    stl = STL(x)
    xx = x[:365]
    result1 = stl.moving_average_smoothing(15, xx)
    result2 = stl.moving_average_smoothing(7, result1)
    result3 = stl.moving_average_smoothing(3, result2)
    print("\nresult1")
    print(result1)
    print("\nresult2")
    print(result2)
    print("\nresult3")
    print(result3)
# [165.8, 164.1, 158.8, 158.0, 131.35714285714283, 127.14285714285714, 123.07142857142857, 118.81428571428572, 114.74285714285715,
# 112.12857142857145, 110.55714285714285, 108.42857142857144, 106.5, 105.07142857142857, 103.38571428571429, 101.42857142857143,
# 99.45714285714284, 97.64285714285714, 96.02857142857144, 93.74285714285715, 91.47142857142858, 89.42857142857143, 87.12857142857145,
# 85.34285714285714, 83.6, 82.3, 81.38571428571429, 82.22857142857143, 90.57142857142857, 102.5, 113.74285714285715, 123.47142857142856,
# 131.1, 136.12857142857143, 133.4857142857143, 127.08571428571427]

def test_weight_least_squares():
    x = ystl().data
    stl = STL(x)
    xx = [1, 2, 3, 4, 5]
    y = x[:5]
    y_ = stl.weight_least_squares(xx, y)
    print(y_)
    # [169.23839733 - 3.05]
    # 157.03839732888136

def test_loess():
    x = ystl().data
    stl = STL(x)
    xx = x[:365]
    result = stl.loess(7, xx)
    print(xx)
    print(result)
    # [165.8, 164.1, 158.8, 153.6181773236651, 146.47608061022692, 138.93107166399847, 133.79015679442506,
    #  130.69068179677933, 128.53937046802895, 127.30696157830303, 126.77608296449756, 124.81859167529898,
    #  121.04208729635558, 118.02256097560972, 116.7978411338167, 115.68182738487614, 113.826546755815,
    #  111.21521800546185, 108.2443167906582, 105.11882945663429, 102.34115265090874, 100.20908983896787,
    #  98.32191355118182, 96.5792282700819, 94.78967652321306, 93.5114841322158, 93.15851539692999, 96.06652462567094,
    #  113.67811940860719, 145.21504614370463, 170.2376777474338

def test_inner_loop():
    x = ystl().data
    stl = STL(x)
    trend = [0]*stl.length
    ni = 2
    trend_i, season_i = stl.inner_loop(x, trend)
    print(trend_i)
    print(season_i)
# [103.76759812692383, 102.20164037269491, 99.738328772511, 99.62958843004897, 98.2223414270767, 96.79183156477477, 96.53564693981045, 97.75621973426932,
# 99.36651326823953, 99.76874293165153, 98.26840677853927, 95.5474021764428, 92.89262740821509, 90.76571673210684, 88.66649746752341, 86.13674183015392,
# 83.2122625967781, 80.24603476363518, 77.55473379197005, 75.16664415298915, 72.9937418115444, 71.4692646445855, 71.16668204417813, 71.94585155934128,
# 73.23464587263436, 74.33493364446744, 74.97696990580846, 77.69157673072425, 85.67355032146186, 97.9056209327276, 109.50081442747407, 115.65118651916113,
# 114.92302464599325, 110.21141633508216, 105.57688065472742, 102.973535785042, 103.73671137632715, 109.04560034616493]

def test_outer_loop():
    x = ystl().pet
    stl = STL(x)
    trend, season, residuals = stl.outer_loop()
    print("\ntrend")
    print(trend)
    print("\nseason")
    print(season)
    print("\nresiduals")
    print(residuals)
# residuals
# [0.10385388 0.14531884 0.04918184 ... 0.01569811 0.00147881 0.00119412]
