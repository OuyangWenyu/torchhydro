
import pandas as pd
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
    x = ystl().pet
    stl = STL(x)
    stl._cycle_subseries(x)
    print(stl.cycle_subseries)
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
    x = ystl().pet
    stl = STL(x)
    xx = x
    result1 = stl.moving_average_smoothing(101, xx)
    result2 = stl.moving_average_smoothing(101, result1)
    result3 = stl.moving_average_smoothing(61, result2)
    pet_mas = pd.DataFrame({"pet": xx, "result1": result1, "result2": result2, "result3": result3})
    pet_mas.index.name = "time"
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_moving_average_smoothing.csv"
    pet_mas.to_csv(file_name, sep=" ")
    print(pet_mas)
# PASSED             [100%]
# time  pet  result1  result2  result3
# 0     1.20     1.20     1.20     1.20
# 1     1.30     1.30     1.30     1.30
# 2     0.90     0.90     0.90     0.90
# 3     0.55     0.55     0.55     0.55
# 4     0.85     0.85     0.85     0.85
# ...    ...      ...      ...      ...
# 5835  0.65     0.65     0.65     0.65
# 5836  0.55     0.55     0.55     0.55
# 5837  0.55     0.55     0.55     0.55
# 5838  0.50     0.50     0.50     0.50
# 5839  0.60     0.60     0.60     0.60
#
# [5840 rows x 4 columns]

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
    pet_stl = pd.DataFrame({"pet": x, "trend": trend, "season": season, "residuals": residuals})
    pet_stl.index.name = "time"
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_stl.csv"
    pet_stl.to_csv(file_name, sep=" ")
    print(pet_stl)
# PASSED                           [100%]
# time  pet     trend    season  residuals
# 0     1.20  1.733627 -0.533627        0.0
# 1     1.30  1.728555 -0.428555        0.0
# 2     0.90  1.723637 -0.823637        0.0
# 3     0.55  1.718873 -1.168873        0.0
# 4     0.85  1.714265 -0.864265        0.0
# ...    ...       ...       ...        ...
# 5835  0.65  1.354457 -0.704457        0.0
# 5836  0.55  1.354057 -0.804057        0.0
# 5837  0.55  1.353730 -0.803730        0.0
# 5838  0.50  1.353479 -0.853479        0.0
# 5839  0.60  1.353301 -0.753301        0.0
#
# [5840 rows x 4 columns]


def test_season_post_smoothing():
    x = ystl().pet
    stl = STL(x)
    trend, season, residuals = stl.outer_loop()
    post_season = stl.season_post_smoothing(season)
    pet_post_season = pd.DataFrame({"pet": x, "trend": trend, "season": season, "residuals": residuals, "post_season": post_season})
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_post_season.csv"
    pet_post_season.to_csv(file_name, sep=" ")
    print(pet_post_season)
