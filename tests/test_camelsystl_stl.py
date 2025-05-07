
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
    x = ystl().pet
    stl = STL(x)
    xx = [1, 2, 3, 4, 5]
    y = x[:5]
    y_ = stl.weight_least_squares(xx, y, degree=2)
    print(y_)
    # PASSED[100 %]
    # 0.7283553118882566

def test_loess():
    x = ystl().pet
    stl = STL(x)
    xx = x[:30]
    result = stl.loess(7, xx)
    print(xx)
    print(result)
    # PASSED[100 %]
    # [1.2, 1.3, 0.9, 0.55, 0.85, 1.15, 0.9, 0.85, 0.7, 0.7, 0.8, 0.95, 1.05, 0.75, 0.6, 0.55, 1.1, 1.75, 1.3, 0.8, 1.05, 1.15, 1.25, 1.85, 1.8, 1.0, 0.75, 0.45, 0.35, 0.95]
    # [1.1037545787545793, 0.9316172957477307, 0.8571906354515053, 0.8671484312788658, 0.911096512183469,
    #  0.9140600944948774, 0.8478738652651697, 0.7736834421617028, 0.7648603811647291, 0.8275601210383823,
    #  0.887389180867442, 0.8759582205234381, 0.7869525402134102, 0.7294712533842974, 0.8482109677761855,
    #  1.0903580718798112, 1.2446912990391255, 1.2305542283803155, 1.1336319477623829, 1.0762820512820515,
    #  1.1775654297393432, 1.401495726495727, 1.5299636353984183, 1.4410933269628923, 1.1683455433455434,
    #  0.8242474916387965, 0.6125497690715083, 0.6255547592504115, 0.703348463131072, 0.5529304029304032]

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
    pet_post_season.index.name = "time"
    file_name = r"D:\minio\waterism\datasets-origin\camels\camels_ystl\pet_post_season.csv"
    pet_post_season.to_csv(file_name, sep=" ")
    print(pet_post_season)
# PASSED                [100%]
# time  pet     trend    season  residuals  post_season
# 0     1.20  1.733627 -0.533627        0.0    -0.533627
# 1     1.30  1.728555 -0.428555        0.0    -0.428555
# 2     0.90  1.723637 -0.823637        0.0    -0.982103
# 3     0.55  1.718873 -1.168873        0.0    -0.905982
# 4     0.85  1.714265 -0.864265        0.0    -0.733111
# ...    ...       ...       ...        ...          ...
# 5835  0.65  1.354457 -0.704457        0.0    -0.767987
# 5836  0.55  1.354057 -0.804057        0.0    -0.822173
# 5837  0.55  1.353730 -0.803730        0.0    -0.818743
# 5838  0.50  1.353479 -0.853479        0.0    -0.853479
# 5839  0.60  1.353301 -0.753301        0.0    -0.753301
#
# [5840 rows x 5 columns]


def test_decomposition():
    x = ystl().pet
    stl = STL(x)
    decomposition = stl.decomposition()
    print(decomposition)
# PASSED                        [100%]
# terminate_trend < 0.01 or terminate_season < 0.01
# outer loop = 2
# time  pet     trend    season  residuals  post_season  post_residuals
# 0     1.20  2.125740 -1.078612   0.152872    -1.155890        0.230150
# 1     1.30  2.123465 -1.071351   0.247885    -1.259466        0.436000
# 2     0.90  2.122294 -1.247946   0.025653    -1.305311        0.083017
# 3     0.55  2.122126 -1.475418  -0.096707    -1.254098       -0.318028
# 4     0.85  2.121422 -1.386995   0.115573    -1.168501       -0.102921
# ...    ...       ...       ...        ...          ...             ...
# 5835  0.65  1.583070 -1.305735   0.372665    -1.284687        0.351617
# 5836  0.55  1.584953 -1.305939   0.270986    -1.293339        0.258386
# 5837  0.55  1.588345 -1.263358   0.225012    -1.287361        0.249016
# 5838  0.50  1.592190 -1.332506   0.240316    -1.285603        0.193414
# 5839  0.60  1.595260 -1.233022   0.237763    -1.288563        0.293303
#
# [5840 rows x 6 columns]
