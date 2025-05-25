
from tests.test_camelsystl_stl_mi import Ystl
from torchhydro.datasets.scalers import SlidingWindowScaler

def test_data():
    x = Ystl().pet
    xx = x[:30]
    print(xx)
# [1.2, 1.3, 0.9, 0.55, 0.85, 1.15, 0.9, 0.85, 0.7, 0.7, 0.8, 0.95, 1.05, 0.75, 0.6, 0.55, 1.1, 1.75, 1.3, 0.8, 1.05, 1.15, 1.25, 1.85, 1.8, 1.0, 0.75, 0.45, 0.35, 0.95]

def test_cal_stat():
    x = Ystl().prcp
    sws = SlidingWindowScaler()
    