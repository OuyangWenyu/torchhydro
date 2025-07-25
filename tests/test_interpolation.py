
import numpy as np

from hydrodataset import CamelsCh, Camels
from test_arch_data import (
    camelsch_streamflow_8487, camelsch_streamflow_8183, camelsch_streamflow_8183_d1,
    camelsch_streamflow_81, camelsch_streamflow_90, camelsch_streamflow_30_3,
    camelsus_streamflow_r516, camelsus_streamflow_01013500_80,
    camelsus_streamflow_01013500_8081, camelsus_streamflow_01013500_8081_d1, camelsus_streamflow_01013500_8081_d2,
    camelsus_streamflow_01013500_80_005nan, camelsus_streamflow_01013500_80_01nan, camelsus_streamflow_01013500_80_015nan,
    camelsus_streamflow_01013500_80_025nan, camelsus_streamflow_01013500_80_030nan, camelsus_streamflow_01013500_80_035nan,
    e_01013500_80,
)
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

def test_delete_nan():
    inter = Interpolation()
    x_dnan = inter.delete_nan()
    n_x_dnan = len(x_dnan)
    print("n_x_dnan = ", n_x_dnan)
    for i in range(n_x_dnan):
        print("x_dnan[" + str(i) + "] = ", x_dnan[i].shape)
# n_x_dnan =  4
# x_dnan[0] =  (12692,)
# x_dnan[1] =  (12692,)
# x_dnan[2] =  (12692,)
# x_dnan[3] =  (12235,)

def test_cal_lose_ratio():
    inter = Interpolation()
    lose_ratio, lose_type = inter.cal_lose_ratio()
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\lose_ratio.txt', lose_ratio)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\lose_type.txt', lose_type)
    print("lose_ratio = " + str(lose_ratio))
    print("lose_type = " + str(lose_type))
# self.gage_id = ["01013500", "01022500", "01030500", "01187300"]
# lose_ratio = [0.007196495619524401, 0.007196495619524401, 0.007196495619524401, 0.04294430538172711]
# self.gage_id = self.datasource.gage
# lose_ratio = [0.007196495619524401, 0.007196495619524401, 0.007196495619524401, 0.007196495619524401,
#               0.0, 0.0, 0.0, 0.0, 0.0, 0.006492490613266622, 0.003832916145181442, 0.04294430538172711,
#               0.04294430538172711, 0.03574780976220271, 0.0032853566958698233, 0.00563204005006257,
#               0.0023466833541927468, 0.0033635794743429592, 0.0066489361702127825
# lose_type = {'intact_series': [4, 5, 6, 7, 8, 39, 44, 52, 54, 55, 56, 57, 58, 63, 64, 65, 81, 91, 92, 101, 104, 125,
#                                127, 132, 133, 152, 178, 180, 181, 184, 186, 189, 195, 197, 205, 206, 207, 208, 211,
#                                212, 213, 214, 220, 228, 232, 237, 244, 245, 247, 252, 258, 262, 267, 270, 271, 292,
#                                293, 295, 296, 301, 304, 306, 311, 313, 314, 315, 316, 317, 321, 325, 326, 327, 369,
#                                377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 390, 392, 393, 395, 396, 398,
#                                400, 402, 403, 404, 405, 407, 408, 409, 412, 413, 424, 428, 430, 431, 444, 447, 448,
#                                452, 453, 456, 457, 458, 462, 463, 465, 466, 467, 470, 471, 472, 473, 474, 475, 477,
#                                478, 484, 489, 494, 496, 511, 512, 514, 516, 517, 519, 524, 525, 528, 529, 530, 533,
#                                534, 535, 538, 543, 548, 549, 550, 552, 553, 554, 556, 559, 560, 561, 564, 566, 568,
#                                571, 577, 579, 581, 585, 586, 591, 597, 598, 599, 604, 605, 606, 607, 608, 609, 610,
#                                620, 621, 622, 624, 625, 626, 628, 629, 631, 632, 634, 635, 637, 638, 640, 648, 649,
#                                650, 651, 652, 654, 655, 658, 662, 664, 666, 667, 668, 670],
#              'slight_lose': [0, 1, 2, 3, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 38, 40, 41, 43, 45, 46, 47,
#                              49, 51, 53, 59, 60, 61, 62, 66, 67, 68, 70, 72, 75, 76, 79, 80, 82, 83, 84, 85, 86, 87,
#                              89, 90, 93, 94, 95, 96, 97, 98, 99, 100, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113,
#                              114, 115, 116, 117, 119, 120, 121, 123, 124, 130, 131, 134, 140, 143, 144, 145, 147, 148,
#                              149, 150, 151, 153, 154, 155, 157, 158, 159, 160, 161, 163, 165, 167, 168, 169, 170, 171,
#                              176, 185, 187, 188, 190, 191, 192, 193, 194, 196, 198, 199, 200, 209, 210, 221, 223, 238,
#                              240, 241, 242, 243, 246, 248, 249, 250, 251, 254, 255, 256, 257, 259, 260, 261, 263, 264,
#                              265, 269, 273, 275, 276, 277, 279, 281, 282, 283, 285, 286, 287, 288, 289, 294, 297, 298,
#                              299, 300, 302, 303, 305, 307, 308, 310, 312, 318, 319, 320, 322, 323, 332, 334, 335, 336,
#                              337, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 356, 357, 358,
#                              359, 360, 361, 362, 364, 365, 367, 368, 370, 371, 372, 373, 374, 375, 376, 388, 391, 394,
#                              399, 401, 406, 410, 411, 414, 415, 416, 417, 418, 420, 423, 426, 427, 429, 432, 433, 435,
#                              437, 438, 440, 441, 442, 446, 450, 451, 454, 455, 460, 461, 464, 476, 479, 480, 481, 482,
#                              483, 485, 486, 488, 490, 491, 492, 493, 497, 498, 499, 500, 502, 503, 504, 505, 506, 518,
#                              521, 526, 527, 531, 536, 539, 541, 545, 546, 547, 551, 555, 557, 558, 562, 563, 567, 569,
#                              570, 572, 573, 575, 576, 578, 580, 582, 583, 587, 588, 590, 592, 593, 594, 595, 600, 611,
#                              612, 619, 623, 627, 630, 636, 639, 643, 644, 645, 646, 647, 653, 659, 660],
#              'dram_lose': [11, 12, 13, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 37, 42, 48, 69, 73, 74, 77, 78, 88, 102,
#                            122, 128, 135, 139, 142, 146, 156, 162, 173, 174, 179, 182, 202, 203, 204, 215, 216, 217,
#                            218, 219, 222, 224, 226, 227, 229, 231, 233, 234, 235, 236, 253, 266, 274, 284, 291, 329,
#                            351, 355, 419, 439, 443, 468, 469, 487, 507, 508, 513, 520, 532, 540, 574, 584, 589, 601,
#                            613, 614, 615, 616, 641, 669],
#              'jot_lose': [36, 50, 126, 129, 136, 138, 141, 164, 172, 175, 183, 225, 239, 272, 278, 280, 290, 309, 328,
#                           330, 363, 366, 397, 422, 425, 445, 449, 459, 501, 515, 522, 523, 565, 602, 603, 633, 642,
#                           657, 661, 663, 665],
#              'moderate_lose': [28, 71, 118, 137, 166, 177, 201, 230, 268, 324, 331, 333, 338, 354, 389, 421, 434, 436,
#                                495, 509, 510, 537, 542, 544, 596, 617, 618, 656],
#              'stack_lose': [],
#              'serious_lose': []}

def test_cal_7_stat_inds():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_80
    statistics_indices = inter.cal_7_stat_inds(x)
    print("[num_point, mean, std, min_, p25, p50, p75, max_]")
    print(statistics_indices)
# camelsus_streamflow_01013500_80
# [num_point, mean, std, min_, p25, p50, p75, max_]
# [364, 959.434065934066, 1030.4250102247397, 195.0, 439.75, 579.5, 922.0, 5560.0]

def test_statistics_indices():
    inter = Interpolation()
    stat_inds = inter.statistics_indices()
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\stat_inds.txt', stat_inds)
    print("stat_inds = " + str(stat_inds.shape))
# stat_inds = (4, 8)
# 12692 	1545.27 	1795.16 	42.00 	480.00 	868.00 	1820.00 	17900.00
# 12692 	508.64 	591.30 	12.00 	157.00 	314.00 	630.00 	6790.00
# 12692 	2751.96 	3456.10 	32.00 	603.00 	1420.00 	3440.00 	25900.00
# 12235 	43.44 	81.88 	0.19 	7.90 	21.00 	47.00 	2350.00
# stat_inds = (671, 8)
# gage	num_point	mean	std	min_	p25	p50	p75	max_
# "01013500"	12692 	1545.273 	1795.160 	42.000 	480.000 	868.000 	1820.000 	17900.000
# "01022500"	12692 	508.640 	591.296 	12.000 	157.000 	314.000 	630.000 	6790.000
# "01030500"	12692 	2751.956 	3456.101 	32.000 	603.000 	1420.000 	3440.000 	25900.000
# "01031500"	12692 	632.019 	1057.706 	6.200 	130.000 	276.000 	679.000 	31700.000

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

def test_read_data():
    inter = Interpolation()
    inter.read_data()
    n_x = inter.x.size
    print("n_x = " + str(n_x))
# n_x = 1461
# n_x = 12784

def test_smooth_test():
    inter = Interpolation()
    x = inter.x_dnan[0]
    b_ = inter.smooth_test(x)
    print("b_ = " + str(b_))
# camelsch_streamflow_8081
# b_ = True
# acmelsch_streamflow_8081_d1
# b_ = True
# camelsus_streamflow_01013500_80
# b_ = False
# camelsus_streamflow_01013500_8081
# b_ = False
# camelsus_streamflow_01013500_8081_d1
# b_ = True
# camelsus_streamflow_01013500_8081_d2
# b_ = True
# inter.x_dnan[0]  camelsus_streamflow_01013500
# b_ = True

def test_degree_ar():
    inter = Interpolation()
    x = inter.x_dnan[0]
    phi = [1.85816724, -0.86378065]
    acf, pacf = inter.degree_ar(x)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\acf.txt', acf)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\pacf.txt', pacf)
    print("n_acf = " + str(len(acf)))
    print("n_pacf = " + str(len(pacf)))
# camelsch_streamflow_8081d1
# n_acf = 731
# n_pacf = 731
# camelsch_streamflow_80
# n_acf = 244
# n_pacf = 244
# camelsus_streamflow_01013500_80
# n_acf = 243
# n_pacf = 243
# camelsus_streamflow_01013500_80   phi=[1.85816724, -0.86378065]
# n_acf = 243
# n_pacf = 243
# camelsus_streamflow_01013500_8081
# n_acf = 488
# n_pacf = 488
# camelsus_streamflow_01013500_8081_d1
# n_acf = 487
# n_pacf = 487
# camelsus_streamflow_01013500_8081_d2
# n_acf = 487
# n_pacf = 487
# inter.x_dnan[0]  camelsus_streamflow_01013500
# n_acf = 501
# n_pacf = 501

def test_arma_parameters():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_8081_d1
    x = (inter.x_dnan[0]).tolist()
    p = 2
    q = 0
    phi, theta, R_2, se_beta = inter.arima_parameter(x, p, q)
    residual, y_t, mean_residual, residual_center, residual_center_2 = inter.arch.x_residual_via_parameters(x, phi, b_y=True, b_center=True)
    print("phi = " + str(phi))
    print("theta = " + str(theta))
    print("R_2 = " + str(R_2))
    print("se_beta = " + str(se_beta))
    print("mean_residual = " + str(mean_residual))
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_t.txt', y_t)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual.txt', residual)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_center.txt', residual_center)
# camelsus_streamflow_01013500_80  p=2
# phi = [ 1.85816724 -0.86378065]
# theta = []
# R_2 = 0.9972033775619407
# se_beta = [0.02650141929080101, 0.02650152518211935]
# mean_residual = 5.350273893593821
# camelsus_streamflow_01013500_8081   p=2
# phi = [ 0.40816167 -2.1158528 ]
# theta = []
# R_2 = 0.9403979532213149
# se_beta = [1.27045545049684, 1.2701498610924138]
# mean_residual = 3923.2654513088487
# camelsus_streamflow_01013500_8081_d1 p=1
# phi = [0.6830197]
# theta = []
# R_2 = 0.4665011262554114
# se_beta = [0.027052426340113594]
# mean_residual = 0.22545816760535747
# camelsus_streamflow_01013500_8081_d1 p=2
# phi = [0.65009565 0.04819934]
# theta = []
# R_2 = 0.4677339258460142
# se_beta = [0.037019421068768536, 0.03702018474274401]
# mean_residual = 0.21947227756428117
# inter.x_dnan[0]  camelsus_streamflow_01013500
# phi = [ 1.6654097 -0.6711734]
# theta = []
# R_2 = 0.99345536607176
# se_beta = [0.006584719668673316, 0.0065848039599755775]
# mean_residual = 8.896472090844359


def test_test_arima_model():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_8081
    x = camelsus_streamflow_01013500_8081_d1
    x = (inter.x_dnan[0]).tolist()
    phi = [1.85816724, -0.86378065]
    theta = []
    se_beta = [0.02650141929080101, 0.02650152518211935]
    phi = [0.40816167, -2.1158528]
    theta = []
    se_beta = [1.27045545049684, 1.2701498610924138]
    phi = [0.65009565, 0.04819934]
    theta = []
    se_beta = [0.037019421068768536, 0.03702018474274401]
    phi = [1.6654097, -0.6711734]
    theta = []
    se_beta = [0.006584719668673316, 0.0065848039599755775]
    m = 6
    significance_level = 0.05
    b_significant_arima, b_significant_para = inter.test_arima_model(x, phi, theta, se_beta, m, significance_level)
    print("b_significant_arima = " + str(b_significant_arima))
    print("b_significant_para = " + str(b_significant_para))
# camelsch_streamflow_8081
# b_significant_arima = False
# b_significant_para = [True, True, True]
# camelsus_streamflow_01013500_80
# b_significant_arima = False
# b_significant_para = [True, True]
# camelsus_streamflow_01013500_8081
# b_significant_arima = False
# b_significant_para = [False, True]  # todo:
# camelsus_streamflow_01013500_8081_d1
# b_significant_arima = True
# b_significant_para = [True, False]
# inter.x_dnan[0]  camelsus_streamflow_01013500
# b_significant_arima = False
# b_significant_para = [True, True]

def test_degree_arch():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_8081
    x = camelsus_streamflow_01013500_8081_d1
    x = (inter.x_dnan[0]).tolist()
    phi = [1.85816724, -0.86378065]
    phi = [0.40816167, -2.1158528]
    phi = [0.65009565, 0.04819934]
    phi = [1.6654097, -0.6711734]
    acf, pacf = inter.degree_arch(x, phi)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\acf.txt', acf)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\pacf.txt', pacf)
    print("n_acf = " + str(len(acf)))
    print("n_pacf = " + str(len(pacf)))
# camelsus_streamflow_01013500_80
# n_acf = 243
# n_pacf = 243
# camelsus_streamflow_01013500_8081
# n_acf = 488
# n_pacf = 488
# camelsus_streamflow_01013500_8081_d1
# n_acf = 487
# n_pacf = 487
# inter.x_dnan[0]  camelsus_streamflow_01013500
# n_acf = 501
# n_pacf = 501

def test_test_arch():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_8081
    x = camelsus_streamflow_01013500_8081_d1
    x = (inter.x_dnan[0]).tolist()
    phi = [1.85816724, -0.86378065]
    phi = [0.40816167, -2.1158528]
    phi = [0.65009565, 0.04819934]
    phi = [1.6654097, -0.6711734]
    q = 4
    q = 2
    significance_level = 0.05
    b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = inter.test_arch(x, phi, q, significance_level)
    print("b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = " + str([b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM]))
# camelsch_streamflow_8183  q=3
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]
# camelsus_streamflow_01013500_80  q=4
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]
# camelsus_streamflow_01013500_8081  q=2
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]
# camelsus_streamflow_01013500_8081_d1  q=2
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]
# inter.x_dnan[0]  camelsus_streamflow_01013500
# b_arch_Q, b_arch_LM, b_arch_F, b_arch_bpLM = [True, False, True, True]

def test_arch_parameter():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    x = camelsus_streamflow_01013500_8081
    x = camelsus_streamflow_01013500_8081_d1
    x = (inter.x_dnan[0]).tolist()
    phi = [1.30134078, -0.67576837, 0.26822102]
    phi = [1.85816724, -0.86378065]
    phi = [0.40816167, -2.1158528]
    phi = [0.65009565, 0.04819934]
    phi = [1.6654097, -0.6711734]
    p = 2
    q = 2
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
# camelsus_streamflow_01013500_80  phi=[1.85816724, -0.86378065]
# ----------iloop = 0----------
# [ 1.85816724e+00 -8.63780650e-01  2.61970529e+02  1.08103901e+00
#   7.16330945e-01  5.46704648e-03  1.44462171e-02]
# gradient = [ 4.17496677e+02  4.24632316e+02 -5.28985504e-03 -9.16735223e-02
#  -2.91980131e-01 -4.06593352e-01 -4.52102413e-01]
# L_theta = -1730.23997619875
# distance_grad_0 = 354616.5422787547
# likelihood_theta_1_0 = 340.19212886107243
# distance_theta_1_0 = 360.0570483354315
# theta1 = [ 1.85816724e+00 -8.63780650e-01  2.61970529e+02  1.08103901e+00
#   7.16330945e-01  5.46704648e-03  1.44462171e-02]
# ----------end----------
# gradient = [ 4.41971512e+02  4.59126299e+02  9.23533817e-03 -8.44832067e-02
#  -2.79028077e-01 -3.54697752e-01 -3.90060591e-01]
# L_theta = -1723.6664110461757
# distance_grad_0 = 406136.13908443606
# likelihood_theta_1_0 = 8.809688370092772e-05
# distance_theta_1_0 = 0.0223975330243141
# iloop = 190
# a0 = [1.86326347e+03 3.79049461e-01]
# R_20 = 0.1436784076950352
# delta_20 = 0.06429598408523504
# a1 = [ 2.43378145e+02  6.02968571e-02 -4.32332382e-03  6.09137318e-03
#   1.57561812e-02]
# R_21 = 0.16691964515893049
# a2 = [2.43036466e+02 5.86986201e-02 0.00000000e+00 5.46704648e-03
#  1.44462171e-02]
# R_22 = 0.1919276424717406
# theta1 = [ 1.85816724e+00 -8.63780650e-01  3.17744057e+02  2.78526653e-01
#   5.45285923e-01  4.38002156e-03  1.15738438e-02]
# camelsus_streamflow_01013500_8081_d1   p=2  q=2
# ----------end----------
# gradient = [ 1.81173454e+02  1.01070508e+02  3.19542763e-03 -2.59508064e-02
#  -3.33178003e-02]
# L_theta = -4131.177583696567
# distance_grad_0 = 43039.06969794017
# likelihood_theta_1_0 = 9.996662083722185e-05
# distance_theta_1_0 = 0.002617318014769557
# iloop = 6975
# a0 = [7.87977642e+03 3.62083731e-01]
# R_20 = 0.13110462847595689
# delta_20 = -0.00023530795626536748
# a1 = [ 1.25145614e+03  4.32782187e-02 -8.83445284e-03]
# R_21 = 0.14056715090405925
# a2 = [1.25005277e+03 2.78953329e-02 0.00000000e+00]
# R_22 = 0.13109179475928712
# theta1 = [6.50095650e-01 4.81993400e-02 1.96972257e+03 4.96447713e-01
#  3.51953991e-01]
# inter.x_dnan[0]  camelsus_streamflow_01013500   p=2  q=2


def test_arch_model():
    inter = Interpolation()
    x = camelsch_streamflow_8183
    x = camelsus_streamflow_01013500_80
    x = (inter.x_dnan[0]).tolist()
    phi = [1.30134078, -0.67576837, 0.26822102]
    theta = [8.68204977e+02, 3.41215781e-02, 0.00000000e+00, 2.05339673e-02, 6.24759671e-03]
    phi = [1.85816724, -0.86378065]
    # theta = [1.85816724e+00, -8.63780650e-01, 2.91988282e+02, 4.26592988e-01, 5.19043732e-01, 0.00000000e+00, 4.33366213e-02]
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    nse = 0.98
    rmse = 55
    max_error = 400
    max_loop =1000
    result = inter.arch_model(x, theta, p ,q, nse, rmse, max_error, max_loop)
    if result["y_arch"] is not None:
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_arch.txt', result["y_arch"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_arima.txt', result["y_arima"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual.txt', result["residual"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_center.txt', result["residual_center"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_2.txt', result["residual_2"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\delta_2.txt', result["delta_2"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\delta.txt', result["delta"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\epsilon.txt', result["epsilon"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\e_.txt', result["e"])
        print("n_loop = " + str(result["i_loop"]))
        print("mean_residual = " + str(result["mean_residual"]))
        print("NSE = " + str(result["nse"]))
        print("RMSE = " + str(result["rmse"]))
        print("max_abs_error = " + str(result["max_abs_error"]))
# mean_residual = 5.350272800384656
# NSE = 0.997154207837962
# RMSE = 54.96901827148393
# max_abs_error = 346.1947733098914
# camelsus_streamflow_01013500_80      nse=0.98  rmse=55  max_error=400  max_loop=1000
# n_loop = 2
# mean_residual = 5.350272800384656
# NSE = 0.9972691656187858
# RMSE = 53.84731740143714
# max_abs_error = 387.33503445110455

def test_lose_index():
    inter = Interpolation()
    range = 100
    n = 10
    index = inter.lose_index(range, n)
    print("index = " + str(index))
# index = [10 17 23 29 36 37 39 54 76 91]

def test_lose_series():
    inter = Interpolation()
    x =(inter.x_dnan[0]).tolist()
    x = camelsus_streamflow_01013500_80
    n_x = len(x)
    ratio = 0.15
    lose_x = inter.lose_series(x, n_x, ratio)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\lose_x.txt', lose_x)
    print("lose_x = " + str(len(lose_x)))
# lose_x = 12692

def test_lose_set():
    inter = Interpolation()
    # x = inter.x_dnan
    x = camelsus_streamflow_01013500_80
    ratio_list = [0.05, 0.1, 0.15, 0.25, 0.3, 0.35]
    lose_set_x = inter.lose_set(x, ratio_list)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\lose_set_x.txt', lose_set_x[0])
    print("length of lose_set_x = " + str(len(lose_set_x[0])))
# x = inter.x_dnan
# length of lose_set_x = 6
# x = camelsus_streamflow_01013500_80
# length of lose_set_x = 6

def test_arch_interpolate():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_80
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    (y_arch, y_arima, residual, mean_residual, residual_center, residual_2, delta_2, delta, epsilon, e_,
     nse, rmse, max_abs_error) = inter.arch_interpolate(x, theta, p ,q)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_arch.txt', y_arch)
    print("length of y_arch = " + str(len(y_arch)))
# length of y_arch = 364

def test_interpolate():
    inter = Interpolation()
    x = inter.x_lose
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    x_inter_set = inter.interpolate(x, theta, p, q)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_inter_set.txt', x_inter_set)
    print("length of lose_set_x = " + str(len(x_inter_set)))

def test_genetate_lose_time_series_single():
    inter = Interpolation()
    x = camelsch_streamflow_90
    range = 90
    n = 9
    lose_x = inter.genetate_lose_time_series_single(x, n)
    print("lose_x")
    print(lose_x)
# lose_x
# [          nan   52.40696541   57.42164809  531.7682515  1083.277402
#   629.0248436   237.4205044   146.0614616   107.0034402    89.55799481
#    78.04541345   69.53457877           nan   63.7782881    51.80661608
#    56.68004009   67.30975477   68.12199211   67.94541877   75.39681345
#    74.09017078   70.84122144   67.66290144   68.15730677   64.44926677
#    58.97549342           nan   52.90137075   51.02969341   50.00556808
#    45.3087174    41.45941873   37.61012006   35.59718406   81.29436279
#   133.0656642    99.30484282   95.24365615           nan  130.0286029
#   145.7436296   135.5730055   114.0663735    98.13945882           nan
#    82.70694946   75.85590412   69.07548811   63.0366801    57.52759209
#    52.08913341   46.68598941   42.2363414    38.13984006   33.19578672
#    29.48774671   26.66257337   25.88565071   24.68495204   29.66432005
#    45.90906674   69.64052277   79.98772012   70.27618678           nan
#   129.7814002   234.4187577   524.1402835   534.2755928   492.9221181
#   856.804444             nan           nan 1020.45261     313.2057792
#   212.2058323   147.2268456   109.5460962    94.85519481           nan
#   125.8614722   231.0638644   494.0521874   651.8381183   688.3534837
#   660.3842677   360.2449152   223.1180643   232.4058217   255.9960191 ]

def test_genetate_lose_time_series():
    inter = Interpolation()
    x = camelsch_streamflow_30_3
    x = np.array(x)
    n = 3
    lose_x = inter.genetate_lose_time_series(x, n)
    print("lose_x")
    print(lose_x)
# lose_x
# [[  55.16150942   52.40696541   57.42164809  531.7682515            nan
#             nan  237.4205044   146.0614616   107.0034402    89.55799481
#     78.04541345   69.53457877   64.87304277   63.7782881    51.80661608
#     56.68004009   67.30975477   68.12199211   67.94541877           nan
#     74.09017078   70.84122144   67.66290144   68.15730677   64.44926677
#     58.97549342   54.80836275   52.90137075   51.02969341   50.00556808]
#  [  45.3087174    41.45941873   37.61012006   35.59718406   81.29436279
#    133.0656642    99.30484282   95.24365615  134.1604189   130.0286029
#    145.7436296            nan  114.0663735    98.13945882   89.55799481
#     82.70694946   75.85590412   69.07548811           nan   57.52759209
#     52.08913341   46.68598941   42.2363414    38.13984006   33.19578672
#     29.48774671   26.66257337   25.88565071   24.68495204           nan]
#  [  45.90906674   69.64052277   79.98772012           nan   63.38982677
#    129.7814002   234.4187577   524.1402835   534.2755928   492.9221181
#    856.804444   1522.203394   1655.516261   1020.45261     313.2057792
#    212.2058323   147.2268456   109.5460962    94.85519481   91.95939214
#    125.8614722   231.0638644   494.0521874   651.8381183   688.3534837
#             nan  360.2449152   223.1180643   232.4058217            nan]]

def test_mse():
    inter = Interpolation()
    x = camelsch_streamflow_90
    y = camelsch_streamflow_8487[:90]
    mse = inter.mse(x, y)
    print("mse = " + str(mse))
# mse = 162924.84484475685

def test_correlation_coefficient_spearman():
    inter = Interpolation()
    x = camelsch_streamflow_90
    n = 10
    rho = inter.correlation_coefficient_spearman(x, n)
    print("rho = " + str(rho))
# rho = -156.33450020041516

def test_split_series_via_nan_single_step():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_80_005nan
    p = 2
    subseries, indices = inter.split_series_via_nan_single_step(x, p)
    print("subseries = ", subseries)
    print("indices = ", indices)
# subseries = [array([ 525,  525, -100]), array([ 455,  445, -100]), array([ 390,  380, -100]),
#              array([ 280,  275, -100]), array([4630, 4470, -100]), array([1940, 1820, -100]),
#              array([ 613,  599, -100]), array([ 581,  571, -100]), array([ 306,  295, -100]),
#              array([ 201,  206, -100]), array([ 480,  525, -100]), array([ 523,  509, -100]),
#              array([1060, 1060, -100]), array([ 982,  955, -100]), array([ 854,  833, -100]),
#              array([ 921,  909, -100]), array([ 711,  709, -100]), array([ 952,  948, -100])]
# indices = [ 15  32  38  64 116 146 175 208 247 261 271 275 290 295 299 303 326 336]

def test_recover_series():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    x_nan = camelsus_streamflow_01013500_80_005nan
    indices = [15, 32, 38, 64, 116, 146, 175, 208, 247, 261, 271, 275, 290, 295, 299, 303, 326, 336]
    interpolate_value =  [522.0529597499999, 433.86422605, 369.22909769999995, 269.137409, 4306.703153299999,
                          1706.1299158000002, 583.5446383100001, 559.15693639, 283.8424569, 209.16254078999998,
                          560.9230889999999, 494.04984521000006, 1054.0497854, 926.3171158999999, 810.1846358199999,
                          893.53204251, 703.29253101, 939.22336472]
    x_interpolated, rmse = inter.recover_series(x_original, x_nan, indices, interpolate_value)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated.txt', x_interpolated)
    print("x_interpolated = ", x_interpolated)
    print("rmse = " + str(rmse))
# x_interpolated =  [ 655  640  625  620  605  585  570  555  540  535  530  525  520  525
#   525  522  545  550  555  545  540  540  535  520  515  505  495  585
#   575  465  455  445  433  425  415  400  390  380  369  360  355  345
#   340  335  330  320  320  315  310  305  305  305  300  295  295  295
#   290  290  285  285  280  280  280  275  269  275  270  270  270  270
#   270  270  270  270  265  265  270  270  270  275  280  285  295  315
#   345  380  405  440  460  520  570  625  670  700  781  857  919  976
#  1090 1370 1660 2260 2940 3960 4970 5460 5560 5380 5130 4930 4610 4510
#  4650 4750 4630 4470 4306 4090 3870 3680 3480 3290 3210 3170 3180 3120
#  3040 2930 2860 2870 2800 2700 2620 2550 2440 2400 2400 2420 2470 2480
#  2420 2300 2170 2060 1940 1820 1706 1640 1560 1480 1410 1350 1280 1220
#  1170 1120 1050 1010  967  937  889  850  802  750  713  682  651  658
#   645  622  586  568  572  613  599  583  561  565  582  540  526  507
#   491  494  476  450  433  439  439  447  453  450  457  500  543  545
#   543  532  522  511  497  530  539  533  548  576  581  571  559  567
#   560  573  579  560  556  561  585  585  580  576  565  546  485  482
#   473  476  463  503  511  501  491  485  467  448  430  407  386  374
#   364  347  330  314  301  295  289  306  295  283  278  273  254  241
#   233  228  218  203  195  204  208  201  206  209  196  198  272  362
#   419  437  449  480  525  560  519  523  509  494  516  620  759  801
#   842  876  890  878  888 1000 1050 1060 1060 1060 1054 1050 1010  982
#   955  926  877  854  833  810  914  921  909  893  899  884  856  845
#   840  821  795  781  781  781  785  771  776  791  793  780  751  743
#   735  719  711  709  703  670  664  766  886  883  909  966  952  948
#   939  915  900  910  950 1020 1090 1160 1210 1210 1190 1150 1100 1050
#   990  950  925  900  870  850  820  780  760  740  700  670  650  630]

def test_split_series_via_nan_mul_step():
    inter = Interpolation()
    x = camelsus_streamflow_01013500_80_035nan
    p = 2
    subseries, indices_group = inter.split_series_via_nan_mul_step(x, p)
    print("subseries = ", subseries)
    print("indices_group = ", indices_group)
# subseries =  [[585, 570, -100], [-100, 540, -100], [525, 525, -100], [540, 535, -100, -100], [505, 495, -100],
#               [575, 465, -100, -100], [-100, 440, -100, -100], [345, 340, -100], [-100, 330, -100], [-100, 320, -100],
#               [-100, 310, -100], [295, 295, -100], [280, 280, -100], [-100, 275, -100, -100, -100],
#               [270, 270, -100, -100, -100, -100, -100, -100], [-100, 270, -100], [285, 295, -100], [380, 405, -100],
#               [460, 520, -100, -100, -100], [919, 976, -100], [-100, 1370, -100], [-100, 2260, -100],
#               [3960, 4970, -100], [4510, 4650, -100], [4470, 4290, -100], [3170, 3180, -100], [-100, 3040, -100, -100],
#               [2870, 2800, -100], [-100, 2620, -100], [2470, 2480, -100], [1940, 1820, -100], [1010, 967, -100],
#               [-100, 889, -100, -100], [581, 561, -100, -100], [-100, 540, -100, -100, -100], [-100, 494, -100],
#               [-100, 450, -100, -100, -100], [-100, 447, -100], [543, 532, -100], [-100, 511, -100],
#               [530, 539, -100, -100], [579, 560, -100], [585, 580, -100], [565, 546, -100], [476, 463, -100],
#               [-100, 511, -100, -100], [467, 448, -100, -100], [-100, 386, -100], [364, 347, -100, -100, -100, -100],
#               [241, 233, -100, -100], [206, 208, -100, -100, -100], [-100, 362, -100], [525, 523, -100],
#               [523, 509, -100, -100], [759, 801, -100], [1000, 1050, -100], [1050, 1050, -100], [877, 854, -100, -100],
#               [-100, 914, -100], [884, 856, -100], [776, 791, -100], [735, 719, -100], [-100, 709, -100],
#               [670, 664, -100, -100], [-100, 883, -100], [915, 900, -100], [1160, 1210, -100], [-100, 1190, -100],
#               [1100, 1050, -100], [900, 870, -100], [780, 760, -100, -100]]
# indices_group =  [[7], [9], [15], [23, 24], [27], [30, 31], [33, 34], [43], [45], [47], [49], [56], [63], [65, 66, 67],
#                   [72, 73, 74, 75, 76, 77], [79], [83], [87], [90, 91, 92], [98], [100], [102], [105], [113], [117],
#                   [125], [127, 128], [131], [133], [140], [146], [159], [161, 162], [177, 178], [180, 181, 182], [184],
#                   [186, 187, 188], [190], [198], [200], [203, 204], [214], [219], [222], [227], [229, 230], [234, 235],
#                   [237], [240, 241, 242, 243], [253, 254], [262, 263, 264], [266], [272], [275, 276], [280], [287],
#                   [292], [298, 299], [301], [307], [318], [324], [326], [329, 330], [332], [339], [345], [347], [350],
#                   [355], [359, 360]]

def test_interpolate_ar_single_step():
    inter = Interpolation()
    x = [525, 525]
    x = [[525, 525, -100],
        [455, 445, -100],
        [390, 380, -100],
        [280, 275, -100],
        [4630, 4470, -100],
        [1940, 1820, -100],
        [613, 599, -100],
        [581, 571, -100],
        [306, 295, -100],
        [201, 206, -100],
        [480, 525, -100],
        [523, 509, -100],
        [1060, 1060, -100],
        [982, 955, -100],
        [854, 833, -100],
        [921, 909, -100],
        [711, 709, -100],
        [952, 948, -100]]
    phi = [1.85816724, -0.86378065]
    p = 2
    l = 1
    x_subseries, interpolate_value = inter.interpolate_ar_single_step(x, phi, p, l)
    print("x_subseries = ", x_subseries)
    print("interpolate_value = ", interpolate_value)
# x_infer = [522.0529597499999]
# x_subseries =  [[525, 525, 522.0529597499999], [455, 445, 433.86422605], [390, 380, 369.22909769999995],
#                 [280, 275, 269.137409], [4630, 4470, 4306.703153299999], [1940, 1820, 1706.1299158000002],
#                 [613, 599, 583.5446383100001], [581, 571, 559.15693639], [306, 295, 283.8424569],
#                 [201, 206, 209.16254078999998], [480, 525, 560.9230889999999], [523, 509, 494.04984521000006],
#                 [1060, 1060, 1054.0497854], [982, 955, 926.3171158999999], [854, 833, 810.1846358199999],
#                 [921, 909, 893.53204251], [711, 709, 703.29253101], [952, 948, 939.22336472]]
# interpolate_value =  [522.0529597499999, 433.86422605, 369.22909769999995, 269.137409, 4306.703153299999,
#                       1706.1299158000002, 583.5446383100001, 559.15693639, 283.8424569, 209.16254078999998,
#                       560.9230889999999, 494.04984521000006, 1054.0497854, 926.3171158999999, 810.1846358199999,
#                       893.53204251, 703.29253101, 939.22336472]

def test_interpolate_ar_mul_step():
    inter = Interpolation()
    subseries =  [[585, 570, -100], [525, 525, -100], [540, 535, -100, -100], [505, 495, -100],
                  [575, 465, -100, -100]]
    phi = [1.85816724, -0.86378065]
    p = 2
    l = 1
    x_subseries, interpolate_value = inter.interpolate_ar_mul_step(subseries, phi, p)
    print("x_subseries = ", x_subseries)
    print("interpolate_value = ", interpolate_value)
# x_subseries =  [[585, 570, 553.84364655], [525, 525, 522.0529597499999], [540, 535, 527.6779223999999, 518.3911809249421],
#                 [505, 495, 483.58355555], [575, 465, 367.37389285, 280.9841302751402]]
# interpolate_value =  [[553.84364655], [522.0529597499999], [527.6779223999999, 518.3911809249421], [483.58355555],
#                       [367.37389285, 280.9841302751402]]

def test_interpolate_ar_reverse_single_step():
    inter = Interpolation()
    phi = [1.85816724, -0.86378065]
    subseries = [[-100, 530, 525], [-100, 505, 495]]
    subseries = [[-100, 527.68, 522.71], [-100, 507.79, 493.53]]
    subseries = [[-100, 605, 585], [-100, 555, 540]]
    subseries = [[-100, 545, 550], [-100, 425, 415]]
    p = 2
    l = 1
    x_subseries, interpolate_value = inter.interpolate_ar_reverse_single_step(subseries, phi, p)
    print("x_subseries = ", x_subseries)
    print("interpolate_value = ", interpolate_value)
#     subseries = [[-100, 530, 525], [-100, 505, 495]]   ratio=0.35
# x_subseries =  [[532.3442209547064, 530, 525], [513.2951938666374, 505, 495]]
# interpolate_value =  [532.3442209547064, 513.2951938666374]
#     subseries = [[-100, 527.68, 522.71], [-100, 507.79, 493.53]]   y_arima
# x_subseries =  [[530.0045667881075, 527.68, 522.71], [520.9988702567024, 507.79, 493.53]]
# interpolate_value =  [530.0045667881075, 520.9988702567024]
#     subseries = [[-100, 605, 585], [-100, 555, 540]]  ratio=0.1
# x_subseries =  [[624.222341864222, 605, 585], [568.7587678654297, 555, 540]]
# interpolate_value =  [624.222341864222, 568.7587678654297]
#     subseries = [[-100, 545, 550], [-100, 425, 415]]  ratio=0.05
# x_subseries =  [[535.669727956976, 545, 550], [433.81508604065164, 425, 415]]
# interpolate_value =  [535.669727956976, 433.81508604065164]

def test_interpolate_ar_series_forward():
    inter = Interpolation()
    x_nan = camelsus_streamflow_01013500_80_005nan
    phi = [1.85816724, -0.86378065]
    p = 2
    x_infer_forward = inter.interpolate_ar_series_forward(x_nan, phi, p)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_infer_forward.txt', x_infer_forward)
    print("x_infer_forward = ", x_infer_forward)
# x_infer_forward =  [655, 640, 625, 620, 605, 585, 570, 555, 540, 535, 530, 525, 520, 525, 525, 522.0529597499999,
#                     545, 550, 555, 545, 540, 540, 535, 520, 515, 505, 495, 585, 575, 465, 455, 445, 433.86422605,
#                     425, 415, 400, 390, 380, 369.22909769999995, 360, 355, 345, 340, 335, 330, 320, 320, 315, 310,
#                     305, 305, 305, 300, 295, 295, 295, 290, 290, 285, 285, 280, 280, 280, 275, 269.137409, 275, 270,
#                     270, 270, 270, 270, 270, 270, 270, 265, 265, 270, 270, 270, 275, 280, 285, 295, 315, 345, 380,
#                     405, 440, 460, 520, 570, 625, 670, 700, 781, 857, 919, 976, 1090, 1370, 1660, 2260, 2940, 3960,
#                     4970, 5460, 5560, 5380, 5130, 4930, 4610, 4510, 4650, 4750, 4630, 4470, 4306.703153299999, 4090,
#                     3870, 3680, 3480, 3290, 3210, 3170, 3180, 3120, 3040, 2930, 2860, 2870, 2800, 2700, 2620, 2550,
#                     2440, 2400, 2400, 2420, 2470, 2480, 2420, 2300, 2170, 2060, 1940, 1820, 1706.1299158000002, 1640,
#                     1560, 1480, 1410, 1350, 1280, 1220, 1170, 1120, 1050, 1010, 967, 937, 889, 850, 802, 750, 713,
#                     682, 651, 658, 645, 622, 586, 568, 572, 613, 599, 583.5446383100001, 561, 565, 582, 540, 526,
#                     507, 491, 494, 476, 450, 433, 439, 439, 447, 453, 450, 457, 500, 543, 545, 543, 532, 522, 511,
#                     497, 530, 539, 533, 548, 576, 581, 571, 559.15693639, 567, 560, 573, 579, 560, 556, 561, 585,
#                     585, 580, 576, 565, 546, 485, 482, 473, 476, 463, 503, 511, 501, 491, 485, 467, 448, 430, 407,
#                     386, 374, 364, 347, 330, 314, 301, 295, 289, 306, 295, 283.8424569, 278, 273, 254, 241, 233, 228,
#                     218, 203, 195, 204, 208, 201, 206, 209.16254078999998, 196, 198, 272, 362, 419, 437, 449, 480,
#                     525, 560.9230889999999, 519, 523, 509, 494.04984521000006, 516, 620, 759, 801, 842, 876, 890, 878,
#                     888, 1000, 1050, 1060, 1060, 1060, 1054.0497854, 1050, 1010, 982, 955, 926.3171158999999, 877,
#                     854, 833, 810.1846358199999, 914, 921, 909, 893.53204251, 899, 884, 856, 845, 840, 821, 795, 781,
#                     781, 781, 785, 771, 776, 791, 793, 780, 751, 743, 735, 719, 711, 709, 703.29253101, 670, 664,
#                     766, 886, 883, 909, 966, 952, 948, 939.22336472, 915, 900, 910, 950, 1020, 1090, 1160, 1210,
#                     1210, 1190, 1150, 1100, 1050, 990, 950, 925, 900, 870, 850, 820, 780, 760, 740, 700, 670, 650,
#                     630]

def test_interpolate_ar_series_backward():
    inter = Interpolation()
    x_nan = camelsus_streamflow_01013500_80_005nan
    phi = [1.85816724, -0.86378065]
    p = 2
    x_infer_backward = inter.interpolate_ar_series_backward(x_nan, phi, p)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_infer_backward.txt', x_infer_backward)
    print("x_infer_backward = ", x_infer_backward)
# x_infer_backward =  [655, 640, 625, 620, 605, 585, 570, 555, 540, 535, 530, 525, 520, 525, 525, 535.669727956976,
#                      545, 550, 555, 545, 540, 540, 535, 520, 515, 505, 495, 585, 575, 465, 455, 445,
#                      433.81508604065164, 425, 415, 400, 390, 380, 363.4489918244869, 360, 355, 345, 340, 335, 330,
#                      320, 320, 315, 310, 305, 305, 305, 300, 295, 295, 295, 290, 290, 285, 285, 280, 280, 280, 275,
#                      279.00137725937714, 275, 270, 270, 270, 270, 270, 270, 270, 270, 265, 265, 270, 270, 270, 275,
#                      280, 285, 295, 315, 345, 380, 405, 440, 460, 520, 570, 625, 670, 700, 781, 857, 919, 976, 1090,
#                      1370, 1660, 2260, 2940, 3960, 4970, 5460, 5560, 5380, 5130, 4930, 4610, 4510, 4650, 4750, 4630,
#                      4470, 4318.114803335777, 4090, 3870, 3680, 3480, 3290, 3210, 3170, 3180, 3120, 3040, 2930, 2860,
#                      2870, 2800, 2700, 2620, 2550, 2440, 2400, 2400, 2420, 2470, 2480, 2420, 2300, 2170, 2060, 1940,
#                      1820, 1721.958316153528, 1640, 1560, 1480, 1410, 1350, 1280, 1220, 1170, 1120, 1050, 1010, 967,
#                      937, 889, 850, 802, 750, 713, 682, 651, 658, 645, 622, 586, 568, 572, 613, 599, 552.7234508436834,
#                      561, 565, 582, 540, 526, 507, 491, 494, 476, 450, 433, 439, 439, 447, 453, 450, 457, 500, 543,
#                      545, 543, 532, 522, 511, 497, 530, 539, 533, 548, 576, 581, 571, 571.4191734672455, 567, 560,
#                      573, 579, 560, 556, 561, 585, 585, 580, 576, 565, 546, 485, 482, 473, 476, 463, 503, 511, 501,
#                      491, 485, 467, 448, 430, 407, 386, 374, 364, 347, 330, 314, 301, 295, 289, 306, 295,
#                      281.98188130285155, 278, 273, 254, 241, 233, 228, 218, 203, 195, 204, 208, 201, 206,
#                      192.41086153064435, 196, 198, 272, 362, 419, 437, 449, 480, 525, 510.99639423504095, 519, 523,
#                      509, 392.24575804053956, 516, 620, 759, 801, 842, 876, 890, 878, 888, 1000, 1050, 1060, 1060,
#                      1060, 1089.4844680764727, 1050, 1010, 982, 955, 897.9278124371042, 877, 854, 833,
#                      899.9563226613146, 914, 921, 909, 910.523231517168, 899, 884, 856, 845, 840, 821, 795, 781,
#                      781, 781, 785, 771, 776, 791, 793, 780, 751, 743, 735, 719, 711, 709, 672.5921109716917, 670,
#                      664, 766, 886, 883, 909, 966, 952, 948, 926.4192530823651, 915, 900, 910, 950, 1020, 1090, 1160,
#                      1210, 1210, 1190, 1150, 1100, 1050, 990, 950, 925, 900, 870, 850, 820, 780, 760, 740, 700, 670,
#                      650, 630]

def test_interpolate_ar_series():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    x_nan = camelsus_streamflow_01013500_80_005nan
    x_nan = camelsus_streamflow_01013500_80_01nan
    x_nan = camelsus_streamflow_01013500_80_015nan
    x_nan = camelsus_streamflow_01013500_80_025nan
    # x_nan = camelsus_streamflow_01013500_80_030nan
    # x_nan = camelsus_streamflow_01013500_80_035nan
    phi = [1.85816724, -0.86378065]
    p = 2
    x_infer_forward, x_infer_backward, x_infer, rmse_forward, rmse_backward, rmse_infer = inter.interpolate_ar_series(x_nan, phi, p, x_original)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_infer_forward.txt', x_infer_forward)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_infer_backward.txt', x_infer_backward)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_infer.txt', x_infer)
    print("rmse_forward = ", rmse_forward)
    print("rmse_backward = ", rmse_backward)
    print("rmse_infer = ", rmse_infer)
# x_nan = camelsus_streamflow_01013500_80_005nan
# rmse_forward =  (6.950849421034852, 115.81536418000007)
# rmse_backward =  (7.661657226901282, 125.75424195946044)
# rmse_infer =  (5.944652302236895, 74.85219837473016)     # (rmse, max_abs_error)
# x_nan = camelsus_streamflow_01013500_80_01nan
# rmse_forward =  (26.888780035509026, 364.2955839458182)
# rmse_backward =  (15.334708419179199, 234.18865437654767)
# rmse_infer =  (16.86079176313648, 193.97831520817726)
# x_nan = camelsus_streamflow_01013500_80_015nan
# rmse_forward =  (24.827410637549402, 358.82187209999984)
# rmse_backward =  (32.79717114852104, 519.9613853355011)
# rmse_infer =  (24.277901768557253, 439.39162871775034)
# x_nan = camelsus_streamflow_01013500_80_030nan
# rmse_forward =  (36.04573933602913, 381.681654473643)
# rmse_backward =  (139.61103502362778, 1544.873026262635)
# rmse_infer =  (64.14869901319953, 736.9065903897481)
# x_nan = camelsus_streamflow_01013500_80_035nan
# rmse_forward =  (37.59175068867898, 354.51980880000065)
# rmse_backward =  (44.65296791860337, 454.0456309768416)
# rmse_infer =  (34.15959675201914, 313.38677028892744)
# x_nan = camelsus_streamflow_01013500_80_025nan  add 625,620.
# rmse_forward =  (29.611969943773968, 316.3039648909446)
# rmse_backward =  (93.98032008006976, 1028.063477806188)
# rmse_infer =  (53.297214328820075, 513.7514154491387)
# x_nan = camelsus_streamflow_01013500_80_025nan
# rmse_forward =  (29.64907053694257, 316.3039648909446)
# rmse_backward =  (93.99201658708033, 1028.063477806188)
# rmse_infer =  (53.317836368009225, 513.7514154491387)
# x_nan = camelsus_streamflow_01013500_80_025nan  add -100,-100,-100 in the end.
# rmse_forward =  (29.721721196002687, 316.3039648909446)
# rmse_backward =  (94.01495896533861, 1028.063477806188)
# rmse_infer =  (53.35827023165319, 513.7514154491387)

def test_interpolate_ar_series_residual():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    x_nan = camelsus_streamflow_01013500_80_005nan
    phi = [1.85816724, -0.86378065]
    p = 2
    residual, y_t, mean_residual, residual_center, residual_center_2 = inter.interpolate_ar_series_residual(x_nan, phi, p, x_original)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual.txt', residual)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_t.txt', y_t)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_center.txt', residual_center)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_center_2.txt', residual_center_2)
    print("mean_residual = ", mean_residual)
# mean_residual =  5.348338061245576

def test_residual_for_arch():
    inter = Interpolation()
    x_nan = camelsus_streamflow_01013500_80_005nan
    phi = [1.85816724, -0.86378065]
    p = 2
    q = 4
    index_nan = 15
    mean_residual = 5.350272800384656
    residual_center_2 = inter.residual_for_arch(x_nan, phi, p, q, index_nan, mean_residual)
    print("residual_center_2 = ", residual_center_2)
# y =  [535.         530.         522.70598945 517.7340565  512.76212355]
# residual_center_2 =  [277.69112769 278.62733785  45.1611023   47.46573921]

def test_interpolate_arch():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    x_nan = camelsus_streamflow_01013500_80_005nan
    e = e_01013500_80
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    mean_residual = 5.350272800384656
    x_interpolated, epsilon = inter.interpolat_arch(x_nan, e, theta, p, q, mean_residual, x_original)
    np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated.txt', x_interpolated)
    print("epsilon", epsilon)
# epsilon [-1.7077067256307739, -68.57097349269718, 15.491102495032449, 1.8671569165405904, -2.041285302604978,
# -8.95687602522637, 24.42886150860708, -21.34102435042942, -2.6496437636908596, 8.649453196647936, -37.56983753635273,
# -529.1618099943597, 11.339433480015623, 63.058511946802675, 39.92314597090154, 308.2649433793956,
# -5.6630605567057835, -41.968257481555256]

def test_interpolate_arch_model():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    # x_nan = camelsus_streamflow_01013500_80_005nan
    # x_nan = camelsus_streamflow_01013500_80_01nan
    # x_nan = camelsus_streamflow_01013500_80_015nan
    # x_nan = camelsus_streamflow_01013500_80_025nan
    # x_nan = camelsus_streamflow_01013500_80_030nan
    x_nan = camelsus_streamflow_01013500_80_035nan
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    mean_residual = 5.350272800384656
    nse = 0.96
    rmse = 35
    max_error = 250
    max_loop = 3000
    result = inter.interpolate_arch_model(x_nan, theta, p, q, mean_residual, nse, rmse, max_error, max_loop, x_original)
    # x_interpolated_ar, x_interpolated, epsilon, e, nse, rmse, max_abs_error
    if result["x_interpolated"] is not None:
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated_ar.txt', result["x_interpolated_ar"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated.txt', result["x_interpolated"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\epsilon.txt', result["epsilon"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\e.txt', result["e"])
        print("n_loop = " + str(result["i_loop"]))
        print("NSE = " + str(result["nse"]))
        print("RMSE = " + str(result["rmse"]))
        print("max_abs_error = " + str(result["max_abs_error"]))
# x_nan = camelsus_streamflow_01013500_80_005nan
# n_loop = 1
# NSE = 0.9999149139118393
# RMSE = 9.504858808079241
# max_abs_error = 105.
# nse = 0.96 rmse = 5 max_error = 70 max_loop = 1000
# n_loop = 587
# NSE = 0.9999767846696925
# RMSE = 4.96482362499573
# max_abs_error = 51.34466029470349
# nse = 0.96 rmse = 5 max_error = 55 max_loop = 3000
# n_loop = 2164
# NSE = 0.9999799574075087
# RMSE = 4.6131049962316295
# max_abs_error = 39.68649999866386
# x_nan = camelsus_streamflow_01013500_80_015nan
# e0 nse=0.92 rmse=25 max_error=450 max_loop=3000
# n_loop = 2
# NSE = 0.9994137689283079
# RMSE = 24.948865856031865
# max_abs_error = 268.6459812543419
# e1 nse=0.96 rmse=20 max_error=250 max_loop=3000
# n_loop = 38
# NSE = 0.9998405028491549
# RMSE = 13.013462231748022
# max_abs_error = 94.40341191604409
# e2 nse=0.96 rmse=15 max_error=100 max_loop=3000
# n_loop = 249
# NSE = 0.9998253820948303
# RMSE = 13.616352257134706
# max_abs_error = 98.60792922769133
# x_nan = camelsus_streamflow_01013500_80_035nan
# e0 nse=0.96 rmse=35 max_error=320 max_loop=3000
# n_loop = 48
# NSE = 0.9988620753798418
# RMSE = 34.75946938671989
# max_abs_error = 289.79785405632083
# e1 nse=0.96 rmse=35 max_error=320 max_loop=3000
# n_loop = 31
# NSE = 0.9991299842215099
# RMSE = 30.39344688733817
# max_abs_error = 249.16493569145513
# e2 nse=0.96 rmse=35 max_error=320 max_loop=3000
# n_loop = 429
# NSE = 0.9988970662793493
# RMSE = 34.22087411462162
# max_abs_error = 209.82292325646443
# e3 nse=0.96 rmse=35 max_error=250 max_loop=3000
# n_loop = 76
# NSE = 0.9990815744296837
# RMSE = 31.227583229873932
# max_abs_error = 211.7235038126836

def test_interpolate_arch_model_():
    inter = Interpolation()
    x_original = camelsus_streamflow_01013500_80
    x_nan = camelsus_streamflow_01013500_80_005nan
    # x_nan = camelsus_streamflow_01013500_80_01nan
    # x_nan = camelsus_streamflow_01013500_80_015nan
    # x_nan = camelsus_streamflow_01013500_80_025nan
    # x_nan = camelsus_streamflow_01013500_80_030nan
    # x_nan = camelsus_streamflow_01013500_80_035nan
    theta = [1.85816724e+00, -8.63780650e-01, 3.17744057e+02, 2.78526653e-01, 5.45285923e-01, 4.38002156e-03, 1.15738438e-02]
    p = 2
    q = 4
    mean_residual = 5.350272800384656
    nse = 0.96
    rmse = 60
    max_error = 350
    max_loop = 3000
    result_arch, result_interpolated = inter.interpolate_arch_model_(x_nan, theta, p, q, nse, rmse, max_error, max_loop, x_original)
    if result_arch["y_arch"] is not None:
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_arch.txt', result_arch["y_arch"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\y_arima.txt', result_arch["y_arima"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual.txt', result_arch["residual"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_center.txt', result_arch["residual_center"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\residual_2.txt', result_arch["residual_2"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\delta_2.txt', result_arch["delta_2"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\delta.txt', result_arch["delta"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\epsilon.txt', result_arch["epsilon"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\e.txt', result_arch["e"])
        print("n_loop_arch = " + str(result_arch["i_loop"]))
        print("mean_residual_arch = " + str(result_arch["mean_residual"]))
        print("NSE_arch = " + str(result_arch["nse"]))
        print("RMSE_arch = " + str(result_arch["rmse"]))
        print("max_abs_error_arch = " + str(result_arch["max_abs_error"]))
    # x_interpolated_ar, x_interpolated, epsilon, e, nse, rmse, max_abs_error
    if result_interpolated["x_interpolated"] is not None:
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated_ar.txt', result_interpolated["x_interpolated_ar"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\x_interpolated.txt', result_interpolated["x_interpolated"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\epsilon.txt', result_interpolated["epsilon"])
        np.savetxt(r'D:\minio\waterism\datasets-origin\camels\camels_ystl\interpolation\e.txt', result_interpolated["e"])
        print("NSE_interpolated = " + str(result_interpolated["nse"]))
        print("RMSE_interpolated = " + str(result_interpolated["rmse"]))
        print("max_abs_error_interpolated = " + str(result_interpolated["max_abs_error"]))
# x_nan = camelsus_streamflow_01013500_80_005nan
# e0 nse=0.96 rmse=100 max_error=500 max_loop=3000
# n_loop_arch = 1
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.996296688859486
# RMSE_arch = 63.076406633353905
# max_abs_error_arch = 422.1720639908772
# NSE_interpolated = 0.999644739277563
# RMSE_interpolated = 19.421822850875078
# max_abs_error_interpolated = 331.79156850791924
# e1 nse=0.96 rmse=100 max_error=500 max_loop=3000
# n_loop_arch = 1
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9970960643936727
# RMSE_arch = 55.855421615318456
# max_abs_error_arch = 450.9454050488116
# NSE_interpolated = 0.9998600302606823
# RMSE_interpolated = 12.190835417656928
# max_abs_error_interpolated = 200.1238261295896
# e2 nse=0.96 rmse=60 max_error=350 max_loop=3000
# n_loop_arch = 17
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9974686725454015
# RMSE_arch = 52.14900279827845
# max_abs_error_arch = 348.3890526906198
# NSE_interpolated = 0.9999288719351255
# RMSE_interpolated = 8.690342397886111
# max_abs_error_interpolated = 91.91660625932008
# e3 nse=0.96 rmse=50 max_error=330 max_loop=3000
# n_loop_arch = 479
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9978839848513783
# RMSE_arch = 47.67944919718735
# max_abs_error_arch = 275.6895733676529
# NSE_interpolated = 0.9999084561152658
# RMSE_interpolated = 9.858958994221648
# max_abs_error_interpolated = 92.11234757622378
# e4 nse=0.96 rmse=50 max_error=330 max_loop=3000
# n_loop_arch = 285
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9979446376171421
# RMSE_arch = 46.991146954777115
# max_abs_error_arch = 277.5432383355171
# NSE_interpolated = 0.9997925255851353
# RMSE_interpolated = 14.842213191671744
# max_abs_error_interpolated = 130.729278299322
# e5 nse=0.96 rmse=50 max_error=280 max_loop=3000
# n_loop_arch = 2880
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9979066194092374
# RMSE_arch = 47.42375518837534
# max_abs_error_arch = 267.58685929874355
# NSE_interpolated = 0.9998986075767466
# RMSE_interpolated = 10.375741484183191
# max_abs_error_interpolated = 140.740161300183
# e6 nse=0.96 rmse=50 max_error=270 max_loop=3000
# n_loop_arch = 446
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.9980241388286094
# RMSE_arch = 46.07337820084073
# max_abs_error_arch = 265.0156330479376
# NSE_interpolated = 0.9998771039560382
# RMSE_interpolated = 11.423135879138458
# max_abs_error_interpolated = 123.79255222635402
# e7 nse=0.96 rmse=46 max_error=265 max_loop=3000
# n_loop_arch = 104
# mean_residual_arch = 5.348338061245576
# NSE_arch = 0.997337437707914
# RMSE_arch = 53.48373545614522
# max_abs_error_arch = 325.628756458942
# NSE_interpolated = 0.9999410921731966
# RMSE_interpolated = 7.90865975621777
# max_abs_error_interpolated = 81.13278812993985
