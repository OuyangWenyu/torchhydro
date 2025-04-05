"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-09-10 10:44:23
LastEditors: Wenyu Ouyang
Description: some basic config for hydrological models
FilePath: \torchhydro\torchhydro\configs\model_config.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from collections import OrderedDict

# NOTE: Don't change the parameter settings

MODEL_PARAM_DICT = {
    "xaj": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "CS",  # The recession constant of channel system
            "L",  # Lag time
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "CS": [0.0, 1.0],
                "L": [1.0, 10.0],  # unit is day
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
            }
        ),
    },
    "xaj_mz": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "A",  # parameter of mizuRoute
            "THETA",  # parameter of mizuRoute
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1.0, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
            }
        ),
    },
    "gr4j": {
        "param_name": ["x1", "x2", "x3", "x4"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
            }
        ),
    },
    "hymod": {
        "param_name": ["cmax", "bexp", "alpha", "ks", "kq"],
        "param_range": OrderedDict(
            {
                "cmax": [1.0, 500.0],
                "bexp": [0.1, 2.0],
                "alpha": [0.1, 0.99],
                "ks": [0.001, 0.10],
                "kq": [0.1, 0.99],
            }
        ),
    },
    "sac": {  # 21
        "param_name": [
            "KC",  # coefficient of potential evapotranspiration to reference crop evaporation generally  K
            "PCTIM",  # ratio of the permanent impervious area to total area of the basin 永久不透水面积占比
            "ADIMP",  # ratio of the alterable impervious area to total area of the basin 可变不透水面积占比
            "UZTWM",  # tension water capacity in the upper layer 上土层张力水容量  zone  M
            "UZFWM",  # free water capacity in the upper layer 上土层自由水容量
            "LZTWM",  # tension water capacity in the lower layer 下土层张力水容量
            "LZFSM",  # speedy free water capacity in the lower layer 下土层快速自由水容量
            "LZFPM",  # slow free water capacity in the lower layer 下土层慢速自由水容量
            "RSERV",  # ratio of the part which do not evaporate in lower layer free water 下土层自由水中不参与蒸散发的比例
            "PFREE",  # ratio of supplying the free water which infiltrating from upper layer to lower layer 从上土层向下土层下渗的水量中补充自由水的比例
            "RIVA",  # ratio of river net, lakes and hydrophyte area to total area of the basin 河网、湖泊及水生植物面积占全流域的面积比例
            "ZPERC",  # parameter about the maximal infiltrating ratio与最大下渗率有关的参数
            "REXP",  # exponent of the infiltrating curve 下渗曲线指数
            "UZK",  # daily outflow coefficient of the upper layer free water 上土层自由水日出流系数
            "LZSK",  # daily outflow coefficient of the lower layer speedy free water 下土层快速自由水日出流系数
            "LZPK",  # daily outflow coefficient of the lower layer slow free water 下土层慢速自由水日出流系数
            "CI",  # recession coefficient of the interflow 壤中流日消退系数
            "CGS",  # recession coefficient of speedy groundwater 快速地下水消退系数
            "CGP",  # recession coefficient of slow groundwater 慢速地下水消退系数
            "KE",  # confluence duration in riverway (hourly) 河道汇流时间(小时)
            "XE",  # flow weight coefficient of riverway confluence 河道汇流流量比重系数
        ],
        "param_range": OrderedDict(
            {
                "KC": [0.1, 1.0],
                "PCTIM": [0.0, 0.6],
                "ADIMP": [0.0, 0.3],
                "UZTWM": [10, 200],
                "UZFWM": [10, 100],
                "LZTWM": [10, 160],
                "LZFSM": [10, 50],
                "LZFPM": [10, 150],
                "RSERV": [0.1, 0.4],
                "PFREE": [0.01, 0.5],
                "RIVA": [0.01, 0.09],
                "ZPERC": [2, 25],
                "REXP": [1, 5],
                "UZK": [0.1, 1.0],
                "LZSK": [0.05, 1.0],
                "LZPK": [0.002, 0.7],
                "CI": [0.25, 1.0],
                "CGS": [0.5, 1.0],
                "CGP": [0.70, 1.0],
                "KE": [0.0, 1000],
                "XE": [-0.5, 0.5],
            }
        ),
    },
    "tank": {  # 20
        "param_name": [
            "KC",  # coefficient of potential evapotranspiration to reference crop evaporation generally 蒸散发折算系数
            "W1",  # the first soil moisture layer saturation capacity 第一土壤水分层饱和容量
            "W2",  # the second soil moisture layer saturation capacity 第二土壤水分层饱和容量
            "K1",  # ratio of lower layer free water supplied upper layer tension water 下层自由水补充上层张力水比例系数
            "K2",  # water exchange coefficient of the first layer with the second layer 第一层与第二层的水量交换系数
            "a0",  # coefficient of the first layer free water infiltrated into the second layer 第一层自由水下渗到第二层的下渗系数
            "b0",  # coefficient of the second layer free water infiltrated into the interflow 第二层自由水下渗到壤中流的下渗系数
            "c0",  # coefficient of the interflow infiltrated into the shoal layer groundwater 壤中流下渗到浅层地下水的下渗系数
            "h1",  # hole height of surface runoff discharging via bottom hole 地面径流底孔出流孔高度
            "h2",  # hole height of surface runoff discharging via surface hole 地面径流表孔出流孔高度
            "a1",  # coefficient of surface runoff discharging via bottom hole 地面径流底孔出流系数
            "a2",  # coefficient of surface runoff discharging via surface hole 地面径流表孔出流系数
            "h3",  # hole height of interflow discharging via hole 壤中流出流孔高度
            "b1",  # coefficient of interflow discharging 壤中流出流系数
            "h4",  # hole height of shoal layer groundwater discharging 浅层地下水出流孔高度
            "c1",  # coefficient of shoal layer groundwater discharging 浅层地下水出流系数
            "d1",  # coefficient of deep layer groundwater discharging  深层地下水出流系数
            "e1",  # coefficient of bottom hole discharging 底孔出流系数
            "e2",  # coefficient of surface hole discharging 表孔出流系数
            "h",  # hole height of river-tank discharging 河道水箱出流孔高度

        ],
        "param_range": OrderedDict(
            {
                "KC": [0.20, 1.80],
                "W1": [1.0, 80.0],
                "W2": [10.0, 300.0],
                "K1": [0.1, 0.99],
                "K2": [0.1, 0.9],
                "a0": [0.05, 0.5],
                "b0": [0.01, 0.5],
                "c0": [0.05, 0.5],
                "h1": [0.0, 15.0],
                "h2": [10.0, 25.0],
                "a1": [0.01, 0.7],
                "a2": [0.01, 0.55],
                "h3": [1.0, 15.0],
                "b1": [0.1, 0.5],
                "h4": [1.0, 30.0],
                "c1": [0.1, 0.5],
                "d1": [0.05, 0.5],
                "e1": [0.01, 0.5],
                "e2": [0.1, 0.99],
                "h": [5.0, 90.0],
            }
        )
    }
}


MODEL_PARAM_TEST_WAY = {
    # 0. "train_final" -- use the final training period's parameter for each test period
    "final_train_period": "train_final",
    # 1. "final" -- use the final testing period's parameter for each test period
    "final_period": "final",
    # 2. "mean_time" -- Mean values of all training periods' parameters are used
    "mean_all_period": "mean_time",
    # 3. "mean_basin" -- Mean values of all basins' final training periods' parameters is used
    "mean_all_basin": "mean_basin",
    # 4. "var" -- use time series parameters and constant parameters in testing period
    "time_varying": "var",
    "time_scroll": "dynamic",
}
