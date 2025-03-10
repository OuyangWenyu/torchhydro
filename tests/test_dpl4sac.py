import os
import pytest
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture()    # 用于测试的特征  装饰器是装饰器，魔法函数是魔法函数。
def dpl4sac_args():  # todo:
    project_name = os.path.join("test", "expdpl4sac")  #
    train_period = ["2010-01-01", "2011-01-01"]  # 训练期
    valid_period = ["2011-01-01", "2012-01-01"]  # 验证期
    test_period = ["2012-01-01", "2013-01-01"]  # 测试期
    return cmd(  # parameters inputted from control console
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",  # time_range: ["1980-01-01", "2014-12-31]
            "source_path": "D:\minio\waterism\datasets-origin\camels\camels_us",
            # "source_path": SETTING["local_data_path"]["datasets-interim"],
            # "other_settings": {"time_unit": ["1D"]},
        },
        model_type="Normal",  # help="The type of DL model",  "Normal": DeepHydro,   # todo: this model used DeepHydro?
        ctx=[-1],  # help="Running Context -- gpu num or cpu. E.g `--ctx 0 1` means run code in gpu 0 and 1; -1 means cpu",
        model_name="DplAnnSac",
        model_hyperparam={  # the __init__ function parameters of model class
            # "n_input_features": 2,
            # "n_input_features": 9,
            "n_input_features": 9,
            # "n_output_features": 10,
            "n_output_features": 2,  # streamflow, evaporation
            "n_hidden_states": 30,
            "warmup_length": 365,
            "param_limit_func": "clamp",
            "param_test_way": "final",
        },
        loss_func="NSELoss",  # loss function todo: detail
        dataset="DplDataset",  # help="Choose a dataset class for PyTorch", "DplDataset": DplDataset,  todo: there are "source_name": "camels_us"
        scaler="DapengScaler",  # only numerical scaler: for categorical vars, they are transformed to numerical vars when reading them
        scaler_params={  # todo:
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [  # todo"
                "total_precipitation_hourly",
                "potential_evaporation_hourly",
            ],
            "pbm_norm": True,
        },
        # 数据预处理，标准化、归一化  归一化：将一列数据变换到某个固定区间（范围）中，通常是[0，1]区间。  标准化：将数据变换为均值为0，标准差为1的分布。 所以到底是归一化还是标准化？
        gage_id=[
            # "camels_01013500",
            # "camels_01022500",
            # "camels_01030500",
            # "camels_01031500",
            # "camels_01047000",
            # "camels_01052500",
            # "camels_01054200",
            # "camels_01055000",
            # "camels_01057000",
            # "camels_01170100",
            # # "camels_01073000",
            "1013500",
            "1022500",
            "1030500",
            "1031500",
            "1047000",
            "1052500",
            "1054200",
            "1055000",
            "1057000",
            "1073000",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=10,  # todo: means basin in this model?
        # hindcast_length=10,  # 回算、反演、反算，hindcast = reforecast, 回报、回算。
        forecast_length=365,
        var_t=[  # 7
            "prcp",  # (mm/day)   todo: pet
            "dayl",  # dayl(s)
            "srad",  # (W/m2)
            "swe",  # (mm)
            "tmax",  # (C)
            "tmin",  # (C)
            "vp",  # (Pa)
        ],
        var_c=[  # todo: where to use this data items?
            "elev_mean",
            "slope_mean",
            "area_gages2",    # area  todo: how to deliver into pb model?
            "frac_forest",
            "lai_max",
            "lai_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "geol_1st_class",
            "geol_2nd_class",
            "geol_porostiy",
            "geol_permeability",
        ],
        var_out=["streamflow", "evaporation"],  # output variables, output two features, streamflow and evaporation
        target_as_input=0,  # help="if true, we will use target data as input for data assimilation or physics-based models",   todo: what's mean of target data? 目标数据？目标值，NSE?
        constant_only=0,  # help="if true, we will only use attribute data as input for deep learning models. now it is only for dpl models and it is only used when target_as_input is False. "
        train_epoch=30,  # 训练30个来回，反复迭代训练30遍
        save_epoch=1,  # 保存最后一次训练的模型参数和结果
        model_loader={  # 模型加载，加载最优模型
            "load_way": "best",
            # "test_epoch": 1,
        },
        warmup_length=365,  # 预热期一年
        opt="Adadelta",  # 通过动态调整学习率，能有效避免过度衰减的问题，适合长时间训练任务，且无需手动调整学习率。  adaptive delta
        which_first_tensor="sequence",  #序列优先，即时间是第一维
    )

def test_dpl4sac(dpl4sac_args):
    cfg = default_config_file()
    update_cfg(cfg, dpl4sac_args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
