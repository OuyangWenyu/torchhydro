import logging
import pandas as pd
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv("data/basin_id(46+1).csv", dtype={"id": str})
gage_id = show["id"].values.tolist()


def main():
    # 创建测试配置
    config_data = create_config()

    # 运行测试函数
    test_seq2seq(config_data)


def create_config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = "train_with_era5land/ex1"
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land",
                "target": "basins-origin/hour_data/1h/mean_data/streamflow_basin",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Seq2Seq",
        model_hyperparam={
            "input_size": 19,  # dual比single少2
            "output_size": 1,
            "hidden_size": 256,
            "cnn_size": 120,
            "forecast_length": 24,
            "model_mode": "dual",
            "prec_window": 1,  # 将前序径流一起作为输出，选择的时段数，该值需小于等于rho，建议置为1
        },
        model_loader={"load_way": "best"},
        # gage_id=[
        #     # "21401550",#碧流河
        #     "01181000",
        #     # "01411300",  # 2020年缺失
        #     "01414500",
        #     # "02016000",
        #     # "02018000",
        #     # "02481510",
        #     # "03070500",
        #     # "08324000",#-3000
        #     # "11266500",
        #     # "11523200",
        #     # "12020000",
        #     # "12167000",
        #     # "14185000",
        #     # "14306500",
        # ],
        gage_id=gage_id,
        batch_size=1024,
        rho=336,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
            "sm_rootzone",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow"],
        dataset="ERA5LandDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=50,
        save_epoch=1,
        train_period=[
            ("2015-06-01", "2015-09-30"),
            ("2016-06-01", "2016-09-30"),
            ("2017-06-01", "2017-09-30"),
            ("2018-06-01", "2018-09-30"),
            ("2019-06-01", "2019-09-30"),
            ("2020-06-01", "2020-09-30"),
            ("2021-06-01", "2021-09-30"),
            ("2022-06-01", "2022-09-30"),
        ],
        test_period=[
            ("2023-06-01", "2023-09-30"),
        ],
        valid_period=[
            ("2023-06-01", "2023-09-30"),  # 目前只支持一个时段
        ],
        loss_func="RMSESum",
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="batch",
        rolling=False,
        static=False,
        early_stopping=True,
        patience=10,
        ensemble=True,
        ensemble_items={
            "kfold": 9,
            "batch_sizes": [1024],
        },
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


def test_seq2seq(config_data):
    # 运行测试
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
