"""
Author: Wenyu Ouyang
Date: 2024-05-20 10:40:46
LastEditTime: 2024-05-27 08:43:55
LastEditors: Wenyu Ouyang
Description:
FilePath: \torchhydro\experiments\train_with_era5land.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""
import glob
import logging
import os

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

chn_gage_id = [gage_id.split('/')[-1].split('.')[0] for gage_id in glob.glob('/ftproot/basins-interim/timeseries/3h/*.csv')
               if 'songliao' in gage_id]


def test_run_model():
    '''
    parser = argparse.ArgumentParser(
        description="Train a Seq2Seq or Transformer model."
    )
    parser.add_argument(
        "--m_name",
        type=str,
        choices=["Seq2Seq", "Transformer"],
        default="Seq2Seq",
        help="Type of model to train: Seq2Seq or Transformer",
    )

    args = parser.parse_args()

    if args.m_name == "Seq2Seq":
        config_data = create_config_Seq2Seq()
    elif args.m_name == "Transformer":
        config_data = create_config_Transformer()
    '''
    config_data = create_config_Seq2Seq()
    train_and_evaluate(config_data)


def create_config_Seq2Seq():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("train_with_era5land", "ex1_539_basins")
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": "/ftproot/basins-interim/",
        },
        ctx=[2],
        model_name="Seq2SeqGNN",
        model_hyperparam={
            "en_input_size": 24,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 56,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=chn_gage_id,
        batch_size=256,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
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
        var_out=["streamflow", "sm_surface"],
        dataset="GNNDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2016-01-01-01", "2023-11-30-01")],
        test_period=[("2015-01-01-01", "2016-01-01-01")],
        valid_period=[("2015-01-01-01", "2016-01-01-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="batch",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        model_type="MTL",
        # continue_train=True,
        network_shp='/home/wangyang1/songliao_cut_single_new.shp',
        node_shp="/home/wangyang1/463_nodes_sl/463_nodes_sl.shp"
    )

    # 更新默认配置
    update_cfg(config_data, args)
    return config_data


def create_config_Transformer():
    project_name = os.path.join("train_with_era5land_trans", "ex1")
    config_data = default_config_file()

    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Transformer",
        model_hyperparam={
            "n_encoder_inputs": 20,
            "n_decoder_inputs": 19,
            "n_decoder_output": 2,
            "channels": 256,
            "num_embeddings": 512,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "prec_window": 0,
        },
        model_loader={"load_way": "best"},
        gage_id=chn_gage_id,
        batch_size=128,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
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
        var_out=["streamflow", "sm_surface"],
        dataset="TransformerDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2015-05-01-01", "2022-12-20-01")],
        test_period=[("2023-01-01-01", "2023-11-20-01")],
        valid_period=[("2023-01-01-01", "2023-11-20-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="sequence",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        patience=10,
        model_type="GNN_MTL",
        network_shp='/home/wangyang1/songliao_cut_single.shp',
        node_shp='/home/jiaxuwu/463_nodes.shp'
    )
    update_cfg(config_data, args)
    return config_data


def test_convert_nc_zarr():
    import xarray as xr
    '''
    ncfiles = glob.glob("/ftproot/local_data_forcing_era5land_streamflow/*.nc", recursive=True)
    for ncfile in ncfiles:
        zarr_fname = ncfile.split('/')[-1].replace(".nc", ".zarr")
        zarr_path = f'/ftproot/local_data_forcing_era5land_streamflow/local_data_forcing_era5land_zarr/'
        nc_ds = xr.open_dataset(ncfile)
        nc_ds.to_zarr(f'{zarr_path}{zarr_fname}', mode="w")
        '''
    attr_nc = xr.open_dataset("/ftproot/attributes.nc")
    attr_nc.to_zarr("/ftproot/attributes.zarr", mode="w")


if __name__ == "__main__":
    test_run_model()
