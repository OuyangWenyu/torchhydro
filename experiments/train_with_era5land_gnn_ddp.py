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

import dgl

from experiments.fabric_launcher_config import create_config_fabric
from torchhydro.datasets.create_graph import get_upstream_graph
from torchhydro.configs.config import cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

prechn_gage_id = [gage_id.split('/')[-1].split('.')[0] for gage_id in glob.glob('/ftproot/basins-interim/timeseries/1h/*.csv', recursive=True)]
camels_hourly_usgs = [file.split('/')[-1].split('-')[0] for file in glob.glob('/ftproot/camels_hourly/data/usgs_streamflow_csv/*.csv', recursive=True)]
pre_gage_ids = [gage_id for gage_id in prechn_gage_id if (gage_id.split('_')[-1] in camels_hourly_usgs) | ('songliao' in gage_id)]
remove_list = ['03604000']
pre_gage_ids = [gage_id for gage_id in pre_gage_ids if gage_id.split('_')[1] not in remove_list]

def test_run_model():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['POLARS_VERBOSE'] = '1'
    config_data, graph_tuple = create_config_Seq2Seq()
    config_data['data_cfgs']['graph'] = graph_tuple[0]
    config_data['model_cfgs']['model_hyperparam']['graph'] = dgl.from_networkx(graph_tuple[0])
    config_data['data_cfgs']['basins_stations_df'] = graph_tuple[1]
    train_and_evaluate(config_data)


def create_config_Seq2Seq():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("train_with_era5land", "ex1_645")
    config_data = create_config_fabric()
    network_shp_path = "gnn_data/SL_USA_HydroRiver_single.shp"
    node_shp_path = "gnn_data/iowa_usgs_hml_sl_stations.shp"
    graph_tuple = get_upstream_graph(pre_gage_ids, os.path.join(os.getcwd(), 'results', project_name),
                                     network_shp_path, node_shp_path)
    train_stas_basins = (graph_tuple[1])['station_id'].to_list()
    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset_pq",
            "source": "HydroMean",
            "source_path": "/ftproot/basins-interim/",
        },
        model_name="Seq2SeqMinGNN",
        model_hyperparam={
            "en_input_size": 30,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": len(train_stas_basins),
            "forecast_length": 56,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
            "pre_norm": False, # 是否启用预归一化，True则使用DapengScaler等预归一化手段，False则不启用
            # "dropout": 0.1,
            "upstream_cut": 10
        },
        model_loader={"load_way": "best"},
        gage_id=train_stas_basins,
        batch_size=len(train_stas_basins),
        forecast_history=168,
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
        sampler="FullNeighborSampler",
        scaler="DapengScaler",
        train_epoch=0,
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
            "lr": 0.001,
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
        patience=5,
        model_type="GNN_DDP_MTL",
        continue_train=False,
        network_shp=network_shp_path,
        node_shp=node_shp_path,
        basins_shp="/ftproot/basins-interim/shapes/basins.shp",
        master_addr="localhost",
        port="12345",
        num_workers=1,
        pin_memory=True,
        # weight_path='results/train_with_era5land/ex1_643/model_Ep10.pth',
    )
    # 更新默认配置
    update_cfg(config_data, args)
    return config_data, graph_tuple

if __name__ == "__main__":
    test_run_model()
