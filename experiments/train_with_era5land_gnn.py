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
from itertools import chain

import dgl
import numpy as np
import geopandas as gpd
import networkx as nx

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

# from torchhydro.trainers.trainer import train_and_evaluate

# 设置日志记录器的级别为 INFO
logging.basicConfig(level=logging.INFO)

# 配置日志记录器，确保所有子记录器也记录 INFO 级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

prechn_gage_id = [gage_id.split('/')[-1].split('.')[0] for gage_id in glob.glob('/ftproot/basins-interim/timeseries/1h/*.csv', recursive=True)]
camels_hourly_usgs = [file.split('/')[-1].split('-')[0] for file in glob.glob('/ftproot/camels_hourly/data/usgs_streamflow_csv/*.csv', recursive=True)]
pre_gage_ids = [gage_id for gage_id in prechn_gage_id if (gage_id.split('_')[-1] in camels_hourly_usgs) | ('songliao' in gage_id)]
# basin_stations.csv, L100-148, L150-156
remove_list = ['02018000','02027000','02027500','02028500','02038850','02046000','02051500','02053200','02053800',
               '02055100','02056900','02059500','02064000','02065500','02069700','02070000','02074500','02077200',
               '02081500','02082950','02092500','02096846','02102908','02108000','02111180','02111500','02118500',
               '02128000','02137727','02140991','02143000','02143040','02149000','02152100','02177000','02178400',
               '02193340','02196000','02198100','02202600','02212600','02215100','02216180','02221525','02231000',
               '02245500','02246000','02296500','02297155','02298123','02298608','02299950','02300700','02349900',
               '02350900','02361000']
pre_gage_ids = [gage_id for gage_id in pre_gage_ids if gage_id.split('_')[1] not in remove_list]


def test_run_model():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config_data, graph_tuple = create_config_Seq2Seq()
    config_data['data_cfgs']['graph'] = graph_tuple[0]
    config_data['model_cfgs']['model_hyperparam']['graph'] = dgl.from_networkx(graph_tuple[0])
    config_data['data_cfgs']['basins_stations_df'] = graph_tuple[1]
    train_and_evaluate(config_data)


def create_config_Seq2Seq():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join("train_with_era5land", "ex1_540_basins")
    config_data = default_config_file()
    network_shp_path = "/home/wangyang1/sl_sx_usa_shps/SL_USA_HydroRiver_single.shp"
    node_shp_path = "/home/wangyang1/sl_sx_usa_shps/sl_stcd_locs/iowa_usgs_sl_stations.shp"
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
        ctx=[1, 2],
        model_name="Seq2SeqGNN",
        model_hyperparam={
            "en_input_size": 50,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": len(train_stas_basins),
            "forecast_length": 56,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=train_stas_basins,
        batch_size=len(train_stas_basins),
        forecast_history=150,
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
        train_epoch=0,
        save_epoch=1,
        train_period=[("2016-01-01-01", "2023-11-30-01")],
        test_period=[("2015-01-01-01", "2016-01-01-01")],
        valid_period=[("2015-01-01-01", "2016-01-01-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [2],
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
        model_type="GNN_MTL",
        # continue_train=True,
        network_shp=network_shp_path,
        node_shp=node_shp_path,
        basins_shp="/ftproot/basins-interim/shapes/basins.shp",
        num_workers=4,
        pin_memory=True,
        weight_path='/home/wangyang1/torchhydro/experiments/results/train_with_era5land/ex1_540_basins/model_Ep2.pth',
        layer_norm=True
    )
    # ['songliao_11205200', 'songliao_11200400', 'songliao_11007700', 'songliao_10811000', 'songliao_10541278', 'camels_03604000']
    # 更新默认配置
    update_cfg(config_data, args)
    return config_data, graph_tuple


def get_upstream_graph(basin_list, res_dir, network_path, node_path):
    import pandas as pd
    nx_graph_path = os.path.join(res_dir, f'total_graph_{len(basin_list)}.gexf')
    basin_station_path = os.path.join(res_dir, f'basin_stations_{len(basin_list)}.csv')
    if (os.path.exists(nx_graph_path)) & (os.path.exists(basin_station_path)):
        total_graph = nx.read_gexf(nx_graph_path)
        basin_station_df = pd.read_csv(basin_station_path, engine='c')
        graph_tuple = (total_graph, basin_station_df)
    else:
        # 保存成csv文件，存储节点所对应的站名、流域、上游几个点
        total_graph = nx.DiGraph()
        index_station_dict = {}
        index_basin_dict = {}
        up_len_dict = {}
        network_features = gpd.read_file(network_path)
        node_features = gpd.read_file(node_path)
        basins = [sta_id.split('_')[-1] for sta_id in basin_list]
        upstream_graphs = prepare_graph(network_features, node_features, basins)
        for key in upstream_graphs.keys():
            upstream_graph = upstream_graphs[key]
            id_col = 'ID' if 'ID' in node_features.columns else 'STCD'
            # basin_id = node_features[id_col][node_features.index == key].values[0]
            if len(upstream_graph) != 0:
                if upstream_graph.dtype == 'O':
                    # upstream_graph = array(list1, list2, list3)
                    if upstream_graph.shape[0] > 1:
                        nodes_arr = np.unique(list(chain.from_iterable(upstream_graph)))
                    else:
                        # upstream_graph = array(list1)
                        nodes_arr = upstream_graph
                else:
                    # upstream_graph = array(list1, list2) and dtype is not object
                    nodes_arr = np.unique(upstream_graph)
            else:
                upstream_graph = nodes_arr = [key]
            nodes_arr = np.append(nodes_arr, key) if key not in nodes_arr else nodes_arr
            for node in nodes_arr:
                node_id = node_features[id_col][node_features.index == node].values[0]
                basin = node_id if node_id in basins else node_features[id_col][node_features.index == key].values[0]
                index_station_dict[int(node)] = (
                    f'songliao_{node_id}' if '_' not in node_id and f'songliao_{node_id}' in basin_list
                    else f'camels_{node_id}' if '_' not in node_id
                    else node_id)
                index_basin_dict[int(node)] = f'songliao_{basin}' if f'songliao_{basin}' in basin_list else f'camels_{basin}'
            up_len_dict[int(key)] = len(nodes_arr)
            for path in upstream_graph:
                path = [path] if isinstance(path, int | np.int64) else path
                path = np.append(path, key) if key not in path else path
                nx.add_path(total_graph, path)
        basin_station_df = pd.DataFrame([index_station_dict, index_basin_dict, up_len_dict]).T
        basin_station_df = basin_station_df[~basin_station_df[0].isna()].rename(columns={0: 'station_id', 1: 'basin_id', 2: 'upstream_len'})
        basin_station_df = basin_station_df.reset_index().rename(columns={'index': 'node_id'})
        graph_tuple = (total_graph, basin_station_df)
        # 有孤立点的情况不适用于edgelist, int点不适合gml
        nx.write_gexf(total_graph, nx_graph_path)
        basin_station_df.to_csv(basin_station_path)
    return graph_tuple

def prepare_graph(network_features: gpd.GeoDataFrame, node_features: gpd.GeoDataFrame, nodes: list[int] | list[str],
                  cutoff=4):
    import hydrotopo.ig_path as htip
    # test_df_path = 's3://stations-origin/zq_stations/hour_data/1h/zq_CHN_songliao_10800300.csv'
    if isinstance(nodes[0], int):
        node_idx = nodes
    else:
        id_col = 'ID' if 'ID' in node_features.columns else 'STCD'
        node_features[id_col] = node_features[id_col].astype(str)
        node_idx = node_features.index[node_features[id_col].isin(nodes)]
    if len(node_idx) != 0:
        graph_dict = htip.find_edge_nodes_bulk_up(node_features, network_features, node_idx, cutoff)
    else:
        graph_dict = {}
    return graph_dict


if __name__ == "__main__":
    test_run_model()
