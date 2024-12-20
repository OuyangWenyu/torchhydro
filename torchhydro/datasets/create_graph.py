import os
import networkx as nx
import geopandas as gpd
import numpy as np
from itertools import chain

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
