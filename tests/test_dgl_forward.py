import dgl
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import hydrodatasource.configs.config as hdscc
from dgl.nn.pytorch import GATv2Conv
from hydrotopo import ig_path as dig
from torch import nn
from torch.nn import Transformer
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset

from tests.test_gen_train_dataset_from_csv import gen_train_test_xy


def test_train_graph():
    epoch = 10
    batch_size = 256
    t_node = '21401550'
    node_features = gpd.read_file('463_sl_nodes.shp', engine='pyogrio')
    network_features = gpd.read_file("/home/wangyang1/songliao_cut_single.shp", engine='pyogrio')
    test_dgl_graph, nodes_arr, train_test_tensor_x, train_test_tensor_y = prepare_xy(network_features=network_features, node_features=node_features,
                    node=t_node, start_time='2019-01-01 00:00:00', end_time='2024-05-31 23:00:00')
    # 对于水文站而言，点特征是经纬度、距离它最近的河流等（构成的数组第一维与节点数一致，才能正确分配到不同节点上），而不是输入x和输出y
    test_dgl_graph.ndata['features'] = train_test_tensor_x
    test_dgl_graph.ndata['train_mask'] = torch.full((test_dgl_graph.num_nodes(), test_dgl_graph.num_nodes(), int(len(train_test_tensor_x) / 24)), True)
    test_dgl_graph.ndata['test_mask'] = torch.full((test_dgl_graph.num_nodes(), test_dgl_graph.num_nodes(), int(len(train_test_tensor_x) / 24)), True)
    test_dgl_graph.ndata['label'] = train_test_tensor_x
    node_features = test_dgl_graph.ndata['features']
    train_mask = test_dgl_graph.ndata['train_mask']
    valid_mask = test_dgl_graph.ndata['test_mask']
    node_labels = test_dgl_graph.ndata['label']
    test_dataset = TensorDataset(train_test_tensor_x)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = TestModel(len(test_dgl_graph.nodes()), 128, len(test_dgl_graph.nodes()))
    opt = torch.optim.Adam(model.parameters())
    # test_mask = test_dgl_graph.ndata['test_mask']
    for epoch in range(epoch):
        for batch_idx, (data,) in enumerate(test_dataloader):
            model.train()
            # forward propagation by using all nodes
            logits = model(test_dgl_graph, data)
            # compute loss
            loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
            # compute validation accuracy
            acc = evaluate(model, test_dgl_graph, node_features, node_labels, valid_mask)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item(), f'acc: {acc}')
    torch.save(model, 'model.pth')


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def convert_topo2dgl(paths: list):
    nx_interim = nx.DiGraph()
    for path_list in paths:
        nx.add_path(nx_interim, path_list)
    dgl_tensor = dgl.from_networkx(nx_interim)
    return dgl_tensor


def prepare_xy(network_features: gpd.GeoDataFrame, node_features: gpd.GeoDataFrame, node: int | str,
               start_time=None, end_time=None, cutoff=2147483647, pre_index=24, post_index=72):
    # test_df_path = 's3://stations-origin/zq_stations/hour_data/1h/zq_CHN_songliao_10800300.csv'
    # test_df_stcd = test_df_path.split('.')[0].split('/')[-1]
    if isinstance(node, int):
        node_idx = node
    else:
        if 'STCD' in node_features.columns:
            node_idx = node_features.index[node_features['STCD'] == node]
        else:
            node_idx = node_features.index[node_features['ID'] == node]
    ig_graph = dig.find_edge_nodes(node_features, network_features, node_idx, 'up', cutoff)
    dgl_graph = convert_topo2dgl(ig_graph)
    origin_df = read_data_with_stcd_from_minio(node)
    dt_range = pd.date_range(start_time, end_time, freq='h')
    train_test_df_x = pd.DataFrame()
    # ig_graph展成list之后，节点序号也是从小到大排列，与dgl_graph.nodes()排号形成对应，不需要再排序添加数据
    nodes_arr = np.unique(list(chain.from_iterable(ig_graph)))
    # up_nodes_arr = nodes_arr[nodes_arr != node_idx[0]]
    for up_node in nodes_arr:
        # 应有一个up_nodes到Sequence[Path]的方法
        if 'STCD' in node_features.columns:
            up_node_name = node_features['STCD'][node_features.index == up_node].to_list()[0]
        else:
            up_node_name = node_features['ID'][node_features.index == up_node].to_list()[0]
        upper_origin_df = read_data_with_stcd_from_minio(up_node_name)
        # 不同站点的dt_range可能不同, 会对学习效果产生影响
        train_test_node_x = gen_train_test_xy(upper_origin_df, dt_range, pre_index=pre_index, post_index=post_index)[0]
        train_test_df_x = pd.concat([train_test_df_x, train_test_node_x], axis=1)
    train_test_df_y = gen_train_test_xy(origin_df, dt_range, pre_index=pre_index, post_index=post_index)[1]
    # train_test_data_x = train_test_df_x.drop(columns=['index'])
    # train_test_data_y = train_test_df_y.drop(columns=['index'])
    train_test_array_x = train_test_df_x.apply(lambda x: np.array_split(x.to_numpy(), x.to_numpy().shape[0] // (pre_index+1)), axis=1)
    train_test_array_y = train_test_df_y.apply(lambda y: np.array_split(y.to_numpy(), y.to_numpy().shape[0] // (post_index+1)), axis=1)
    train_test_tensor_x = torch.Tensor(train_test_array_x).transpose(0, 1)
    train_test_tensor_y = torch.Tensor(train_test_array_y).transpose(0, 1)
    return dgl_graph, nodes_arr, train_test_tensor_x, train_test_tensor_y


def prepare_features(node_features: gpd.GeoDataFrame, nodes_graph):
    node_x = node_features.geometry[node_features.index.isin(nodes_graph)].x.to_numpy()
    node_y = node_features.geometry[node_features.index.isin(nodes_graph)].y.to_numpy()
    graph_features = np.concatenate([node_x, node_y]).reshape(len(nodes_graph), -1).transpose()
    return torch.Tensor(graph_features)


def read_data_with_stcd_from_minio(stcd: str):
    minio_path_zq_chn = f's3://stations-origin/zq_stations/hour_data/1h/zq_CHN_songliao_{stcd}.csv'
    minio_path_zz_chn = f's3://stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_{stcd}.csv'
    minio_path_zq_usa = f's3://stations-origin/zq_stations/hour_data/1h/zq_USA_usgs_{stcd}.csv'
    minio_path_zz_usa = f's3://stations-origin/zz_stations/hour_data/1h/zz_USA_usgs_{stcd}.csv'
    minio_path_zq_usa_new = f's3://stations-origin/zq_stations/hour_data/1h/usgs_datas_462_basins_after_2019/zz_USA_usgs_{stcd}.csv'
    camels_hourly_files = f's3://datasets-origin/camels-hourly/data/usgs_streamflow_csv/{stcd}-usgs-hourly.csv'
    minio_data_paths = [minio_path_zq_chn, minio_path_zz_chn, minio_path_zq_usa, minio_path_zz_usa, minio_path_zq_usa_new, camels_hourly_files]
    hydro_df = None
    for data_path in minio_data_paths:
        if hdscc.FS.exists(data_path):
            hydro_df = pd.read_csv(data_path, engine='c', storage_options=hdscc.MINIO_PARAM)
            break
    if hydro_df is None:
        interim_df = pd.read_sql(f"SELECT * FROM ST_RIVER_R WHERE stcd = '{stcd}'", hdscc.PS)
        if len(interim_df) == 0:
            interim_df = pd.read_sql(f"SELECT * FROM ST_RSVR_R WHERE stcd = '{stcd}'", hdscc.PS)
        hydro_df = interim_df
    return hydro_df


class TestModel(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, in_feats)
        self.conv1 = GATv2Conv(
            in_feats=in_feats, out_feats=hid_feats, num_heads=3)
        self.conv2 = GATv2Conv(
            in_feats=hid_feats, out_feats=out_feats, num_heads=3)
        # https://blog.csdn.net/zhaohongfei_358/article/details/126019181
        # d_model should be max(amount of patches/stations on upstream)
        # however amount of stations in actual graph is different from max
        # probably need to add some nodes which in/out degree are both 0 and add self.loop
        self.trans = Transformer(d_model=256, nhead=4)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        inputs = F.relu(self.linear(inputs))
        # RuntimeError: the feature number of src and tgt must be equal to d_model
        trans_output = self.trans(inputs)
        h = self.conv1(graph, trans_output)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


