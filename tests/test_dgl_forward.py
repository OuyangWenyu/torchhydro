import dgl
import torch
import networkx as nx
from dgl.nn.pytorch import GATv2Conv
from torch.nn import Transformer
from torch import nn
import torch.nn.functional as F
from hydro_topo import ig_path as dig
from hydro_topo import ig_path_test as ipt


def test_train_graph():
    epoch = 10
    test_graph = dig.find_edge_nodes(ipt.gpd_df_node, ipt.gpd_df_network, 0, 'up')
    test_dgl_graph = convert_topo2dgl(test_graph)
    model = TestModel(2, 10, 2)
    opt = torch.optim.Adam(model.parameters())
    test_dgl_graph.ndata['features'] = torch.zeros(test_dgl_graph.num_nodes(), 360, 1460, dtype=torch.float)
    test_dgl_graph.ndata['train_mask'] = torch.ones(test_dgl_graph.num_nodes(), 72, 365, dtype=torch.float)
    test_dgl_graph.ndata['val_mask'] = torch.ones(test_dgl_graph.num_nodes(), 72, 365, dtype=torch.float)
    test_dgl_graph.ndata['label'] = torch.zeros(test_dgl_graph.num_nodes(), 360, 1460, dtype=torch.float)
    node_features = test_dgl_graph.ndata['features']
    train_mask = test_dgl_graph.ndata['train_mask']
    valid_mask = test_dgl_graph.ndata['val_mask']
    node_labels = test_dgl_graph.ndata['label']
    # test_mask = test_dgl_graph.ndata['test_mask']
    for epoch in range(epoch):
        model.train()
        # forward propagation by using all nodes
        logits = model(test_dgl_graph, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, test_graph, node_features, node_labels, valid_mask)
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
        # probably add some nodes which in/out degree are both 0 and add self.loop
        self.trans = Transformer(d_model=256, nhead=4)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        inputs = F.relu(self.linear(inputs))
        trans_output = self.trans(inputs)
        h = self.conv1(graph, trans_output)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
