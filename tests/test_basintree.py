
from hydrodataset import camels_fr
from torchhydro.models.basintree import Node, Basin, BasinTree
import pytest

# @pytest.fixture()
def test_Node():
    # camelsfr = camels_fr.CamelsFr()
    # gage_ids = camelsfr.gage
    gage_id = [
        "A105003001",
        "A107020001",
        "A112020001",
        "A116003002",
        "A140202001",
        "A202030001",
        "A204010101",
        "A211030001",
        "A212020002",
        "A231020001",
        "A234021001",
        "A251020001",
        "A270011001",
        "A273011002",
        "A284020001",
        "A330010001",
        "A361011001",
        "A369011001",
        "A373020001",
        "A380020001",
    ],
    # nestedness_info = camelsfr.read_nestedness_csv()
    # basin_tree = BasinTree(nestedness_info)
    basin_id = "A511061001"
    node_id = "node_" + basin_id
    node = Node(node_id, basin_id)
    basin_us = [
        "A405062001",
        "A417301001",
        "A420063001",
        "A402061001",
        "A436203001",
        "A414020202",
        "A443064001",
        "A433301001",
    ]
    for i in range(len(basin_us)):
        node.add_basin_us(basin_us[i])
    n_basin_us = node.amount_basin_us()
    print(n_basin_us)
    # 8

def test_Basin():
    basin_id = "A511061001"
    basin = Basin(basin_id)
    node_id = "node_" + basin_id
    node = Node(node_id, basin_id)
    basin_us = [
        "A405062001",
        "A417301001",
        "A420063001",
        "A402061001",
        "A436203001",
        "A414020202",
        "A443064001",
        "A433301001",
    ]
    for i in range(len(basin_us)):
        node.add_basin_us(basin_us[i])
    basin_node = basin.set_node(node)
    basin_us = basin.node.basin_us


