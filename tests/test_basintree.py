
from hydrodataset import camels_fr
from torchhydro.models.basintree import Node, Basin, BasinTree
import pytest

# @pytest.fixture()
def test_Node():
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
    print(n_basin_us)  # 8


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
    basin.set_node(node)
    basin_us = basin.node.basin_us
    print(basin_us)     #['A405062001', 'A417301001', 'A420063001', 'A402061001', 'A436203001', 'A414020202', 'A443064001', 'A433301001']

def test_basin_type():
    camelsfr = camels_fr.CamelsFr()
    basin_ids = camelsfr.gage
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_tree.basin_type()
    n_single_river = basin_tree.n_single_river
    n_leaf = basin_tree.n_leaf
    n_limb = basin_tree.n_limb
    n_river_tree_root = basin_tree.n_river_tree_root
    print("n_single_river =" + str(n_single_river))
    print("n_leaf =" + str(n_leaf))
    print("n_limb =" + str(n_limb))
    print("n_river_tree_root =" + str(n_river_tree_root))
    # n_single_river = 284
    # n_leaf = 212
    # n_limb = 75
    # n_river_tree_root = 83

def test_basin_order():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id = "A550061001"
    order = basin_tree.basin_order(basin_id)
    print(order)
    # [1, 4, 2, 3, 5, 5, 5, 3, 5, 4, 2, 3, 4, 6]

def test_BasinTree():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id = "A550061001"
    basin = Basin(basin_id)
    node_id = "node_" + basin_id
    node = Node(node_id, basin_id)
    basin_us = [
        "A443064001",
        "A433301001",
        "A524201001",
        "A436203001",
        "A414020202",
        "A615103001",
        "A402061001",
        "A511061001",
        "A550061001",
        "A673122001",
        "A543101001",
        "A405062001",
        "A526102003",
        "A843101001",
        "A764201001",
        "A420063001",
        "A542201001",
        "A788101001",
        "A807101001",
        "A417301001",
        "A662121202",
        "A605102001",
        "A712201001",
    ]

