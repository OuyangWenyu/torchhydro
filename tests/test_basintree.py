from hydrodataset import camels_fr
from torchhydro.models.basintree import Node, Basin, BasinTree

def test_Node():
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
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_tree._region_basin_type()
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

def test_get_basin_index():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id_list = [
        "A405062001",
        "A417301001",
        "A420063001",
        "A402061001",
        "A436203001",
        "A414020202",
        "A443064001",
        "A433301001",
    ]
    basin_id = "A420063001"
    basin_index = basin_tree._get_basin_index(basin_id, basin_id_list)
    print(basin_index)
    # 2

def test_basin_order():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id = "A550061001"
    basin_tree, max_order, basin_list, order_list = basin_tree.basin_tree_and_order(basin_id)
    print("--basin_tree--")
    print(basin_tree)
    print("--max_order--")
    print(max_order)
    print("--basin_list--")
    print(basin_list)
    print("--order_list--")
    print(order_list)
# --basin_tree--
# [<torchhydro.models.basintree.Basin object at 0x7faef8dbf770>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8ddb110>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8ae3110>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8aaefd0>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8b0ce20>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8ae32f0>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8ddafd0>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8b05f50>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8d31d30>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8d475c0>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8489910>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8d779b0>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8d93c50>, 
# <torchhydro.models.basintree.Basin object at 0x7faef8d31a90>]
# --max_order--
# 6
# --basin_list--
# ['A550061001', 'A511061001', 'A543101001', 'A443064001', 'A542201001', 
# 'A526102003', 'A436203001', 'A524201001', 'A420063001', 'A417301001', 
# 'A405062001', 'A433301001', 'A414020202', 'A402061001']
# --order_list--
# [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6]

def test_single_basin():
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

def test_figure_out_root_single_basin():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id_list = ["A550061001", "A369011001", "A330010001"]
    root_basin, single_basin = basin_tree.figure_out_root_single_basin(basin_id_list)
    print("--root_basin--")
    print(root_basin)
    print("--single_basin--")
    print(single_basin)
    # --root_basin--  13
    # ['A405062001', 'A420063001', 'A436203001', 'A443064001', 'A511061001', 'A526102003', 'A543101001', 'A550061001',
    #  'A116003002', 'A204010101', 'A212020002', 'A273011002', 'A369011001']
    # --single_basin--  15
    # ['A140202001', 'A231020001', 'A234021001', 'A251020001', 'A284020001', 'A330010001', 'A373020001', 'A380020001',
    #  'A107020001', 'A402061001', 'A414020202', 'A417301001', 'A433301001', 'A524201001', 'A542201001']

    # --root_basin-- 2
    # ['A550061001', 'A369011001']
    # --single_basin-- 1
    # ['A330010001']


def test_get_basin_tree():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id_list = ["A550061001", "A369011001", "A284020001", "A330010001"]
    nested_model = basin_tree.get_basin_trees(basin_id_list)
    basin_trees = nested_model["basin_trees"]
    max_order = nested_model["basin_tree_max_order"]
    basin_list = nested_model["basin_list"]
    basin_list_array = nested_model["basin_list_array"]
    order_list = nested_model["order_list"]
    n_basin_per_order_list = nested_model["n_basin_per_order_list"]
    n_basin_per_order = nested_model["n_basin_per_order"]

    print("test_--basin_id_list--")
    print(basin_id_list)
    print("--basin_trees--")
    print(len(basin_trees))
    print(basin_trees)
    print("--max_order--")
    print(max_order)
    print("--basin_list_array--")
    print(basin_list_array)
    print("--basin_list--")
    print(basin_list)
    print("--order_list--")
    print(order_list)
    print("--n_basin_per_order_list--")
    print(n_basin_per_order_list)
    print("--n_basin_per_order--")
    print(n_basin_per_order)

# test_--basin_id_list--
# ['A550061001', 'A369011001', 'A284020001', 'A330010001']
# --basin_trees--
# 3
# [[[<torchhydro.models.basintree.Basin object at 0x7fc7550596a0>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc754c4c410>, <torchhydro.models.basintree.Basin object at 0x7fc754e56b70>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc7550b3bb0>, <torchhydro.models.basintree.Basin object at 0x7fc754c049e0>, 
# <torchhydro.models.basintree.Basin object at 0x7fc754e56d50>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc755139e50>, <torchhydro.models.basintree.Basin object at 0x7fc754c01950>, 
# <torchhydro.models.basintree.Basin object at 0x7fc754e96510>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc754e0f100>, <torchhydro.models.basintree.Basin object at 0x7fc754c17d10>, 
# <torchhydro.models.basintree.Basin object at 0x7fc754c047c0>, <torchhydro.models.basintree.Basin object at 0x7fc754c01750>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc754e482f0>]], 
# [[<torchhydro.models.basintree.Basin object at 0x7fc754c3e410>], 
# [<torchhydro.models.basintree.Basin object at 0x7fc754e40710>]], 
# [[<torchhydro.models.basintree.Basin object at 0x7fc754e40890>, <torchhydro.models.basintree.Basin object at 0x7fc755005ff0>]]]
# --max_order--
# 6
# --basin_list_array--
# [[['A550061001'], 
# ['A511061001', 'A543101001'], 
# ['A443064001', 'A542201001', 'A526102003'], 
# ['A436203001', 'A524201001', 'A420063001'], 
# ['A417301001', 'A405062001', 'A433301001', 'A414020202'], 
# ['A402061001']], 
# [['A369011001'], 
# ['A361011001']], 
# [['A284020001', 'A330010001']]]
# --basin_list--
# ['A550061001', 'A511061001', 'A543101001', 'A443064001', 'A542201001', 'A526102003', 'A436203001', 'A524201001', 'A420063001', 
# 'A417301001', 'A405062001', 'A433301001', 'A414020202', 'A402061001', 'A369011001', 'A361011001', 'A284020001', 'A330010001']
# --order_list--
# [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 1, 2, 1, 2]
# --n_basin_per_order_list--
# [[1, 2, 3, 3, 4, 1], [1, 1], [2]]
# --n_basin_per_order--
# [4, 3, 3, 3, 4, 1]

# >>>PYTHON-EXEC-OUTPUT
# Running pytest with args: ['-p', 'vscode_pytest', '--rootdir=/home/yulili/code/torchhydro/tests', '/home/yulili/code/torchhydro/tests/test_basintree.py::test_Basin']
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_basintree.py .                                                      [100%]

# Finished running tests!