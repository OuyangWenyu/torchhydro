
from hydrodataset import camels_fr
from torchhydro.models.basintree import Node, Basin, BasinTree

# @pytest.fixture()
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
    basin_ids = camelsfr.gage
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
    basin_tree, max_order = basin_tree.basin_tree_and_order(basin_id)
    print(basin_tree)
    # [1, 4, 2, 3, 5, 5, 5, 3, 5, 4, 2, 3, 4, 6]
    # [<torchhydro.models.basintree.Basin object at 0x0000016114C38080>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A0F0>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0B500>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A450>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D09F40>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D09A60>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A660>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0B560>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A120>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A570>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A480>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A390>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0A180>,
    # <torchhydro.models.basintree.Basin object at 0x0000016114D0B4D0>]

def test_single_basin():
    camelsfr = camels_fr.CamelsFr()
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
    basin_id_list = [
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
        "A402061001",
        "A405062001",
        "A414020202",
        "A417301001",
        "A420063001",
        "A433301001",
        "A436203001",
        "A443064001",
        "A511061001",
        "A524201001",
        "A526102003",
        "A542201001",
        "A543101001",
        "A550061001",
    ]
    root_basin, single_basin = basin_tree.figure_out_root_single_basin(basin_id_list)
    print("--root_basin--")
    print(root_basin)
    print("--single_basin--")
    print(single_basin)
    # --root_basin--
    # ['A405062001', 'A420063001', 'A436203001', 'A443064001', 'A511061001', 'A526102003', 'A543101001', 'A550061001',
    #  'A116003002', 'A204010101', 'A212020002', 'A273011002', 'A369011001']
    # --single_basin--
    # ['A140202001', 'A231020001', 'A234021001', 'A251020001', 'A284020001', 'A330010001', 'A373020001', 'A380020001',
    #  'A107020001', 'A402061001', 'A414020202', 'A417301001', 'A433301001', 'A524201001', 'A542201001']

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

def test_get_basin_tree():
    camelsfr = camels_fr.CamelsFr()
    nestedness_info = camelsfr.read_nestedness_csv()
    basin_tree = BasinTree(nestedness_info)
    basin_id_list = [
        "A116003002",
        "A140202001",
        "A550061001",

    ]
    basin_trees, max_order = basin_tree.get_basin_trees(basin_id_list)
    print("--basin_trees--")
    print(len(basin_trees))
    print(basin_trees)
    print("--max_order--")
    print(max_order)
# --basin_trees--
# 14
# [[<torchhydro.models.basintree.Basin object at 0x000001DEEE5CD790>, <torchhydro.models.basintree.Basin object at 0x000001DEEE5CD100>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEC0DAA20>, <torchhydro.models.basintree.Basin object at 0x000001DEEE4AC080>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE4AD0A0>, <torchhydro.models.basintree.Basin object at 0x000001DEE96F1610>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEC016000>],
# [<torchhydro.models.basintree.Basin object at 0x000001DED916E990>, <torchhydro.models.basintree.Basin object at 0x000001DEE9779460>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE507CB0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DB20>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69DAC0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DC10>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69DAF0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DC70>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69C050>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DF40>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69E030>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E000>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69DEE0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DDF0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69D8E0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DBE0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69C500>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69DBB0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69DA00>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69C350>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69C440>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69D460>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69D880>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69D7C0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69D5B0>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69D3D0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E210>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69E780>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E420>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69E5A0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E810>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69D0D0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E6F0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69E8A0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E2D0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69E4B0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E3C0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69E660>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69E930>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69E9C0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69EA50>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69EAE0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69EB70>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69EC00>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69EC90>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69ED20>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69EDB0>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69EE40>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69EED0>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69EF60>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F020>],
# [<torchhydro.models.basintree.Basin object at 0x000001DEEE69F0B0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F140>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F1D0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F260>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F2F0>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F380>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F410>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F4A0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F530>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F5C0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F650>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F6E0>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F770>, <torchhydro.models.basintree.Basin object at 0x000001DEEE69F800>,
# <torchhydro.models.basintree.Basin object at 0x000001DEEE69F890>]]
# --max_order--
# 6
# total basin 73

# [[<torchhydro.models.basintree.Basin object at 0x000002223411F1D0>, <torchhydro.models.basintree.Basin object at 0x000002223411C680>,
# <torchhydro.models.basintree.Basin object at 0x00000222342C1310>, <torchhydro.models.basintree.Basin object at 0x000002223411F290>,
# <torchhydro.models.basintree.Basin object at 0x00000222342C1340>, <torchhydro.models.basintree.Basin object at 0x00000222342C12B0>,
# <torchhydro.models.basintree.Basin object at 0x000002223411F350>, <torchhydro.models.basintree.Basin object at 0x00000222342C1370>,
# <torchhydro.models.basintree.Basin object at 0x00000222342C1250>, <torchhydro.models.basintree.Basin object at 0x000002223411EF00>,
# <torchhydro.models.basintree.Basin object at 0x00000222342C1280>, <torchhydro.models.basintree.Basin object at 0x00000222342C1520>,
# <torchhydro.models.basintree.Basin object at 0x00000222342C1430>, <torchhydro.models.basintree.Basin object at 0x00000222342C11F0>], []]
# --basin_trees--
# 2
# --max_order--
# 6

# --basin_trees--
# 3
# [[<torchhydro.models.basintree.Basin object at 0x0000025D89F8EDB0>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8EF30>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8E870>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8ECF0>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8EB10>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8E990>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8ED20>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8EA50>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8E450>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8EE10>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8EC90>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8EC60>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8E930>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8E8D0>],
# [<torchhydro.models.basintree.Basin object at 0x0000025D89F8E270>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8E120>,
# <torchhydro.models.basintree.Basin object at 0x0000025D89F8E750>, <torchhydro.models.basintree.Basin object at 0x0000025D89F8E2D0>],
# [<torchhydro.models.basintree.Basin object at 0x0000025D89F8E690>]]  19
# --max_order--
# 6
