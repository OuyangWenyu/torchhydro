"""
generate basin tree.
"""
import pandas as pd
from pandas import DataFrame as df


class Node:
    """
    basin node, similar to river node.
    """
    def __init__(self, node_id, basin_id):
        self.node_id = node_id
        self.basin_ds = basin_id  # downstream basin
        self.basin_us = []  # upstream basin

    def add_basin_us(self, basin_id):
        self.basin_us.append(basin_id)

    def amount_basin_us(self):
        n_basin_us = len(self.basin_us)
        return n_basin_us


class Basin:
    """
    basin object, similar to river.
    """
    def __init__(self, basin_id):
        self.basin_id = basin_id
        self.basin_type = ""
        self.node = None
        self.basin_order = -1  # basin order, similar to river order
        self.max_order_of_tree = -1
        self.cal_order = self.max_order_of_tree - self.basin_order

    def set_basin_type(self, basin_type):
        self.basin_type = basin_type

    def set_node(self, node: Node = None):
        self.node = node

    def set_basin_order(self, basin_order):
        self.basin_order = basin_order

    def set_max_order_of_tree(self, max_order_of_tree):
        self.max_order_of_tree = max_order_of_tree

    def refresh_cal_order(self):
        self.cal_order = self.max_order_of_tree - self.basin_order


class BasinTree:
    """
    generate the basin tree through catchment nestedness information
    This class use to figure out the link/topology relationship between basins.
    the link relationship between basins for region in camels may need take into account.
    """
    def __init__(
        self,
        nestednessinfo: df = None,
        basin_id_list: list = None,
    ):
        """
        Initialize the basin tree of the whole region.

        Parameters
        ----------
        nestednessinfo
        basin_id_list: basins chose to forecasting.
        """
        # whole region
        self.nestedness = nestednessinfo
        self.basins = nestednessinfo.index.values
        self.n_basin = len(self.basins)
        nes_n_nested_within = nestednessinfo["nes_n_nested_within"].tolist()
        self.max_basin_order = max(nes_n_nested_within) + 1
        self.n_single_river = 0
        self.n_leaf = 0
        self.n_river_tree_root = 0
        self.n_limb = 0
        self._region_basin_type()

        # basin_id_list
        self.basin_id_list = basin_id_list
        self.basin_tree = None
        self.basin_tree_max_order = 0
        self.basin_list = None
        self.order_list = None
        

    def _region_basin_type(self):
        """
        figure out the type for each basin. basin type: single_river, leaf, limb, river_tree_root.

        Returns
        -------

        """
        self.nestedness = pd.concat([
                self.nestedness, 
                pd.DataFrame(columns=[
                    "type_single_river",
                    "type_leaf",
                    "type_limb",
                    "type_river_tree_root",
                    "basin_type"
                ])],
            axis=1, 
            sort=False
        )

        type_single_river = [False]*self.n_basin
        n_single_river = 0
        type_leaf = [False]*self.n_basin
        n_leaf = 0
        type_limb = [False]*self.n_basin
        n_limb = 0
        type_river_tree_root = [False]*self.n_basin
        n_river_tree_root = 0
        basin_type = [""]*self.n_basin

        for i in range(self.n_basin):
            nes_n_nested_within = self.nestedness.at[self.basins[i], "nes_n_nested_within"]
            nes_n_station_ds = self.nestedness.at[self.basins[i], "nes_n_station_ds"]
            if nes_n_nested_within == 0 and nes_n_station_ds == 0:
                type_single_river[i] = True
                n_single_river = n_single_river + 1
                basin_type[i] = "single_river"
            if nes_n_nested_within == 0 and nes_n_station_ds > 0:
                type_leaf[i] = True
                n_leaf = n_leaf + 1
                basin_type[i] = "leaf"
            if nes_n_nested_within > 0 and nes_n_station_ds > 0:
                type_limb[i] = True
                n_limb = n_limb + 1
                basin_type[i] = "limb"
            if nes_n_nested_within > 0 and nes_n_station_ds == 0:
                type_river_tree_root[i] = True
                n_river_tree_root = n_river_tree_root + 1
                basin_type[i] = "river_tree_root"
        self.nestedness['type_single_river'] = type_single_river
        self.nestedness['type_leaf'] = type_leaf
        self.nestedness['type_limb'] = type_limb
        self.nestedness['type_river_tree_root'] = type_river_tree_root
        self.nestedness['basin_type'] = basin_type
        self.n_single_river = n_single_river
        self.n_leaf = n_leaf
        self.n_limb = n_limb
        self.n_river_tree_root = n_river_tree_root

    def single_river(
        self,
        basin_id: str = None
    ):
        basin_us = None
        basin_ds = None
        return basin_us, basin_ds

    def leaf(
        self,
        basin_id: str = None
    ):
        basin_us = None
        basin_ds = self.nestedness.at[basin_id, 2]
        return basin_us, basin_ds

    def limb(
        self,
        basin_id: str = None
    ):
        basin_us = self.nestedness.at[basin_id, 5]
        if not basin_us==None:
            basin_us = basin_us.split(",")
            n_basin_us = len(basin_us)

        basin_ds = self.nestedness.at[basin_id, 2]
        return basin_us, basin_ds

    def river_tree_root(
        self,
        basin_id: str = None
    ):
        basin_us = self.nestedness.at[basin_id, 5]
        if not basin_us==None:
            basin_us = basin_us.split(",")
            n_basin_us = len(basin_us)

        basin_ds = self.nestedness.at[basin_id, 2]
        return basin_us, basin_ds

    def get_basin_type(self, basin_id: str = None):
        basin_type = self.nestedness.at[basin_id, "basin_type"]
        return basin_type

    def get_upstream_basin(self, basin_id: str = None):
        basin_us = self.nestedness.at[basin_id, "nes_station_nested_within"]
        if not basin_us == None:
            basin_us = basin_us.split(",")
        return basin_us

    def get_downstream_basin(self, basin_id: str = None):
        basin_ds = self.nestedness.at[basin_id, "nes_next_station_ds"]
        return basin_ds

    def basin_tree_and_order(
        self,
        basin_id: str = None,
    ):
        """
            give a basin, generate its the basin tree and order.
        Parameters
        ----------
        basin_id: str
            root basin id
        Returns
        -------

        """
        basin_us = self.get_upstream_basin(basin_id)
        basin = [basin_id] + basin_us
        n_basin = len(basin)

        # generate basin object, containing the basin node.
        basin_object = []
        for i in range(n_basin):
            basin_id = basin[i]
            basin_i = self.generate_basin_object(basin_id)
            basin_object.append(basin_i)

        # basin order
        basin_object[0].set_basin_order(1)
        max_order = 1
        order = [1]*n_basin
        for i in range(1, n_basin):
            basin_i = basin[i]
            order_i = 2
            while True:
                basin_ds = self.get_downstream_basin(basin_i)
                if basin_ds == basin[0]:
                    basin_object[i].set_basin_order(order_i)
                    order[i] = order_i
                    if order_i > max_order:
                        max_order = order_i
                    break
                else:
                    order_i = order_i + 1
                    basin_i = basin_ds

        # upstream basin of directly linking to this basin.
        for i in range(n_basin):
            basin_i = basin[i]
            basin_ds = self.get_downstream_basin(basin_i)
            basin_ds_index = self._get_basin_index(basin_ds, basin)
            if basin_ds_index >= 0:
                basin_object[basin_ds_index].node.add_basin_us(basin[i])

        # sort along order
        basin_tree = []
        basin_list = []
        order_list = []
        order_index = list(range(n_basin))
        for i in range(n_basin):
            for j in range(n_basin-1-i):
                if order[j] > order[j+1]:
                    temp_order = order[j+1]
                    order[j+1] = order[j]
                    order[j] = temp_order
                    temp_order_index = order_index[j+1]
                    order_index[j+1] = order_index[j]
                    order_index[j] = temp_order_index
        for i in range(n_basin):
            basin_tree.append(basin_object[order_index[i]])
            basin_list.append(basin_object[order_index[i]].basin_id)
            order_list.append(basin_object[order_index[i]].basin_order)
        # basin list and order list

        return basin_tree, max_order, basin_list, order_list

    def generate_basin_object(self, basin_id: str = None):
        """generate a basin object"""
        basin = Basin(basin_id)
        basin.basin_type = self.get_basin_type(basin_id)
        node_id = "node_" + basin_id
        node = Node(node_id, basin_id)
        basin.set_node(node)

        return basin

    def _get_basin_index(self, basin_id, basin_id_list: list) -> int:
        """return the index of basin in basin_id_list"""
        n = len(basin_id_list)
        index = -1
        for i in range(n):
            if basin_id_list[i] == basin_id:
                index = i
                break
        return index

    def single_basin(
        self,
        basin_id: str = None,
    ):
        """
        Strahler
        Returns
        -------

        """
        basin = Basin(basin_id)
        basin_type = self.get_basin_type(basin_id)
        basin.set_basin_type(basin_type)

        basin_us = None
        basin_ds = None
        if basin_type == "single_river":
            basin_us, basin_ds = self.single_river(basin_id)
        if basin_type == "leaf":
            basin_us, basin_ds = self.leaf(basin_id)
        if basin_type == "limb":
            basin_us, basin_ds = self.limb(basin_id)
        if basin_type == "river_tree_root":
            basin_us, basin_ds = self.river_tree_root(basin_id)

        return basin_us, basin_ds

    def figure_out_root_single_basin(self, basin_id_list: list = None):
        """
            figure out the root basins and single basins among basin_id_list
        Parameters
        ----------
        basin_id_list: list
            id list of basins which need to forcasting.
        Returns
        -------
        root_basins: list
            root basin among basin_id_list
        single_basins: list
            single basin among basin_id_list
        """
        n_basin = len(basin_id_list)

        # sort
        single_river = []
        leaf = []
        limb = []
        river_tree_root = []
        for i in range(n_basin):
            basin_type = self.get_basin_type(basin_id_list[i])
            if basin_type == "single_river":
                single_river.append(basin_id_list[i])
            if basin_type == "leaf":
                leaf.append(basin_id_list[i])
            if basin_type == "limb":
                limb.append(basin_id_list[i])
            if basin_type == "river_tree_root":
                river_tree_root.append(basin_id_list[i])

        # figure out the root basin among basin_id_list
        for i in range(len(river_tree_root)):
            # upstream basin of river_tree_root
            river_tree_root_us = self.get_upstream_basin(river_tree_root[i])
            # checking whether the leafs and limbs are contained or not, if true, remove it from its list
            leaf_in = []
            for j in range(len(leaf)):
                if leaf[j] in river_tree_root_us:
                    leaf_in.append(j)
            for j in range(len(leaf_in)):
                del leaf[leaf_in[j]]
            limb_in = []
            for j in range(len(limb)):
                if limb[j] in river_tree_root_us:
                    limb_in.append(j)
            for j in range(len(limb_in)):
                del limb[limb_in[j]]
        # checking whether a limb are contained in upstream of another limb , if true, remove it from limb list
        while True:
            if len(limb) <= 1:
                break
            i = 0
            lim_us = self.get_upstream_basin(limb[i])
            limb_in = []
            for j in range(1, len(limb)):
                if limb[j] in lim_us:
                    limb_in.append(j)
            if len(limb_in) == 0:
                break
            for j in range(len(limb_in)):
                del limb[limb_in[j]]
        root_basin = limb + river_tree_root

        # the remaining are single basin
        single_basin = single_river + leaf

        return root_basin, single_basin

    def get_basin_trees(self, basin_id_list: list = None):
        """get the basin tree and order of basin_id_list"""
        if basin_id_list is None:
            basin_id_list = self.basin_id_list
        root_basin, single_basin = self.figure_out_root_single_basin(basin_id_list)
        n_root_basin = len(root_basin)
        n_single_basin = len(single_basin)

        basin_trees = []
        basin_list = []
        order_list = []
        # root basin
        max_order = 1
        for i in range(n_root_basin):
            basin_i = root_basin[i]
            basin_tree_i, max_order_i, basin_list_i, order_list_i = self.basin_tree_and_order(basin_i)
            basin_trees.append(basin_tree_i)
            basin_list = basin_list + basin_list_i
            order_list = order_list + order_list_i
            if max_order_i > max_order:
                max_order = max_order_i

        # single basin
        single_basin_object = []
        single_basin_order = []
        for i in range(n_single_basin):
            basin_id = single_basin[i]
            basin = self.generate_basin_object(basin_id)
            single_basin_object.append(basin)
            single_basin_order.append(1)
        basin_trees.append(single_basin_object)
        basin_list = basin_list + basin_list_i
        order_list = order_list + order_list_i

        self.basin_tree = basin_trees
        self.basin_tree_max_order = max_order
        self.basin_list = basin_list
        self.order_list = order_list

        return basin_trees, max_order, basin_list, order_list

    def get_cal_order(self, basin_trees: list = None):
        """
            set the calculate order of basin_id_list and its tree
        Parameters
        ----------
        basin_id_list
        basin_trees

        Returns
        -------

        """
        

        cal_order = 0

        return cal_order
