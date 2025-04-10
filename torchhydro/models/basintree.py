import pandas as pd
from pandas import DataFrame as df
import numpy as np


class Node:
    """
    basin node, similar to river node.
    """
    def __init__(self, node_id, basin_id):
        self.node_name = node_id
        self.basin_ds = basin_id
        self.basin_us = []

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
        self.basin_name = basin_id
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
    This class use to calculate the link/topology relationship between basins.   the link relationship between basins for regions in camels may need take into account.
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
        self.nestedness = nestednessinfo
        self.basins = nestednessinfo.index.values
        self.n_basin = len(self.basins)
        nes_n_nested_within = nestednessinfo["nes_n_nested_within"].tolist()
        self.max_basin_order = max(nes_n_nested_within) + 1
        self.n_single_river = 0
        self.n_leaf = 0
        self.n_river_tree_root = 0
        self.n_limb = 0
        self.basin_type()  #

    def basin_type(self):
        """
        calculate the type for each basin. basin type: single_river, leaf, limb, river_tree_root.
        Returns
        -------

        """
        self.nestedness = pd.concat([self.nestedness,pd.DataFrame(columns=["type_single_river", "type_leaf", "type_limb", "type_river_tree_root", "basin_type"])],axis=1,sort=False)

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


    def basin_order(
        self,
        basin_id: str = None,
    ):
        """
        calculate the basin order
        Parameters
        ----------
        basin_id

        Returns
        -------

        """
        basin_us = self.nestedness.at[basin_id, "nes_station_nested_within"]
        if not basin_us==None:
            basin_us = basin_us.split(",")
        basin = [basin_id] + basin_us
        n_basin = len(basin)

        basin_object = [Basin]*n_basin
        for i in range(n_basin):
            basin_i = Basin(basin[i])
            basin_i.basin_type = self.nestedness.at[basin[i], "basin_type"]
            node_id = "node_" + basin[i]
            node = Node(node_id, basin[i])
            basin_i.set_node(node)
            basin_object[i] = basin_i

        # basin order
        basin_object[0].set_basin_order(1)
        for i in range(1, n_basin):
            basin_i = basin[i]
            order_i = 2
            while True:
                basin_ds = self.nestedness.at[basin_i, "nes_next_station_ds"]
                if basin_ds == basin[0]:
                    basin_object[i].set_basin_order(order_i)
                    break
                else:
                    order_i = order_i + 1
                    basin_i = basin_ds

        # upstream basin
        for i in range(n_basin):
            basin_i = basin[i]
            basin_ds = self.nestedness.at[basin_i, "nes_next_station_ds"]
            basin_ds_index = self._get_basin_index(basin_ds, basin)
            if basin_ds_index >= 0:
                basin_object[basin_ds_index].node.add_basin_us(basin[i])

        # sort along order

        return basin_object

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
        basin_type = self.nestedness.at[basin_id, "basin_type"]
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

    def get_tree(self, basin_id):
        """
        return the tree of basin_id_list
        Returns
        -------

        """

        return 0


    def figure_out_root_single_basin(self, basin_id_list: list = None):
        """figure out the root basins and single basins among basin_id_list"""

        n_basin = len(basin_id_list)

        # sort
        single_river = []
        leaf = []
        limb = []
        river_tree_root = []
        for i in range(n_basin):
            basin_type = self.nestedness.at[basin_id_list[i], "basin_type"]
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
            river_tree_root_us = self.nestedness.at[river_tree_root[i], "nes_station_nested_within"]
            river_tree_root_us.split(",")
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
            lim_us = self.nestedness.at[limb[i], "nes_next_station_ds"]
            lim_us = lim_us.split(",")
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


    def set_cal_order(self, basin_id_list: list = None):
        """set the calculate order of basin_id_list and its tree"""

        return 0
