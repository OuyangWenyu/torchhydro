import pandas as pd
from pandas import DataFrame as df
import numpy as np

class BasinTree:
    """
    generate the basin tree through catchment nestedness information
    """
    def __init__(
        self,
        nestednessinfo: df = None,
    ):
        """

        Parameters
        ----------
        nestednessinfo
        """
        self.nestedness = nestednessinfo
        self.basins = nestednessinfo.index.values
        self.n_basin = len(self.basins)
        self.max_basin_step = max(nestednessinfo.columns[3]) + 1
        self.n_single_river = 0
        self.n_leaf = 0
        self.n_river_tree_root = 0
        self.n_limb = 0



    def basin_type(self):
        """

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
            nes_n_nested_within = self.nestedness.at[i, 3]
            nes_n_station_ds = self.nestedness.at[i, 1]
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

    def single_basin(
        self,
        basin_id: str = None,
    ):
        """

        Returns
        -------

        """
        basin_type = self.nestedness.at[basin_id, "basin_type"]
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

        Returns
        -------

        """
