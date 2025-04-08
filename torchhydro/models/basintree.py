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
        self.nestedness = pd.concat([self.nestedness,pd.DataFrame(columns=["type_single_river", "type_leaf", "type_limb", "type_river_tree_root"])],axis=1,sort=False)

        type_single_river = [False]*self.n_basin
        n_single_river = 0
        type_leaf = [False]*self.n_basin
        n_leaf = 0
        type_limb = [False]*self.n_basin
        n_limb = 0
        type_river_tree_root = [False]*self.n_basin
        n_river_tree_root = 0

        for i in range(self.n_basin):
            nes_n_nested_within = self.nestedness.at[i, 3]
            nes_n_station_ds = self.nestedness.at[i, 1]
            if nes_n_nested_within == 0 and nes_n_station_ds == 0:
                type_single_river[i] = True
                n_single_river = n_single_river + 1
            if nes_n_nested_within == 0 and nes_n_station_ds > 0:
                type_leaf[i] = True
                n_leaf = n_leaf + 1
            if nes_n_nested_within > 0 and nes_n_station_ds > 0:
                type_limb[i] = True
                n_limb = n_limb + 1
            if nes_n_nested_within > 0 and nes_n_station_ds == 0:
                type_river_tree_root[i] = True
                n_river_tree_root = n_river_tree_root + 1
        self.nestedness['type_single_river'] = type_single_river
        self.nestedness['type_leaf'] = type_leaf
        self.nestedness['type_limb'] = type_limb
        self.nestedness['type_river_tree_root'] = type_river_tree_root
        self.n_single_river = n_single_river
        self.n_leaf = n_leaf
        self.n_limb = n_limb
        self.n_river_tree_root = n_river_tree_root

    def single_river(
        self,
        basin_id: str = None
    ):
        basin_up = None
        basin_ds = None
        return basin_up, basin_id, basin_ds

    def leaf(
        self,
        basin_id: str = None
    ):
        basin_up = None
        basin_ds = self.nestedness.at[basin_id, 2]
        return basin_up, basin_id, basin_ds

    def singel_basin(self):
        """

        Returns
        -------

        """


    def get_tree(self, basin_id):
        """

        Returns
        -------

        """
