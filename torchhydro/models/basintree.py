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
        self.n_basin_tree = 1
        self.tree_root = []




    def generate_basin_tree(self):
        """

        Returns
        -------

        """
        self.nestedness = pd.concat([self.nestedness,pd.DataFrame(columns=["tree_root","tree_leaf","single_river","river_tree","limb"])],axis=1,sort=False)
        tree_root = []
        for i in range(self.n_basin):
            if self.nestedness.at[i,1] < 1:
                tree_root.append(True)
            else:
                tree_root.append(False)


    def get_tree(self, basin_id):
        """

        Returns
        -------

        """
