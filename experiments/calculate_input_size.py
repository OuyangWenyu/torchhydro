
import xarray as xr
import numpy as np
import os
import sys
from torchhydro.configs.config import default_config_file
from torchhydro.configs.data_resolver import open_dataset

def calculate_input_size():
    # Initialize CamelsUS data source via the unified resolver
    source = open_dataset("camels_us", region="US")
    
    # Attributes to check
    one_hot_cols = ["dom_land_cover", "geol_1st_class", "geol_2nd_class"]
    
    # Get all gage IDs (or a subset if faster)
    # We need to check ALL gages to be sure about categories
    # Get all gage IDs
    gages = source.read_object_ids()
    
    print(f"Checking {len(gages)} gages...")
    
    total_categories = 0
    
    for col in one_hot_cols:
        # Read attribute
        # Note: read_constant_cols returns (N, 1)
        #data = source.read_constant_cols(gages, [col])
        ds = source.read_attr_xrdataset(gages, [col])
        data = ds[col].values
        
        # Get unique values
        unique_vals = np.unique(data)
        unique_vals = unique_vals[~np.isnan(unique_vals)]
        
        print(f"{col}: {len(unique_vals)} categories: {unique_vals}")
        total_categories += len(unique_vals)
        
    print(f"Total extra columns from one-hot: {total_categories}")
    
    # Base input size
    # 7 dynamic + 17 static = 24
    # Minus 3 static that are converted = 21
    # Plus total_categories
    
    base_size = 21
    final_size = base_size + total_categories
    print(f"Estimated Input Size: {final_size}")

if __name__ == "__main__":
    calculate_input_size()
