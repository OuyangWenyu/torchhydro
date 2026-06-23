
import xarray as xr
import numpy as np
import os
from torchhydro.datasets.data_sources import CamelsUs

def check_categories_subset():
    camels_dir = "/Volumes/Untitled/data"
    source = CamelsUs(camels_dir, download=False, region="US")
    
    # The 5 gages specified by the user
    gage_id = ['01013500', '01022500', '01030500', '01031500', '01047000']
    
    one_hot_cols = ["dom_land_cover", "geol_1st_class", "geol_2nd_class"]
    
    print(f"Checking {len(gage_id)} gages: {gage_id}")
    
    total_categories = 0
    
    for col in one_hot_cols:
        ds = source.read_attr_xrdataset(gage_id, [col])
        data = ds[col].values
        print(f"Raw data for {col}: {data.flatten()}")
        unique_vals = np.unique(data)
        unique_vals = unique_vals[~np.isnan(unique_vals)]
        unique_vals.sort()
        
        print(f"{col}: {len(unique_vals)} categories: {unique_vals}")
        total_categories += len(unique_vals)
        
    print(f"Total one-hot columns: {total_categories}")
    
    # Original static attributes count: 17
    # 3 of them are converted to one-hot
    # Remaining static: 17 - 3 = 14
    # New static dimension: 14 + total_categories
    
    print(f"Original static attributes: 17")
    print(f"Static attributes converted to one-hot: 3")
    print(f"Remaining static attributes: 14")
    print(f"New static dimension: {14 + total_categories}")
    
    # Dynamic attributes: 7
    # Total input size: 7 + 14 + total_categories
    print(f"Total Input Size: {7 + 14 + total_categories}")

if __name__ == "__main__":
    check_categories_subset()
