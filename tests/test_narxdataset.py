
import pytest
import os
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.datasets.narxdataset import NarxDataset


class TestDatasource:
    """
    a calss for testing the narxdataset.
    """
    def __init__(self, source_cfgs, time_unit="1D"):
        self.ngrid = 2
        self.nt = 366
        self.data_cfgs = source_cfgs

    def read_ts_xrdataset(self, basin_id, t_range, var_lst):
        """
        
        """

        return 0

def test_NarxDataset():
    """

    Returns
    -------

    """
    temp_test_path = "results/test_camels/narxdataset_camelsfr"
    os.makedirs(temp_test_path, exist_ok=True)
    data_cfgs = {
        "source_cfgs": {
            "source_name": "camels_fr",
            "source_path": "camels/camels_fr",
        },
        "test_path": str(temp_test_path),
        "object_ids": [
            "A550061001", "A369011001", "A330010001"
        ],  # Add this line with the actual object IDs
        "t_range_train": [
            "2017-10-01", 
            "2018-10-01",
        ],  # Add this line with the actual start and end dates for training.
        "t_range_test": [
            "2018-10-01", 
            "2019-10-01"
        ],  # Add this line with the actual start and end dates for validation.
        "relevant_cols": [
            # List the relevant column names here.
            "tsd_prec",
            "tsd_pet_ou",
            # ... other relevant columns ...
        ],
        "target_cols": [
            # List the target column names here.
            "streamflow",
            # ... other target columns ...
        ],
        "constant_cols": [
            # List the constant column names here.
            # "top_altitude_mean",
            # "top_slo_mean",
            # "sta_area_snap",
            # "top_drainage_density",
            # ... other constant columns ...
        ],
        "forecast_history": 10,
        "warmup_length": 10,
        "forecast_length": 1,
        "min_time_unit": "D",
        "min_time_interval": 1,
        "target_rm_nan": True,
        "relevant_rm_nan": True,
        "constant_rm_nan": True,
        "scaler": "StandardScaler",  # Add the scaler configuration here
        "stat_dict_file": None,  # Added the missing configuration
        "b_nestedness": True,
    }
    is_tra_val_te = "train"
    dataset = NarxDataset(data_cfgs, is_tra_val_te)
    print(dataset)

# test_narxdataset.py Finish Normalization


#   0%|          | 0/3 [00:00<?, ?it/s]
# 100%|██████████| 3/3 [00:00<00:00, 850.71it/s]
# Finish Normalization


#   0%|          | 0/3 [00:00<?, ?it/s]
# 100%|██████████| 3/3 [00:00<00:00, 1245.34it/s]

# test_camelsfr_netsednarx.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 1255.47it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 580.73it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 13739.30it/s]