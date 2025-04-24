
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
            "A550061001", "A369011001", "A284020001", "A330010001", 
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
        # "scaler": "StandardScaler",  # Add the scaler configuration here
        "scaler": "DapengScaler",
        "scaler_params": {
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "tsd_prec",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "tsd_pet_ou",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        "stat_dict_file": None,  # Added the missing configuration
        "b_nestedness": True,
    }
    is_tra_val_te = "train"
    dataset = NarxDataset(data_cfgs, is_tra_val_te)
    print(dataset)


# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# Backend tkagg is interactive backend. Turning interactive mode on.
# collected 1 item

# test_narxdataset.py Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 548.94it/s]
# <torchhydro.datasets.narxdataset.NarxDataset object at 0x7fd4cc0efcb0>