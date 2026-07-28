"""
Unified DPL-MC training/inference with dynamic reservoir coupling.

This script only keeps the configuration (``BASIN_CONFIGS`` + ``Config``) and
dispatches to the model / training / inference logic that now lives inside the
torchhydro package:

* model side  -> ``torchhydro.models.dplmc_rsvr``
* driver side -> ``torchhydro.trainers.dplmc_rsvr_trainer`` (``train`` / ``run_inference``)

Data is read (unchanged, NetCDF) from the DPL-MC data root configured in
``hydro_setting.yml`` under ``local_data_path.datasets-dplmc``.
"""

import os
import argparse

from torchhydro import SETTING
from torchhydro.trainers.dplmc_rsvr_trainer import train, run_inference, set_seed

# Root for all DPL-MC reservoir data/results (data/, results/, rsvr_test/), from hydro_setting.yml
GNN_ROOT = SETTING["local_data_path"]["datasets-dplmc"]


# ======================================================================================
# Per-basin configuration. Add a new basin by adding a new entry here.
# ======================================================================================
BASIN_CONFIGS = {
    'kuqu': {
        'DATA_DIR': os.path.join(GNN_ROOT, 'data', 'kuqu_hourly_0616'),
        'OUTPUT_DIR': os.path.join(GNN_ROOT, 'results', 'kuqu_hourly_new-DPLMC-RSVR'),
        'MODEL_PATH': 'model_best.pth',
        'RESULT_FILE': 'inference_results_obsdis.nc',
        'TRAIN_PERIOD': ('2021-06-01', '2023-12-31'),
        'TEST_PERIOD': ('2024-07-01', '2024-07-20'),
        'LR': 0.005,
        'USE_BOUNDARY': True,
        'BOUNDARY_FILE': os.path.join(GNN_ROOT, 'data', 'kuqu_hourly_0616', 'boundary_data.nc'),
        'BOUNDARY_STATIONS': ['sanxia_60613950', 'sanxia_60703800', 'sanxia_60803000',
                              'sanxia_60603300', 'sanxia_60407100', 'sanxia_13020000'],
        'NO_OBS_STATIONS': ['sanxia_00000001', 'sanxia_00000002',
                            'sanxia_00000003', 'sanxia_00000004'],
        'TARGET_STATIONS': ['sanxia_60703700', 'sanxia_60703780', 'sanxia_60703800',
                            'sanxia_60701200', 'sanxia_60715955'],
    },
    'jialingjiang': {
        'DATA_DIR': os.path.join(GNN_ROOT, 'data', 'jialingjiang_hourly'),
        'OUTPUT_DIR': os.path.join(GNN_ROOT, 'results', 'jialingjiang_hourly-DPLMC-RSVR'),
        'MODEL_PATH': 'model_best.pth',
        'RESULT_FILE': 'inference_results.nc',
        'TRAIN_PERIOD': ('2021-01-01', '2023-12-31'),
        'TEST_PERIOD': ('2024-05-01', '2024-11-30'),
        'LR': 0.002,
        'USE_BOUNDARY': False,
        'BOUNDARY_FILE': None,
        'BOUNDARY_STATIONS': [],
        'NO_OBS_STATIONS': [],
        'TARGET_STATIONS': [],
    },
    'wujiang': {
        'DATA_DIR': os.path.join(GNN_ROOT, 'data', 'wujiang_hourly'),
        'OUTPUT_DIR': os.path.join(GNN_ROOT, 'results', 'wujiang_hourly-DPLMC-RSVR'),
        'MODEL_PATH': 'model_best.pth',
        'RESULT_FILE': 'inference_results.nc',
        'TRAIN_PERIOD': ('2021-01-01', '2023-12-31'),
        'TEST_PERIOD': ('2024-05-01', '2024-11-30'),
        'LR': 0.002,
        'USE_BOUNDARY': False,
        'BOUNDARY_FILE': None,
        'BOUNDARY_STATIONS': [],
        'NO_OBS_STATIONS': [],
        'TARGET_STATIONS': [],
    },
    'dpl_mc_test': {
        'DATA_DIR': os.path.join(GNN_ROOT, 'data', 'dpl_mc_test'),
        'OUTPUT_DIR': os.path.join(GNN_ROOT, 'results', 'dpl_mc_test-DPLMC-RSVR'),
        'MODEL_PATH': 'model_best.pth',
        'RESULT_FILE': 'inference_results.nc',
        'TRAIN_PERIOD': ('2021-06-01', '2021-06-20'),
        'TEST_PERIOD': ('2024-07-01', '2024-07-20'),
        'LR': 0.005,
        'USE_BOUNDARY': False,
        'BOUNDARY_FILE': None,
        'BOUNDARY_STATIONS': [],
        'NO_OBS_STATIONS': [],
        'TARGET_STATIONS': [],
    },
}


class Config:
    # ---- Basin selection ----
    BASIN = 'kuqu'                    # key into BASIN_CONFIGS

    # ---- Experiment mode ----
    MODE = 'inference'               # 'train' or 'inference'
    SEED = 42
    DEVICE = 'cpu'                   # e.g. 'cuda:0', 'cuda:1', 'cpu'

    # ---- Filled from BASIN_CONFIGS at runtime ----
    DATA_DIR = None
    OUTPUT_DIR = None
    MODEL_PATH = 'model_best.pth'
    RESULT_FILE = 'inference_results.nc'
    METRICS_FILE = 'training_metrics.json'
    TRAIN_PERIOD = None
    TEST_PERIOD = None
    USE_BOUNDARY = False
    BOUNDARY_FILE = None
    BOUNDARY_STATIONS = []
    NO_OBS_STATIONS = []
    TARGET_STATIONS = []
    LR = 0.005

    # ---- Data settings ----
    STATIC_FEATURE_LIST = ['control:area', 'interval:area',
                           'interval:river_length', 'interval:river_slope']
    CONVERT_UNITS = True             # mm/h -> m3/s
    USE_DISCHARGE = True             # treat nodes with discharge records as reservoir nodes

    # ---- DPL router hyperparameters ----
    MODEL_TYPE = 'DPL-MC'
    WARMUP = 72                      # hours discarded before loss / before forecast starts
    FORECAST_HORIZON = 240           # 预见期 (hours)
    INFERENCE_STRIDE = 1             # 滚动预报步长 (hours)
    EPOCHS = 50
    LOSS_MODE = 'confluence'         # 'all', 'confluence', or 'specific'
    LOSS_FUNCTION = 'RMSE'

    # ==================================================================================
    # Dynamic reservoir coupling
    # ==================================================================================
    USE_RESERVOIR_COUPLING = False
    # When False, reservoir nodes fall back to the (static) observed-discharge injection.
    RESERVOIR_DATA_DIR = os.path.join(GNN_ROOT, 'rsvr_test', 'reservoir_v3')
    RESERVOIR_MODEL_DIR = os.path.join(
        GNN_ROOT, 'rsvr_test', 'lstm_discharge_output', 'reservoir_v3',
        '240h_forecast', 'seq2seq_residual_no_time', 'bs128_lr0p001')
    RESERVOIR_MODEL_FILE = 'lstm_best_model.pth'
    RESERVOIR_LOAD_PRETRAINED = True   # load ed_codex checkpoint weights
    RESERVOIR_TRAINABLE = False        # False: frozen (no param updates); True: joint end-to-end

    # Must match the pretrained ed_codex checkpoint.
    RESERVOIR_INPUT_LEN = 240          # history window fed to the encoder
    RESERVOIR_OUTPUT_LEN = 240         # decode horizon before re-encoding (streaming)
    RESERVOIR_HIDDEN_DIM = 64
    RESERVOIR_NUM_LAYERS = 2
    RESERVOIR_DROPOUT = 0.2
    RESERVOIR_DENSE_DIM = 64
    RESERVOIR_FEATURE_LIST = ['discharge', 'streamflow', 'precipitation_obs',
                              'temperature_2m', 'reservoir_storage', 'water_level']
    RESERVOIR_PAST_FEATURES = ['discharge', 'streamflow', 'precipitation_obs',
                               'temperature_2m', 'reservoir_storage', 'water_level']
    RESERVOIR_FUTURE_FEATURES = ['streamflow', 'precipitation_obs', 'temperature_2m']
    RESERVOIR_STATIC_FEATURES = ['area', 'dead_water_level', 'normal_water_level',
                                 'total_reservoir_capacity', 'beneficial_reservoir_capacity']

    @classmethod
    def resolve(cls):
        """Copy the selected basin's settings onto the Config class attributes."""
        if cls.BASIN not in BASIN_CONFIGS:
            raise ValueError(f"Unknown BASIN '{cls.BASIN}'. Options: {list(BASIN_CONFIGS)}")
        for k, v in BASIN_CONFIGS[cls.BASIN].items():
            setattr(cls, k, v)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basin', type=str, default=None, help='basin key in BASIN_CONFIGS')
    parser.add_argument('--mode', type=str, default=None, help='train or inference')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no-coupling', action='store_true', help='disable reservoir coupling')
    parser.add_argument('--reservoir-trainable', action='store_true',
                        help='train reservoir jointly (end-to-end) instead of frozen')
    args = parser.parse_args()

    if args.basin:
        Config.BASIN = args.basin
    if args.mode:
        Config.MODE = args.mode.lower().strip()
    if args.device:
        Config.DEVICE = args.device
    if args.no_coupling:
        Config.USE_RESERVOIR_COUPLING = False
    if args.reservoir_trainable:
        Config.RESERVOIR_TRAINABLE = True

    cfg = Config.resolve()
    set_seed(cfg.SEED)
    print(f"Basin={cfg.BASIN} | Mode={cfg.MODE} | Device={cfg.DEVICE} | "
          f"ReservoirCoupling={cfg.USE_RESERVOIR_COUPLING}")

    if cfg.MODE == 'train':
        train(cfg)
    elif cfg.MODE == 'inference':
        run_inference(cfg)
    else:
        print(f"Unknown mode: {cfg.MODE}")
