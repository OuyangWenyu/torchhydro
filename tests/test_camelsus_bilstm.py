
import os
import pytest

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture
def var_c():
    return [
        # "elev_mean",
        # "slope_mean",
        # "area_gages2",
        # "frac_forest",
        # "lai_max",
        # "lai_diff",
        # "dom_land_cover_frac",
        # "dom_land_cover",
        # "root_depth_50",
        # "soil_depth_statsgo",
        # "soil_porosity",
        # "soil_conductivity",
        # "max_water_content",
        # "geol_1st_class",
        # "geol_2nd_class",
        # "geol_porostiy",
        # "geol_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        # NOTE: prcp must be the first variable
        "prcp",
        # "PET"
        # "dayl",
        # "srad",
        # "swe",
        # "tmax",
        # "tmin",
        # "vp",
    ]

@pytest.fixture
def arg_camelsus_biLstm(
    var_c,
    var_t,
):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    project_name = os.path.join("test_camels", "bilstm_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1981-10-01", "2012-09-30"]
    valid_period = ["2012-10-01", "2013-09-30"]
    test_period = ["2013-10-01", "2014-09-30"]

    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        # model_name="CpuLSTM",
        model_name="biLSTM",
        model_hyperparam={
            "input_size": 3,  # trend, season, residuals
            "output_size": 3,  # trend, season, residuals
            "hidden_size": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StlDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "potential_evaporation",
                "ET",
                "PET",
                "ET_sum",
                "ssm",
            ],
            "pbm_norm": True,
        },
        batch_size=2,
        forecast_history=0,
        forecast_length=5,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=2,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 2,
        },
        gage_id=[
            "01013500",
            "01022500",
            # # "01030500",
            # # "01031500",
            # # "01047000",
            # # "01052500",
            # # "01054200",
            # # "01055000",
            # # "01057000",
            # # "01073000",
            # # "01078000",
            # # "01118300",
            # # "01121000",
            # # "01123000",
            # # "01134500",
            # # "01137500",
            # # "01139000",
            # # "01139800",
            # # "01142500",
            # # "01144000",
            # "02092500",  # 02108000 -> 02092500
            # "02108000",
        ],
        b_decompose=True,
        which_first_tensor="sequence",
    )

def test_camelsus_biLstm(arg_camelsus_biLstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_biLstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")

# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_camelsus_bilstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 27.07it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 10305.42it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 18315.74it/s]
# Torch is using cpu
# I0522 15:21:07.267000 27141 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpq3ii5d78
# I0522 15:21:07.271000 27141 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpq3ii5d78/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 2.3139 time 965.41 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=3, bias=True)
# )
# Epoch 1 Valid Loss 2.7251 Valid Metric {'NSE of trend': [-12.647359848022461, -229.62474060058594], 'RMSE of trend': [0.878373920917511, 0.6349567174911499], 
# 'R2 of trend': [-12.647359848022461, -229.62474060058594], 'KGE of trend': [-0.8100809992014595, -0.7267826093028262], 
# 'FHV of trend': [-113.24185180664062, -81.14249420166016], 'FLV of trend': [-133.34332275390625, -83.63639831542969], 
# 'NSE of season': [-0.01040506362915039, -0.1497095823287964], 'RMSE of season': [1.3807753324508667, 0.3355520963668823], 
# 'R2 of season': [-0.01040506362915039, -0.1497095823287964], 'KGE of season': [-5.604892748868133, -43.33410522027678], 
# 'FHV of season': [-102.18177795410156, -112.67562866210938], 'FLV of season': [-87.43279266357422, -63.515438079833984], 
# 'NSE of residuals': [-0.029011964797973633, -0.07007730007171631], 'RMSE of residuals': [1.147045612335205, 0.7101570963859558], 
# 'R2 of residuals': [-0.029011964797973633, -0.07007730007171631], 'KGE of residuals': [-1.932237100620656, -2.2113554020398123], 
# 'FHV of residuals': [-102.90813446044922, -104.17162322998047], 'FLV of residuals': [-89.30455780029297, -77.36113739013672]}
# Epoch 2 Loss 2.2923 time 878.89 lr 1.0
# biLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10, bidirectional=True)
#   (linearOut): Linear(in_features=512, out_features=3, bias=True)
# )
# Epoch 2 Valid Loss 2.7167 Valid Metric {'NSE of trend': [-10.699970245361328, -281.1546630859375], 'RMSE of trend': [0.8132938146591187, 0.7023196220397949], 
# 'R2 of trend': [-10.699970245361328, -281.1546630859375], 'KGE of trend': [-0.7228441958486531, -0.7242270631235213], 
# 'FHV of trend': [-106.57161712646484, -90.57194519042969], 'FLV of trend': [-116.65676879882812, -91.79499816894531], 
# 'NSE of season': [-0.017010807991027832, -0.26401567459106445], 'RMSE of season': [1.3852814435958862, 0.35183748602867126], 
# 'R2 of season': [-0.017010807991027832, -0.26401567459106445], 'KGE of season': [-7.357128538063229, -57.878731060992756], 
# 'FHV of season': [-102.91275024414062, -116.95680236816406], 'FLV of season': [-83.22085571289062, -51.235260009765625], 
# 'NSE of residuals': [-0.017116904258728027, -0.03984987735748291], 'RMSE of residuals': [1.1403965950012207, 0.7000550031661987], 
# 'R2 of residuals': [-0.017116904258728027, -0.03984987735748291], 'KGE of residuals': [-1.4761835256212765, -1.57853845849325], 
# 'FHV of residuals': [-101.80068969726562, -102.59357452392578], 'FLV of residuals': [-93.3528060913086, -85.9005126953125]}
# Weights sucessfully loaded
# All processes are finished!