
import os
import sys
import pytest

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

@pytest.fixture
def var_c():
    return [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "frac_forest",
        "lai_max",
        "lai_diff",
        "dom_land_cover_frac",
        "dom_land_cover",
        "root_depth_50",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "geol_1st_class",
        "geol_2nd_class",
        "geol_porostiy",
        "geol_permeability",
    ]

@pytest.fixture
def var_t():
    return [
        # NOTE: prcp must be the first variable
        "prcp",
        "dayl",
        "srad",
        "swe",
        "tmax",
        "tmin",
        "vp",
    ]


@pytest.fixture
def arg_camelsus_lstm(
    var_c,
    var_t,
):
    project_name = os.path.join("test_camels", "lstm_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1985-10-01", "1995-10-01"]
    valid_period = ["1995-10-01", "2000-10-01"]
    test_period = ["2000-10-01", "2010-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),  # 17 + 7 = 24
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="StreamflowDataset",
        # scaler="DapengScaler",
        scaler="SlidingWindowScaler",
        scaler_params={
        #     "prcp_norm_cols": [
        #         "streamflow",
        #     ],
        #     "gamma_norm_cols": [
        #         "prcp",
        #         "pr",
        #         "total_precipitation",
        #         "potential_evaporation",
        #         "ET",
        #         "PET",
        #         "ET_sum",
        #         "ssm",
        #     ],
            "pbm_norm": False,
            "sw_width": 30,
        },
        batch_size=2,
        forecast_history=0,
        forecast_length=365,
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
        which_first_tensor="sequence",
    )

def test_camelsus_lstm(arg_camelsus_lstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_lstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# scaler="SlidingWindowScaler",
# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# collected 1 item

# test_camelsus_lstm.py Backend tkagg is interactive backend. Turning interactive mode on.
# update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 127.94it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3292.23it/s]
# Finish Normalization


#   0%|          | 0/2 [00:00<?, ?it/s]
# 100%|██████████| 2/2 [00:00<00:00, 3067.13it/s]
# Torch is using cpu
# I0527 19:21:31.061000 27658 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpm8qk8sff
# I0527 19:21:31.065000 27658 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpm8qk8sff/_remote_module_non_scriptable.py
# using 0 workers

# Epoch 1 Loss 0.3164 time 26.77 lr 1.0
# CpuLstmModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (lstm): LstmCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.3284 Valid Metric {'NSE of streamflow': [0.7822074294090271, 0.33345848321914673], 
#                                         'RMSE of streamflow': [0.96100914478302, 1.961666464805603], 
#                                         'R2 of streamflow': [0.7822074294090271, 0.33345848321914673], 
#                                         'KGE of streamflow': [0.8475312852379371, 0.5463136067367054], 
#                                         'FHV of streamflow': [-18.784934997558594, 15.276570320129395], 
#                                         'FLV of streamflow': [16.961681365966797, 70.06387329101562]}
# Epoch 2 Loss 0.3077 time 26.41 lr 1.0
# CpuLstmModel(
#   (linearIn): Linear(in_features=24, out_features=256, bias=True)
#   (lstm): LstmCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.3011 Valid Metric {'NSE of streamflow': [0.8009583353996277, 0.5871822237968445], 
#                                         'RMSE of streamflow': [0.9187090396881104, 1.5437990427017212], 
#                                         'R2 of streamflow': [0.8009583353996277, 0.5871822237968445], 
#                                         'KGE of streamflow': [0.8045612774455126, 0.7252354148504488], 
#                                         'FHV of streamflow': [-24.86610221862793, -9.503461837768555], 
#                                         'FLV of streamflow': [7.042361736297607, 40.02037811279297]}
# Weights sucessfully loaded
# All processes are finished!
# metric_streamflow.csv
# basin_id,NSE,RMSE,R2,KGE,FHV,FLV
# 01013500,0.725595235824585,1.072420597076416,0.725595235824585,0.7469940439791914,-23.9079532623291,10.028379440307617
# 01022500,0.5565850138664246,1.8757661581039429,0.5565850138664246,0.6752055076523344,-32.160552978515625,28.089014053344727