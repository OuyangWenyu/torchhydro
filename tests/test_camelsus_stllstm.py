"""test stldataset"""

import os
import sys
import pytest
cur_path = os.path.abspath(os.path.dirname(__file__))

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
def arg_camelsus_sltLstm(
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
    project_name = os.path.join("test_camels", "stldataset_camelsus")
    # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period = ["1980-10-01", "2012-09-30"]
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
        model_name="sLSTM",
        model_hyperparam={
            "input_size": 3,  # len(var_c) + len(var_t),  # 17 + 7 = 24  trend, season, residuals
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
        forecast_length=30,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        train_epoch=10,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 10,
        },
        # # the gage_id.txt file is set by the user, it must be the format like:
        # # GAUGE_ID
        # # 01013500
        # # 01022500
        # # ......
        # # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        # # gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_us\\gage_id.txt",
        # gage_id_file="/mnt/d/minio/waterism/datasets-origin/camels/camels_us/gage_id.txt",
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

def test_camels_sltLstm(arg_camelsus_sltLstm):
    config_data = default_config_file()
    update_cfg(config_data, arg_camelsus_sltLstm)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# Epoch 10 Loss 2.8410 time 42.85 lr 1.0
# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 10 Valid Loss 2.6992 Valid Metric {'NSE of trend': [-10.742765426635742, -280.1724548339844], 'RMSE of trend': [0.8123769760131836, 0.699028491973877], 
# 'R2 of trend': [-10.742765426635742, -280.1724548339844], 'KGE of trend': [-0.7173322639514923, -0.7621903570331725], 
# 'FHV of trend': [-107.58654022216797, -89.39804077148438], 'FLV of trend': [-119.32977294921875, -90.78811645507812], 
# 'NSE of season': [-0.0001901388168334961, 5.4001808166503906e-05], 'RMSE of season': [1.3842213153839111, 0.3153139352798462], 
# 'R2 of season': [-0.0001901388168334961, 5.4001808166503906e-05], 'KGE of season': [-0.633294196847848, -0.40856529671440267], 
# 'FHV of season': [-99.96641540527344, -99.80545043945312], 'FLV of season': [-100.18541717529297, -100.53850555419922], 
# 'NSE of residuals': [-0.005127072334289551, -0.010450601577758789], 'RMSE of residuals': [1.132514476776123, 0.6893935203552246], 
# 'R2 of residuals': [-0.005127072334289551, -0.010450601577758789], 'KGE of residuals': [-0.7433606238392234, -0.8154845091379512], 
# 'FHV of residuals': [-100.05054473876953, -100.07250213623047], 'FLV of residuals': [-99.81097412109375, -99.59794616699219]}
# Weights sucessfully loaded
# All processes are finished!