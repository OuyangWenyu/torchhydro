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

# Epoch 2 Loss 2.7986 time 43.89 lr 1.0
# sLSTM(
#   (linearIn): Linear(in_features=3, out_features=256, bias=True)
#   (lstm): LSTM(256, 256, num_layers=10)
#   (linearOut): Linear(in_features=256, out_features=3, bias=True)
# )
# Epoch 2 Valid Loss 2.7040 Valid Metric {'NSE of trend': [-12.38329792022705, -236.13333129882812], 'RMSE of trend': [0.8672692775726318, 0.6419553160667419], 
# 'R2 of trend': [-12.38329792022705, -236.13333129882812], 'KGE of trend': [-0.9011639694661886, -0.5613267276293776], 
# 'FHV of trend': [-112.68045806884766, -82.27946472167969], 'FLV of trend': [-133.78668212890625, -83.89842987060547], 
# 'NSE of season': [-0.00021910667419433594, 3.457069396972656e-05], 'RMSE of season': [1.3842414617538452, 0.3153170049190521], 
# 'R2 of season': [-0.00021910667419433594, 3.457069396972656e-05], 'KGE of season': [-0.6694788902904354, -0.6035367504201312], 
# 'FHV of season': [-99.99250793457031, -99.9565658569336], 'FLV of season': [-100.03409576416016, -100.09902954101562], 
# 'NSE of residuals': [-0.004571080207824707, -0.008946776390075684], 'RMSE of residuals': [1.1322011947631836, 0.6888803243637085], 
# 'R2 of residuals': [-0.004571080207824707, -0.008946776390075684], 'KGE of residuals': [-0.7146822089099494, -0.6529984882004352], 
# 'FHV of residuals': [-99.88750457763672, -99.83871459960938], 'FLV of residuals': [-100.2201919555664, -100.46831512451172]}
# Weights sucessfully loaded
# All processes are finished!