import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate

VAR_C_CHOSEN_FROM_CAMELS_CH = [
    "elev_mean",
    "slope_mean",
    "area",
    "scrub_perc",  # note: this field in original data file is different with its in data description pdf file, choose the former for convience.
    "mixed_wood_perc",  # note: this field in original data file is different with its in data description pdf file, choose the former for convience.
    "rock_perc",
    "dom_land_cover",
    "crop_perc",
    "root_depth_50",
    "root_depth",
    "porosity",
    "conductivity",
    "tot_avail_water",
    "unconsol_sediments",
    "siliciclastic_sedimentary",
    "geo_porosity",
    "geo_log10_permeability",
]
VAR_T_CHOSEN_FROM_CH = [
    "precipitation",
    "ET",
    "waterlevel",
    "temperature_min",
    "temperature_mean",
    "temperature_max",
    "rel_sun_dur",
    "swe",
]

def run_camelschdplsac(
    train_period=None,
    valid_period=None,
    test_period=None
):
    """
    Use attr and forcing as input for dPL model

    Parameters
    ----------
    config

    Returns
    -------

    """
    if train_period is None:  # camels-ch time_range: ["1981-01-01", "2020-12-31"]
        train_period = ["2017-10-01", "2018-10-01"]
    if valid_period is None:
        valid_period = ["2018-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "dplsac_lstm_camelsch"),
        # sub=os.path.join("test_camels", "dplsac_ann_camelsch"),
        source_cfgs={
            "source_name": "camels_ch",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_ch"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": len(VAR_T_CHOSEN_FROM_CH)+len(VAR_C_CHOSEN_FROM_CAMELS_CH),  # 8 + 17 = 25
            "n_output_features": 21,
            "n_hidden_states": 256,
            "warmup_length": 10,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "precipitation",
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
        # gage_id_file = "D:\\minio\\waterism\\datasets-origin\\camels\\camels_ch\\gage_id.txt",
        gage_id_file="mnt/d:/minio/waterism/datasets-origin/camels/camels_ch/gage_id.txt",
        # gage_id=[
        #     "2009",
        #     "2011",
        #     "2016",
        #     "2018",
        #     "2019",
        #     "2020",
        #     "2024",
        #     "2029",
        #     # "2030",  #
        #     "2033",
        #     "2034",
        #     "2044",
        #     "2053",  #
        #     "2067",
        #     "2068",  #
        #     "2070",
        # ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=50,
        forecast_history=0,
        forecast_length=365,
        var_t=VAR_T_CHOSEN_FROM_CH,
        var_c=VAR_C_CHOSEN_FROM_CAMELS_CH,
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        train_epoch=1,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 1,
        },
        warmup_length=10,
        opt="Adadelta",
        rs=1234,
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")


run_camelschdplsac(  # camels-ch time_range: ["1981-01-01", "2020-12-31"]
    train_period=["2017-10-01", "2018-10-01"],
    valid_period=["2018-10-01", "2019-10-01"],
    test_period=["2019-10-01", "2020-10-01"],
)


# 229
# 17


# problems
#  lack of evaporation
#  lack of streamflow for part of stations/basins, nan values    key item
#  some fields of forcing data is absent in part of time range, nan values

# experiment
# swe  nan values before 1998.09.02
#  1981.01.01-1981.02.09 (10,30)
#   use swe, all of target values are null
#   commented swe, all of target values are not null
#   use swe and covered nan values with 0, all of target values are not null. while the target values get worse.
#  1998.08.01-1998.11.01 (40,52)
#   use swe, all of target values are null
#   use swe and covered nan values with 0, all of target values are not null. get bad target values.
#  1998.10.01-1999.10.01 (10,30)  no nan values
#   use swe, all of target values are not null
#   use swe and covered nan values with 0,  all of target values are not null. while the target values get worse.
#

# covering nan values with 0 will increase the time-step of time sequence, thus, target values get worse.
# lack of streamflow   download from somewhere, following us.  reject stations/basins lacking of streamflow directly.
# lack of evaporation   download from somewhere.  not a necessary item for nn model, while pb model need it.
#
