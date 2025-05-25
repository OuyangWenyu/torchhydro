import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate


def run_camelsdplsac(
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
    if train_period is None:  # camels-us time_range: ["1980-01-01", "2014-12-31"]
        train_period = ["1985-10-01", "1995-10-01"]
    if valid_period is None:
        valid_period = ["1995-10-01", "2000-10-01"]
    if test_period is None:
        test_period = ["2000-10-01", "2010-10-01"]
    config = default_config_file()
    args = cmd(
        sub=os.path.join("test_camels", "expdpllstmsac"),
        # sub=os.path.join("test_camels", "expdplannsac"),
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        model_name="DplLstmSac",
        # model_name="DplAnnSac",
        model_hyperparam={
            "n_input_features": 25,
            "n_output_features": 21,
            "n_hidden_states": 256,
            "warmup_length": 10,
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        # scaler="DapengScaler",
        scaler="StandardScaler",
        # scaler_params={
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
        #     "pbm_norm": True,
        # },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01073000",
            "01078000",
            "01118300",
            "01121000",
            "01123000",
            "01134500",
            "01137500",
            "01139000",
            "01139800",
            "01142500",
            "01144000",
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=50,
        forecast_history=0,
        forecast_length=30,
        var_t=[
            "prcp",
            "PET",
            "dayl",
            "srad",
            "swe",
            "tmax",
            "tmin",
            "vp",
        ],
        var_out=["streamflow"],
        target_as_input=0,
        constant_only=0,
        train_epoch=10,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            "test_epoch": 10,
        },
        warmup_length=10,
        opt="Adadelta",
        which_first_tensor="sequence",
    )
    update_cfg(config, args)
    train_and_evaluate(config)
    print("All processes are finished!")


run_camelsdplsac(  # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period=["1985-10-01", "1986-10-01"],
    valid_period=["1986-10-01", "1987-10-01"],
    test_period=["1987-10-01", "1988-10-01"],
)

# Epoch 10 Valid Loss 335011.5938 Valid Metric {'NSE of streamflow': [-107660156928.0, -71665025024.0, -51016638464.0, -1223137.75, -693232.4375, -35935576064.0, 
# -7996686336.0, -398631.28125, -568764.3125, -1588726.25, -1744883.25, -2065819.75, -2755202.75, -3858982.5, -774977.25, -1411873.5, -4603554.0, -3558735.0, 
# -3631374.25, -53720981504.0], 'RMSE of streamflow': [567229.9375, 682862.8125, 545221.4375, 6951.21533203125, 5191.74169921875, 636820.8125, 636130.0625, 
# 3839.41357421875, 4173.09716796875, 4850.21044921875, 4399.84375, 5083.23095703125, 5109.7197265625, 5482.0830078125, 3728.561279296875, 4289.61181640625, 
# 3433.4072265625, 3354.166748046875, 3471.457275390625, 590623.1875], 'R2 of streamflow': [-107660156928.0, -71665025024.0, -51016638464.0, -1223137.75, -693232.4375, 
# -35935576064.0, -7996686336.0, -398631.28125, -568764.3125, -1588726.25, -1744883.25, -2065819.75, -2755202.75, -3858982.5, -774977.25, -1411873.5, -4603554.0, 
# -3558735.0, -3631374.25, -53720981504.0], 'KGE of streamflow': [-415657.0797930128, -352325.6062975008, -398535.6743891149, -2500.26021571342, -1496.8942700162945, 
# -297587.2131016022, -225430.6960879537, -1499.7396698776615, -1703.2494910977623, -1829.559886357929, -2067.4573752596966, -1793.9854565450926, -2074.871783938255, 
# -2330.638381366036, -1757.1572164385732, -1850.0990326163032, -2496.8849690648785, -1945.256816714617, -2349.943088551262, -327860.51847546163], 
# 'FHV of streamflow': [12318678.0, 8064047.5, 7125652.5, 97699.0546875, 70051.7734375, 6033131.0, 2898143.5, 47796.703125, 61671.6484375, 84118.8671875, 73432.6171875, 
# 87476.7890625, 107002.34375, 127906.9453125, 48548.5703125, 63809.38671875, 131492.609375, 114670.3359375, 121941.5546875, 7391991.5], 
# 'FLV of streamflow': [45199064.0, 33699460.0, 66540256.0, 426842.78125, 237176.703125, 35740376.0, 34641736.0, 205520.734375, 297757.96875, 283099.4375, 198386.96875, 
# 481406.875, 331953.875, 252673.0, 271178.96875, 211578.6875, 146783.78125, 110805.515625, 158286.859375, 28394940.0]}
# Weights sucessfully loaded
# All processes are finished!