import os
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro import SETTING
from torchhydro.trainers.trainer import train_and_evaluate


def run_camelsdpltank(
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
        sub=os.path.join("test_camels", "expdpllstmtank"),
        # sub=os.path.join("test_camels", "expdplanntank"),
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        model_name="DplLstmTank",
        # model_name="DplAnnTank",  #
        model_hyperparam={
            "n_input_features": 25,
            "n_output_features": 20,
            "n_hidden_states": 256,
            "warmup_length": 10,
        },
        loss_func="RMSESum",
        dataset="DplDataset",
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
        batch_size=20,
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
    # with torch.autograd.set_detect_anomaly(True):
    train_and_evaluate(config)
    print("All processes are finished!")


run_camelsdpltank(  # camels-us time_range: ["1980-01-01", "2014-12-31"]
    train_period=["1985-07-01", "1986-07-01"],
    valid_period=["1986-10-01", "1987-10-01"],
    test_period=["1987-10-01", "1988-10-01"],
)

# Epoch 10 Loss 2.5188 time 33.88 lr 1.0
# DplLstmTank(
#   (dl_model): SimpleLSTM(
#     (linearIn): Linear(in_features=25, out_features=256, bias=True)
#     (lstm): LSTM(256, 256)
#     (linearOut): Linear(in_features=256, out_features=20, bias=True)
#   )
#   (pb_model): Tank4Dpl()
# )
# Epoch 10 Valid Loss 32.2348 Valid Metric {'NSE of streamflow': [-97.11750030517578, -117.6788101196289, -71.01522064208984, -29.881248474121094, -19.860435485839844, -88.31390380859375, -25.386953353881836, -14.873613357543945, -23.94521713256836, -124.90320587158203, -78.0575942993164, -199.13186645507812, -276.44500732421875, -416.5133972167969, -21.97373390197754, -76.4012451171875, -76.1880111694336, -67.93043518066406, -86.1152572631836, -48.3878059387207], 'RMSE of streamflow': [17.123985290527344, 27.788585662841797, 20.484678268432617, 34.92772674560547, 28.479671478271484, 31.74786376953125, 36.54142761230469, 24.227943420410156, 27.636661529541016, 43.17718505859375, 29.615955352783203, 50.03243637084961, 51.27535629272461, 57.022247314453125, 20.300762176513672, 31.760982513427734, 14.058972358703613, 14.761900901794434, 17.002914428710938, 17.90805435180664], 'R2 of streamflow': [-97.11750030517578, -117.6788101196289, -71.01522064208984, -29.881248474121094, -19.860435485839844, -88.31390380859375, -25.386953353881836, -14.873613357543945, -23.94521713256836, -124.90320587158203, -78.0575942993164, -199.13186645507812, -276.44500732421875, -416.5133972167969, -21.97373390197754, -76.4012451171875, -76.1880111694336, -67.93043518066406, -86.1152572631836, -48.3878059387207], 'KGE of streamflow': [-11.166320549114708, -12.986876759659163, -13.60675979704919, -14.941021858600466, -8.648069079923296, -13.518775820132117, -11.37312401603039, -9.202226186298708, -12.10749924107997, -17.597470973017316, -14.161477023080844, -17.896781201167038, -21.69164379494291, -25.351659257379687, -8.601558169353371, -13.149108644078284, -8.883095819100843, -7.086369398364392, -10.408330675717893, -8.242183093102586], 'FHV of streamflow': [253.71661376953125, 243.29052734375, 175.1735382080078, 93.93026733398438, 56.612693786621094, 195.67213439941406, 58.48682403564453, 36.87171173095703, 47.313812255859375, 209.16827392578125, 150.16830444335938, 237.058837890625, 290.2238464355469, 380.9022521972656, 53.13356018066406, 137.38168334960938, 243.94517517089844, 216.99624633789062, 247.34556579589844, 145.09523010253906], 'FLV of streamflow': [1315.555419921875, 1338.348876953125, 2168.139892578125, 1803.7431640625, 1126.8121337890625, 1540.4017333984375, 1690.35107421875, 1008.3289184570312, 1776.6195068359375, 3767.001708984375, 1480.3787841796875, 8758.96484375, 5823.8974609375, 5591.7783203125, 934.5150756835938, 1189.8199462890625, 377.2731018066406, 259.4939270019531, 448.4931335449219, 460.0173034667969]}
# /home/yulili/code/torchhydro/torchhydro/trainers/deep_hydro.py:201: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   checkpoint = torch.load(weight_path, map_location=self.device)
# Weights sucessfully loaded
# All processes are finished!