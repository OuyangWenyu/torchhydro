"""
Author: Lili Yu
Date: 2025-03-10 18:00:00
LastEditTime: 2025-03-10 18:00:00
LastEditors: Lili Yu
Description:
"""

import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

@pytest.fixture
def var_c():
    return [
        "top_altitude_mean",
        "top_slo_mean",
        "sta_area_snap",
        "top_drainage_density",
        "clc_2018_lvl1_1",
        "clc_2018_lvl2_11",
        "clc_2018_lvl3_111",
        "clc_1990_lvl1_1",
        "clc_2018_lvl1_2",
        "top_slo_ori_n",
        "top_slo_ori_ne",
        "top_slo_ori_e",
        "top_slo_flat",
        "top_slo_gentle",
        "top_slo_moderate",
        "top_slo_ori_se",
        "geo_py",
        "geo_pa",
    ]

@pytest.fixture
def var_t():
    return [
        "tsd_prec",
        "tsd_pet_ou",
        "tsd_prec_solid_frac",
        "tsd_temp",
        "tsd_pet_pe",
        "tsd_pet_pm",
        "tsd_wind",
        "tsd_humid",
        "tsd_rad_dli",
        "tsd_rad_ssi",
        "tsd_swi_gr",
        "tsd_swi_isba",
        "tsd_swe_isba",
        "tsd_temp_min",
        "tsd_temp_max",
    ]

@pytest.fixture
def camelsfrlstm_arg(var_c, var_t):
    project_name = os.path.join("test_camels", "lstm_camelsfr")
    # camels-fr time_range: ["1970-01-01", "2022-01-01"]
    train_period = ["2017-10-01", "2018-10-01"]
    valid_period = ["2018-10-01", "2019-10-01"]
    test_period = ["2019-10-01", "2020-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_fr",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_fr"
            ),
        },
        ctx=[-1],
        # model_name="KuaiLSTM",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
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
        forecast_length=366,
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
        # the gage_id.txt file is set by the user, it must be the format like:
        # GAUGE_ID
        # 01013500
        # 01022500
        # ......
        # Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
        # gage_id_file="D:\\minio\\waterism\\datasets-origin\\camels\\camels_fr\\gage_id.txt",
        gage_id=[
            "A105003001",
            "A107020001",
            "A112020001",
            "A116003002",
            "A140202001",
            "A202030001",
            "A204010101",
            "A211030001",
            "A212020002",
            "A231020001",
            "A234021001",
            "A251020001",
            "A270011001",
            "A273011002",
            "A284020001",
            "A330010001",
            "A361011001",
            "A369011001",
            "A373020001",
            "A380020001",
        ],
        which_first_tensor="sequence",
    )


def test_camelsfrlstm(camelsfrlstm_arg):
    config_data = default_config_file()
    update_cfg(config_data, camelsfrlstm_arg)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# scaler="SlidingWindowScaler",
# test_camelsfr_lstm.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Backend tkagg is interactive backend. Turning interactive mode on.
# Finish Normalization
#   0%|          | 0/20 [00:00<?, ?it/s]
# 100%|██████████| 20/20 [00:00<00:00, 39199.10it/s]
# Finish Normalization
#   0%|          | 0/20 [00:00<?, ?it/s]
# 100%|██████████| 20/20 [00:00<00:00, 125390.25it/s]
# Finish Normalization
#   0%|          | 0/20 [00:00<?, ?it/s]
# 100%|██████████| 20/20 [00:00<00:00, 366314.76it/s]
# Torch is using cpu
# I0527 20:32:23.448000 3538 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmppqp1yt4n
# I0527 20:32:23.452000 3538 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmppqp1yt4n/_remote_module_non_scriptable.py
# using 0 workers
# Epoch 1 Loss 0.2402 time 33.28 lr 1.0
# CpuLstmModel(
#   (linearIn): Linear(in_features=33, out_features=256, bias=True)
#   (lstm): LstmCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 1 Valid Loss 0.2125 Valid Metric {'NSE of streamflow': [0.40647685527801514, 0.5301699042320251, nan, 0.4963776469230652, 0.6317888498306274, 0.7251237630844116, nan, 0.5703470706939697, nan, 0.5114293098449707, 0.38894951343536377, 0.7038929462432861, nan, 0.557826042175293, 0.551986813545227, 0.6973429918289185, nan, 0.8070871829986572, 0.590926468372345, 0.7261139154434204], 
#                                         'RMSE of streamflow': [0.28977057337760925, 0.13852745294570923, 0.0, 0.32678207755088806, 3.8409054279327393, 1.4352887868881226, 0.0, 1.4058783054351807, 0.0, 1.021064281463623, 1.1793783903121948, 0.7365151047706604, 0.0, 1.5468082427978516, 0.18230651319026947, 0.14316190779209137, 0.0, 0.08715890347957611, 0.2016409933567047, 0.09689502418041229], 
#                                         'R2 of streamflow': [0.40647685527801514, 0.5301699042320251, nan, 0.4963776469230652, 0.6317888498306274, 0.7251237630844116, nan, 0.5703470706939697, nan, 0.5114293098449707, 0.38894951343536377, 0.7038929462432861, nan, 0.557826042175293, 0.551986813545227, 0.6973429918289185, nan, 0.8070871829986572, 0.590926468372345, 0.7261139154434204], 
#                                         'KGE of streamflow': [0.6044622941960367, 0.6388325456261306, nan, 0.6067723596160504, 0.6919491200168034, 0.7856035255576851, nan, 0.6403588830210891, nan, 0.4908946082742972, 0.4387319141045407, 0.7550205597853996, nan, 0.6076044590469311, 0.6921840799163079, 0.8343126851871244, nan, 0.8785923312751138, 0.6578907850740079, 0.861227460913061], 
#                                         'FHV of streamflow': [-27.738792419433594, -27.173254013061523, nan, -3.6785547733306885, -21.187942504882812, -18.698938369750977, nan, -22.1492919921875, nan, -11.472676277160645, -12.162785530090332, -13.660778045654297, nan, -17.855833053588867, -16.73796844482422, -14.159321784973145, nan, -13.381054878234863, -22.96544075012207, -17.603954315185547], 
#                                         'FLV of streamflow': [30.167705535888672, 53.878395080566406, nan, 14.294857025146484, 4.691139221191406, 5.855714321136475, nan, 15.349573135375977, nan, 22.521587371826172, 8.973915100097656, 4.743655204772949, nan, 3.619293212890625, 5.072465419769287, 3.823078155517578, nan, 2.5084636211395264, 16.336227416992188, 2.6132969856262207]}
# Epoch 2 Loss 0.2123 time 25.77 lr 1.0
# CpuLstmModel(
#   (linearIn): Linear(in_features=33, out_features=256, bias=True)
#   (lstm): LstmCellTied()
#   (linearOut): Linear(in_features=256, out_features=1, bias=True)
# )
# Epoch 2 Valid Loss 0.2438 Valid Metric {'NSE of streamflow': [0.5298289656639099, 0.5728253126144409, nan, 0.6508733034133911, 0.7204433679580688, 0.7455662488937378, nan, 0.7063430547714233, nan, 0.7311277389526367, 0.700262188911438, 0.7378284931182861, nan, 0.727491021156311, 0.7162236571311951, 0.7614421844482422, nan, 0.7292697429656982, 0.6133453845977783, 0.7363171577453613], 
#                                         'RMSE of streamflow': [0.2579071521759033, 0.13208946585655212, 0.0, 0.2720803916454315, 3.3467252254486084, 1.3808869123458862, 0.0, 1.1622751951217651, 0.0, 0.757464587688446, 0.8260101675987244, 0.6930269002914429, 0.0, 1.2143120765686035, 0.14509247243404388, 0.12710098922252655, 0.0, 0.10325226187705994, 0.19603776931762695, 0.09507305175065994], 
#                                         'R2 of streamflow': [0.5298289656639099, 0.5728253126144409, nan, 0.6508733034133911, 0.7204433679580688, 0.7455662488937378, nan, 0.7063430547714233, nan, 0.7311277389526367, 0.700262188911438, 0.7378284931182861, nan, 0.727491021156311, 0.7162236571311951, 0.7614421844482422, nan, 0.7292697429656982, 0.6133453845977783, 0.7363171577453613], 
#                                         'KGE of streamflow': [0.5265994443501647, 0.5159672860069769, nan, 0.5650893396212933, 0.5746520693668156, 0.6049153493201198, nan, 0.5833699347222929, nan, 0.5488127084438061, 0.6146645581103604, 0.5742246209639226, nan, 0.5885436975887635, 0.6866279830501969, 0.7308380732696117, nan, 0.6803893329262489, 0.4535589346812069, 0.7356495977632999], 
#                                         'FHV of streamflow': [-45.85350036621094, -46.438777923583984, nan, -33.670047760009766, -42.34329605102539, -36.969993591308594, nan, -43.7490348815918, nan, -41.353515625, -37.395328521728516, -39.6359977722168, nan, -41.12302780151367, -37.52633285522461, -31.173425674438477, nan, -31.14383316040039, -50.05552291870117, -32.613887786865234], 
#                                         'FLV of streamflow': [-39.42356872558594, -38.75595474243164, nan, -28.706724166870117, -31.191238403320312, -16.46652603149414, nan, -17.642074584960938, nan, -91.10343170166016, -41.17237854003906, -12.794230461120605, nan, -17.9259033203125, -14.039051055908203, -5.375644207000732, nan, -9.718718528747559, -37.84851837158203, -3.530280590057373]}
# Weights sucessfully loaded
# All processes are finished!
# metric_streamflow.csv
# basin_id,NSE,RMSE,R2,KGE,FHV,FLV
# A105003001,0.6111407279968262,0.41595694422721863,0.6111407279968262,0.5164932894311012,-50.74959945678711,-34.7824592590332
# A107020001,0.6722093820571899,0.3297474682331085,0.6722093820571899,0.5447558576242761,-49.40348815917969,-22.409873962402344
# A112020001,,0.0,,,,
# A116003002,0.6581969261169434,0.43973851203918457,0.6581969261169434,0.6126185928525392,-35.30162048339844,-29.699954986572266
# A140202001,0.7505388259887695,3.6500887870788574,0.7505388259887695,0.8298681940200481,-17.505146026611328,-55.57586669921875
# A202030001,0.7344731688499451,1.786149263381958,0.7344731688499451,0.5121360908150139,-39.257347106933594,-23.659372329711914
# A204010101,,0.0,,,,
# A211030001,0.6535025238990784,1.3996028900146484,0.6535025238990784,0.44759784899086585,-40.47301483154297,-18.878738403320312
# A212020002,,0.0,,,,
# A231020001,0.7517532110214233,0.8345324993133545,0.7517532110214233,0.5446499002786311,-46.096702575683594,-63.0106086730957
# A234021001,0.6875720024108887,0.9683476686477661,0.6875720024108887,0.5240000646590797,-39.066619873046875,-25.899795532226562
# A251020001,0.7823805212974548,0.7112948894500732,0.7823805212974548,0.5549180903723014,-40.34050369262695,-14.028251647949219
# A270011001,,0.0,,,,
# A273011002,0.7630099058151245,1.272855520248413,0.7630099058151245,0.5800485407014455,-37.147186279296875,-16.110042572021484
# A284020001,0.7132047414779663,0.2509232461452484,0.7132047414779663,0.5875045759356465,-45.68743133544922,-15.143413543701172
# A330010001,0.7835125923156738,0.2622676193714142,0.7835125923156738,0.8077835887697394,-14.567475318908691,-10.469076156616211
# A361011001,,0.0,,,,
# A369011001,0.8165388107299805,0.20164461433887482,0.8165388107299805,0.6552869235451396,-34.48461151123047,-13.969808578491211
# A373020001,0.627342700958252,0.45634058117866516,0.627342700958252,0.4540087786606639,-46.18911361694336,-103.25392150878906
# A380020001,0.8580394983291626,0.11110107600688934,0.8580394983291626,0.7773521012536128,-18.268245697021484,-5.147747993469238