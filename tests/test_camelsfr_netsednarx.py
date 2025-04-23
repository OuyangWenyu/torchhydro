
import os
import pytest
import torch

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


@pytest.fixture
def var_c():
    return [
        # "top_altitude_mean",
        # "top_slo_mean",
        # "sta_area_snap",
        # "top_drainage_density",
        # "clc_2018_lvl1_1",
        # "clc_2018_lvl2_11",
        # "clc_2018_lvl3_111",
        # "clc_1990_lvl1_1",
        # "clc_2018_lvl1_2",
        # "top_slo_ori_n",
        # "top_slo_ori_ne",
        # "top_slo_ori_e",
        # "top_slo_flat",
        # "top_slo_gentle",
        # "top_slo_moderate",
        # "top_slo_ori_se",
        # "geo_py",
        # "geo_pa",
    ]

@pytest.fixture
def var_t():
    return [
        "tsd_prec",
        "tsd_pet_ou",
        # "tsd_prec_solid_frac",
        # "tsd_temp",
        # "tsd_pet_pe",
        # "tsd_pet_pm",
        # "tsd_wind",
        # "tsd_humid",
        # "tsd_rad_dli",
        # "tsd_rad_ssi",
        # "tsd_swi_gr",
        # "tsd_swi_isba",
        # "tsd_swe_isba",
        # "tsd_temp_min",
        # "tsd_temp_max",
        # "streamflow",
    ]

@pytest.fixture
def camelsfr_narx_arg(var_c, var_t):
    project_name = os.path.join("test_camels", "NestedNarx_camelsfr")
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
        model_name="NestedNarx",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 64,
            "input_delay": 2,
            "feedback_delay": 1,
            # "num_layers": 1,
            "close_loop": False,
            "nested_model": None,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="NarxDataset",
        # scaler="DapengScaler",  # todo: ValueError: operands could not be broadcast together with shapes (17,366) (3,366)  ../../../.conda/envs/torchhydro/lib/python3.13/site-packages/hydroutils/hydro_stat.py:554: ValueError def cal_stat_prcp_norm(x, meanprep): flowua = x / tempprep
        scaler="StandardScaler",
        # gage_id=[
        #     "A105003001",
        #     "A107020001",
        #     "A112020001",
        #     "A116003002",
        #     "A140202001",
        #     "A202030001",
        #     "A204010101",
        #     "A211030001",
        #     "A212020002",
        #     "A231020001",
        #     "A234021001",
        #     "A251020001",
        #     "A270011001",
        #     "A273011002",
        #     "A284020001",
        #     "A330010001",
        #     "A361011001",
        #     "A369011001",
        #     "A373020001",
        #     "A380020001",
        # ],
        gage_id = [
            "A550061001",
            "A369011001",
            "A284020001",
            "A330010001",
        ],
        batch_size=3,
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
        which_first_tensor="sequence",
        b_nestedness=True,
    )


def test_camelsfr_nestednarx(camelsfr_narx_arg):
    config_data = default_config_file()
    update_cfg(config_data, camelsfr_narx_arg)
    with torch.autograd.set_detect_anomaly(True):
        train_and_evaluate(config_data)
    print("All processes are finished!")

# ============================= test session starts ==============================
# platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
# rootdir: /home/yulili/code/torchhydro/tests
# configfile: ../setup.cfg
# plugins: mock-3.14.0
# Backend tkagg is interactive backend. Turning interactive mode on.
# collected 1 item

# test_camelsfr_netsednarx.py update config file
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the PRECIPITATION variable is in the 1st location in var_t setting!!---------
# If you have POTENTIAL_EVAPOTRANSPIRATION, please set it the 2nd!!!-
# !!!!!!NOTE!!!!!!!!
# -------Please make sure the STREAMFLOW variable is in the 1st location in var_out setting!!---------
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 1282.49it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 2798.48it/s]
# Finish Normalization


#   0%|          | 0/18 [00:00<?, ?it/s]
# 100%|██████████| 18/18 [00:00<00:00, 32388.45it/s]
# Torch is using cpu
# I0422 21:30:20.475000 24631 site-packages/torch/distributed/nn/jit/instantiator.py:22] Created a temporary directory at /tmp/tmpiqnlxmuz
# I0422 21:30:20.480000 24631 site-packages/torch/distributed/nn/jit/instantiator.py:73] Writing /tmp/tmpiqnlxmuz/_remote_module_non_scriptable.py
# using 0 workers

#   0%|          | 0/13 [00:00<?, ?it/s]
#   8%|▊         | 1/13 [00:44<08:57, 44.79s/it]
#   8%|▊         | 1/13 [03:05<37:07, 185.60s/it]
# F