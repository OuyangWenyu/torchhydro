import os

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import pytest

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
        "streamflow",
    ]

@pytest.fixture
def camelsfrnarx_arg(var_c, var_t):
    project_name = os.path.join("test_camels", "narx_camelsfr")
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
            "n_hidden_states": 256,
            "input_delay": 1,
            "feedback_delay": 1,
            # "num_layers": 1,
            "close_loop": False,
        },
        loss_func="RMSESum",
        sampler="KuaiSampler",
        dataset="NarxDataset",
        scaler="DapengScaler",
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
            "A330010001",
        ],
        batch_size=1,
        forecast_history=0,
        forecast_length=1,
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


def test_camelsfrnarx(camelsfrnarx_args):
    config_data = default_config_file()
    update_cfg(config_data, camelsfrnarx_args)
    train_and_evaluate(config_data)
    print("All processes are finished!")
