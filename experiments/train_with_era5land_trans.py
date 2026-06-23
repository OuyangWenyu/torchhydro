import os
import glob


prechn_gage_id = [gage_id.split('/')[-1].split('.')[0] for gage_id in glob.glob('/ftproot/basins-interim/timeseries/1h/*.csv', recursive=True)]
camels_hourly_usgs = [file.split('/')[-1].split('-')[0] for file in glob.glob('/ftproot/camels_hourly/data/usgs_streamflow_csv/*.csv', recursive=True)]
pre_gage_ids = [gage_id for gage_id in prechn_gage_id if (gage_id.split('_')[-1] in camels_hourly_usgs) | ('songliao' in gage_id)]
# basin_stations.csv, L100-148, L150-156
remove_list = ['02018000','02027000','02027500','02028500','02038850','02046000','02051500','02053200','02053800',
               '02055100','02056900','02059500','02064000','02065500','02069700','02070000','02074500','02077200',
               '02081500','02082950','02092500','02096846','02102908','02108000','02111180','02111500','02118500',
               '02128000','02137727','02140991','02143000','02143040','02149000','02152100','02177000','02178400',
               '02193340','02196000','02198100','02202600','02212600','02215100','02216180','02221525','02231000',
               '02245500','02246000','02296500','02297155','02298123','02298608','02299950','02300700','02349900',
               '02350900','02361000']
pre_gage_ids = [gage_id for gage_id in pre_gage_ids if gage_id.split('_')[1] not in remove_list]
from torchhydro.configs.config import default_config_file, cmd, update_cfg


def create_config_Transformer():
    project_name = os.path.join("train_with_era5land_trans", "ex1")
    config_data = default_config_file()

    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": {
                "forcing": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "target": "basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow",
                "attributes": "basins-origin/attributes.nc",
            },
        },
        ctx=[1],
        model_name="Transformer",
        model_hyperparam={
            "n_encoder_inputs": 20,
            "n_decoder_inputs": 19,
            "n_decoder_output": 2,
            "channels": 256,
            "num_embeddings": 512,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "prec_window": 0,
        },
        model_loader={"load_way": "best"},
        gage_id=pre_gage_ids,
        batch_size=128,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "total_precipitation_hourly",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="TransformerDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        train_period=[("2015-05-01-01", "2022-12-20-01")],
        test_period=[("2023-01-01-01", "2023-11-20-01")],
        valid_period=[("2023-01-01-01", "2023-11-20-01")],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [1],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.96,
        },
        which_first_tensor="sequence",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        patience=10,
        model_type="MTL",
        network_shp='/home/wangyang1/songliao_cut_single.shp',
        node_shp='/home/jiaxuwu/463_nodes.shp'
    )
    update_cfg(config_data, args)
    return config_data
