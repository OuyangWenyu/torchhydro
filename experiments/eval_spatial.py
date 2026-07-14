"""
空间敏感性测试: 在 upstream/midstream/downstream 站点扩大降雨
加载已训练的 WDNE 模型, 分别用 normal/sim1/sim2/sim3 评估
"""
import os, shutil, logging, warnings
logging.basicConfig(level=logging.WARNING); warnings.filterwarnings('ignore')
from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

BASIN = 'songliao_21401550'
WDNE_WEIGHT = '/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/best_model.pth'
STAT_SRC = '/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/dapengscaler_stat.json'

for mode in ['normal', 'sim1', 'sim2', 'sim3']:
    print(f'\n=== spatial_{mode} ===')
    out_dir = f'/home/lizilin/code/torchhydro/results/spatial_test/{mode}'
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(STAT_SRC, os.path.join(out_dir, 'dapengscaler_stat.json'))

    config_data = default_config_file()
    args = cmd(sub=f'spatial_test/{mode}',
        source_cfgs={'source_name':'stationhydrodataset','source_path':SETTING['local_data_path']['datasets-interim'],'time_unit':['3h']},
        ctx=[0], model_name='WDNEAdaptiveRoutingNet',
        model_hyperparam={'in_channels':1,'hidden_channels':32,'output_size':1,'seq_len':84,'output_time':12,
            'model_type':'wdne_adaptive','num_static_features':0,'structure_embedding_dim':16,
            'route_steps':2,'route_decay':0.6,'graph_output_mode':'residual','root_output_mode':'root',
            'dropout':0.2,'structure_fusion':'gated','topology_encoder_type':'mlp','local_lstm_layers':1},
        gage_id=[BASIN], batch_size=2,
        warmup_length=72, hindcast_length=0, forecast_length=12,
        min_time_unit='h', min_time_interval=3,
        var_t=['rain'], var_c=['None'], var_out=['inflow','flood_event'],
        dataset='GNNDataset', scaler='DapengScaler',
        train_mode=False,
        train_period=['2019-01-01-02','2024-12-31-23'],
        valid_period=['2019-01-01-02','2024-12-31-23'],
        test_period=['2019-01-01-02','2024-12-31-23'],
        station_cfgs={'station_cols':['DRP'],'station_rm_nan':True,'station_time_units':['3h'],
            'use_adjacency':True,'adjacency_src_col':'ID','adjacency_dst_col':'NEXTDOWNID',
            'adjacency_edge_attr_cols':['dist_hdn','elev_diff','strm_slope'],
            'adjacency_weight_col':None,'use_basin_features':False,'edge_orientation':'downstream','station_scaler_type':'DapengScaler'},
        which_first_tensor='batch',
        model_loader={'load_way':'pth','pth_path':WDNE_WEIGHT},
        evaluator={'eval_way':'floodevent','stride':1},
        opt='Adam', lr_scheduler={'lr':0.001}, loss_func='RMSESum', rolling=-1,
    )
    update_cfg(config_data, args)
    # 设置 spatial_test_mode
    if 'training_cfgs' not in config_data: config_data['training_cfgs'] = {}
    config_data['training_cfgs']['spatial_test_mode'] = mode
    config_data['training_cfgs']['train_mode'] = False

    train_and_evaluate(config_data)
    print(f'  {mode} done')
