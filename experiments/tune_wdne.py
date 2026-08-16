"""WDNE超参数调试 — 修复边方向后重新调参"""
import os, logging, warnings
logging.basicConfig(level=logging.WARNING); warnings.filterwarnings('ignore')
from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

BASINS = ['songliao_20800900','songliao_20810200','songliao_21100150',
          'songliao_21110150','songliao_21401050','songliao_21401550']

base_station = {
    'station_cols':['DRP'],'station_rm_nan':True,'station_time_units':['3h'],
    'use_adjacency':True,'adjacency_src_col':'ID','adjacency_dst_col':'NEXTDOWNID',
    'adjacency_edge_attr_cols':['dist_hdn','elev_diff','strm_slope'],
    'adjacency_weight_col':None,'use_basin_features':False,
    'edge_orientation':'downstream','station_scaler_type':'DapengScaler',
}

configs = [
    # (name, route_steps, route_decay, dropout, hidden_channels, struct_dim)
    ("steps3", 3, 0.5, 0.3, 32, 16),
    ("steps4", 4, 0.5, 0.3, 32, 16),
    ("steps5", 5, 0.4, 0.3, 32, 16),
    ("steps5_h48", 5, 0.4, 0.3, 48, 24),
]

for name, steps, decay, dr, hc, sd in configs:
    print(f"\n{'='*60}\nTuning: {name} (steps={steps}, decay={decay}, dr={dr}, hc={hc}, sd={sd})")

    config_data = default_config_file()
    args = cmd(sub=f'tune/{name}',
        source_cfgs={'source_name':'stationhydrodataset','source_path':SETTING['local_data_path']['datasets-interim'],'time_unit':['3h']},
        ctx=[0], model_name='WDNEAdaptiveRoutingNet',
        model_hyperparam={
            'in_channels':1, 'hidden_channels':hc, 'output_size':1,
            'seq_len':84, 'output_time':12,
            'model_type':'wdne_adaptive', 'num_static_features':0,
            'structure_embedding_dim':sd, 'route_steps':steps,
            'route_decay':decay, 'graph_output_mode':'residual',
            'root_output_mode':'root', 'dropout':dr,
            'structure_fusion':'gated', 'topology_encoder_type':'mlp',
            'local_lstm_layers':1,
        },
        gage_id=BASINS, batch_size=2, warmup_length=72, hindcast_length=0, forecast_length=12,
        min_time_unit='h', min_time_interval=3,
        var_t=['rain'], var_c=['None'], var_out=['inflow','flood_event'],
        dataset='GNNDataset', scaler='DapengScaler', train_mode=True,
        train_epoch=30, save_epoch=1,
        train_period=['1980-01-01-02','2014-12-31-23'],
        valid_period=['2015-01-01-02','2018-12-31-23'],
        test_period=['2019-01-01-02','2024-12-31-23'],
        station_cfgs=base_station, which_first_tensor='batch',
        opt='Adam', lr_scheduler={'lr':0.001,'lr_factor':0.5,'lr_patience':6},
        loss_func='FloodLoss',
        loss_param={'loss_func':'MSELoss','flood_weight':2,'flood_strategy':'weight','device':[0]},
        model_loader={'load_way':'best'},
        evaluator={'eval_way':'floodevent','stride':1}, rolling=-1,
        early_stopping=True, patience=10, calc_metrics=True,
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)

    # 读结果
    import pandas as pd
    try:
        df = pd.read_csv(f'/home/lizilin/code/torchhydro/results/tune/{name}/metric_inflow.csv')
        nses = df['NSE'].tolist()
        basins = [b.replace('songliao_','') for b in df['basin_id'].tolist()]
        print(f'  Avg NSE={np.mean(nses):.3f}, basins: {dict(zip(basins,[round(n,2) for n in nses]))}')
    except: pass

import numpy as np
print("\nDone!")
