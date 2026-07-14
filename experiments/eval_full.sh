#!/bin/bash
# 全时段评估 — GPU空闲时运行
# 评估完成后运行: python plot_train_test_flood.py
echo "确保GPU空闲后运行此脚本"
python -c "
import os, shutil, logging, warnings
logging.basicConfig(level=logging.WARNING); warnings.filterwarnings('ignore')
from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

BASINS=['songliao_20800900','songliao_20810200','songliao_21100150','songliao_21110150','songliao_21401050','songliao_21401550']
FULL=['1980-01-01-02','2024-12-31-23']
sc={'station_cols':['DRP'],'station_rm_nan':True,'station_time_units':['3h'],'use_adjacency':True,'adjacency_src_col':'ID','adjacency_dst_col':'NEXTDOWNID','adjacency_edge_attr_cols':['dist_hdn','elev_diff','strm_slope'],'adjacency_weight_col':None,'use_basin_features':False,'edge_orientation':'downstream','station_scaler_type':'DapengScaler'}

models=[
    ('lstm_gnn','/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal/best_model.pth','/home/lizilin/code/torchhydro/results/baseline_lstm_gnn/songliao_3h_normal/dapengscaler_stat.json',{'in_channels':1,'hidden_channels':32,'num_hidden':2,'output_size':1,'seq_len':84,'output_time':12,'st_blocks':1,'temporal_kind':'gru','temporal_kernel_size':3,'temporal_layers':1,'spatial_kind':'gcn','aggregation':'mean','gnn_type':'gcn2','aggregate_to_graph':True,'use_gnn':True,'use_lstm':True,'n_heads':8,'transformer_layers':4,'dropout':0.3},'SpatioTemporalFloodNet3D'),
    ('lstm_only','/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal/best_model.pth','/home/lizilin/code/torchhydro/results/baseline_lstm_only/songliao_3h_normal/dapengscaler_stat.json',{'in_channels':1,'hidden_channels':32,'num_hidden':2,'output_size':1,'seq_len':84,'output_time':12,'st_blocks':1,'temporal_kind':'gru','temporal_kernel_size':3,'temporal_layers':1,'spatial_kind':'gcn','aggregation':'mean','gnn_type':'gcn2','aggregate_to_graph':True,'use_gnn':False,'use_lstm':True,'n_heads':8,'transformer_layers':4,'dropout':0.3},'SpatioTemporalFloodNet3D'),
    ('wdne','/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/best_model.pth','/home/lizilin/code/torchhydro/results/wdne_adaptive/songliao_3h_normal/dapengscaler_stat.json',{'in_channels':1,'hidden_channels':32,'output_size':1,'seq_len':84,'output_time':12,'model_type':'wdne_adaptive','num_static_features':0,'structure_embedding_dim':16,'route_steps':4,'route_decay':0.5,'graph_output_mode':'residual','root_output_mode':'root','dropout':0.3},'WDNEAdaptiveRoutingNet'),
]

for name, weight, stat, hyper, mn in models:
    print(f'=== {name} ===')
    out_dir=f'/home/lizilin/code/torchhydro/results/eval_full/{name}'
    os.makedirs(out_dir,exist_ok=True)
    shutil.copy(stat,os.path.join(out_dir,'dapengscaler_stat.json'))
    config_data=default_config_file()
    args=cmd(sub=f'eval/{name}',
        source_cfgs={'source_name':'stationhydrodataset','source_path':SETTING['local_data_path']['datasets-interim'],'time_unit':['3h']},
        ctx=[0],model_name=mn,model_hyperparam=hyper,
        gage_id=BASINS,batch_size=8,warmup_length=72,hindcast_length=0,forecast_length=12,
        min_time_unit='h',min_time_interval=3,var_t=['rain'],var_c=['None'],var_out=['inflow','flood_event'],
        dataset='GNNDataset',scaler='DapengScaler',train_mode=False,
        train_period=FULL,valid_period=FULL,test_period=FULL,
        station_cfgs=sc,which_first_tensor='batch',
        model_loader={'load_way':'pth','pth_path':weight},
        evaluator={'eval_way':'floodevent','stride':1},opt='Adam',lr_scheduler={'lr':0.001},loss_func='RMSESum',rolling=-1)
    update_cfg(config_data,args)
    train_and_evaluate(config_data)
    print(f'{name} DONE')
print('ALL DONE — now run: python plot_train_test_flood.py')
"
