import os

from torchhydro import CACHE_DIR
import xarray as xr
from torchhydro.configs.config import update_cfg
import pandas as pd
import torch
import matplotlib.pyplot as plt

def find_common_files_in_subdirs(main_dir, compare_dir):
    # 创建一个集合来存储找到的公共文件基本名称
    common_files = set()

    # 获取比较目录中的文件基本名称
    compare_files = {os.path.splitext(f)[0] for f in os.listdir(compare_dir)}

    # 遍历主目录及其所有子目录
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            # 获取文件的基本名称
            base_name = os.path.splitext(file)[0][:8]
            # 检查这个基本名称是否在比较目录的文件集合中
            if base_name in compare_files:
                # 如果是，添加到公共文件集合中
                common_files.add(base_name)

    return common_files

# # 定义主目录和比较目录的路径
# main_dir = r"C:\Users\jgchu\AppData\Local\hydro\Cache\gages\basin_mean_forcing\basin_mean_forcing\daymet"
# compare_dir = r"C:\Users\jgchu\AppData\Local\hydro\Cache\mopex"
#
# # 调用函数并打印结果
# common_file_names = find_common_files_in_subdirs(main_dir, compare_dir)
# # 将集合转换为 pandas DataFrame
# df = pd.DataFrame(list(common_file_names), columns=['Basin ID'])
#
# # 指定 Excel 文件路径
# excel_path =  r"C:\Users\jgchu\AppData\Local\hydro\Cache\mopex\commonBasin.xlsx"
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel(excel_path, index=False, engine='openpyxl')

# 读取训练数据
def test_train_data(fusion_lstm_args, config_data):
    update_cfg(config_data, fusion_lstm_args)
    target_cols = config_data["data_cfgs"]["target_cols"]
    sites_id = config_data["data_cfgs"]["object_ids"]
    time = config_data["data_cfgs"][f"t_range_train"]
    relevant_cols = config_data["data_cfgs"]["relevant_cols"]
    constant_cols = config_data["data_cfgs"]["constant_cols"]
    ts_gages = xr.open_dataset(CACHE_DIR.joinpath("gages_timeseries.nc"))
    ts_mopex = xr.open_dataset(CACHE_DIR.joinpath("mopex_timeseries.nc"))
    attr = xr.open_dataset(CACHE_DIR.joinpath("gages_attributes.nc"))
    target = ts_gages[target_cols].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_gages = ts_gages[relevant_cols[0]].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_mopex = ts_mopex[relevant_cols[1]].sel(basin=sites_id, time=slice(time[0], time[1]))
    forcing = ts_gages[relevant_cols[2:]].sel(basin=sites_id, time=slice(time[0], time[1]))
    constant = attr[constant_cols].sel(basin=sites_id)
    # null_data = target.isnull().sum(dim='time')
    # print(null_data)
    average_target_over_basin = target.mean(dim='time')
    average_p_gages_over_basin = p_gages.mean(dim='time')
    average_p_mopex_over_basin = p_mopex.mean(dim='time')
    average_forcing_over_basin = forcing.mean(dim='time')
    constant_over_basin = constant
    # average_target_over_basin_df = average_target_over_basin.to_dataframe()

    # average_forcing_over_basin_df = average_forcing_over_basin.to_dataframe()
    # average_constant_over_basin_df = average_constant_over_basin.to_dataframe()
    with pd.ExcelWriter('output_data.xlsx') as writer:
        df = average_target_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Target Average')
        # for var_name in average_target_over_basin.data_vars:
        #     df = average_target_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)
        df_gages = average_p_gages_over_basin.to_dataframe(name='Average P Gages')
        df_mopex = average_p_mopex_over_basin.to_dataframe(name='Average P Mopex')
        df_p_combined = pd.concat([df_gages, df_mopex], axis=1)
        df_p_combined.to_excel(writer, sheet_name='Combined P Data')
        df = average_forcing_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Forcing Average')
        df = constant_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Constant Data')
        # for var_name in average_forcing_over_basin.data_vars:
        #     df = average_forcing_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)

        # average_target_over_basin.to_excel(writer, sheet_name='Target Average')
        # average_p_gages_over_basin.to_excel(writer, sheet_name='Gages Precipitation Average')
        # average_p_mopex_over_basin.to_excel(writer, sheet_name='Mopex Precipitation Average')
        # average_forcing_over_basin.to_excel(writer, sheet_name='Forcing Average')
        # average_constant_over_basin.to_excel(writer, sheet_name='Constant Average')

# 读取nc径流数据
def test_nc_streamflow(fusion_lstm_args, config_data):
    update_cfg(config_data, fusion_lstm_args)
    target_cols = config_data["data_cfgs"]["target_cols"]
    sites_id = config_data["data_cfgs"]["object_ids"]
    time = config_data["data_cfgs"][f"t_range_train"]
    relevant_cols = config_data["data_cfgs"]["relevant_cols"]
    constant_cols = config_data["data_cfgs"]["constant_cols"]
    ts_gages = xr.open_dataset(CACHE_DIR.joinpath("gages_timeseries.nc"))
    ts_mopex = xr.open_dataset(CACHE_DIR.joinpath("mopex_timeseries.nc"))
    attr = xr.open_dataset(CACHE_DIR.joinpath("gages_attributes.nc"))
    target = ts_gages[target_cols].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_gages = ts_gages[relevant_cols[0]].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_mopex = ts_mopex[relevant_cols[1]].sel(basin=sites_id, time=slice(time[0], time[1]))
    forcing = ts_gages[relevant_cols[2:]].sel(basin=sites_id, time=slice(time[0], time[1]))
    constant = attr[constant_cols].sel(basin=sites_id)

    # dates = pd.date_range(start=time[0], end=time[1], periods=366)
    basins = target.basin.size
    times = target.time.size
    results = []

    average_target_over_basin = target.mean(dim='time')
    average_p_gages_over_basin = p_gages.mean(dim='time')
    average_p_mopex_over_basin = p_mopex.mean(dim='time')
    average_forcing_over_basin = forcing.mean(dim='time')
    constant_over_basin = constant

    with pd.ExcelWriter('output_data.xlsx') as writer:
        df = average_target_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Target Average')
        # for var_name in average_target_over_basin.data_vars:
        #     df = average_target_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)
        df_gages = average_p_gages_over_basin.to_dataframe(name='Average P Gages')
        df_mopex = average_p_mopex_over_basin.to_dataframe(name='Average P Mopex')
        df_p_combined = pd.concat([df_gages, df_mopex], axis=1)
        df_p_combined.to_excel(writer, sheet_name='Combined P Data')
        df = average_forcing_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Forcing Average')
        df = constant_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Constant Data')
        for basin in range(basins):
            basin_id = target.basin.values[basin]
            basin_data = target.streamflow.isel(basin=basin)

            # 计算有效日期范围，忽略 NaN
            valid_dates = basin_data.dropna('time').time
            if valid_dates.size > 0:
                time_range = (pd.to_datetime(valid_dates.min().values).strftime('%Y-%m-%d'), pd.to_datetime(valid_dates.max().values).strftime('%Y-%m-%d'))
            else:
                time_range = (None, None)  # 如果全是NaN

            # 计算缺失率
            missing_rate = basin_data.isnull().mean().item()

            # 计算平均径流，忽略缺失值
            average_streamflow = basin_data.mean().item() if valid_dates.size > 0 else float('nan')
            results.append((basin_id, time_range, missing_rate, average_streamflow))

        df = pd.DataFrame(results, columns=['Basin', 'Time Range', 'Missing Rate', 'Average Streamflow'])

        # 按照流域编号排序
        df = df.sort_values(by='Basin').reset_index(drop=True)
        df.to_excel(writer, sheet_name='Streamflow Data')
