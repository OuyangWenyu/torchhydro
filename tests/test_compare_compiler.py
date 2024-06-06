import json

# import matplotlib.pyplot as plt
import hydrodatasource.configs.config as hdscc
import numpy as np
import pandas as pd
import xarray as xr


def test_comp_compiler():
    origin_563_basins_era5l = "/ftproot/era5land/563_basins_era5_land.nc"
    smap_camels_sl_split_0 = "/ftproot/smap/camels_songliao_split_0.nc"
    origin_biliu_str = 's3://basins-origin/hour_data/1h/mean_data/streamflow_basin/streamflow_21401550.nc'
    final_forcing_biliu = 's3://basins-origin/hour_data/1h/mean_data/data_forcing_era5land/data_forcing_21401550.nc'
    final_forcing_str_biliu = 's3://basins-origin/hour_data/1h/mean_data/data_forcing_era5land_streamflow/data_forcing_streamflow_21401550.nc'
    split_era5land_result = split_basin_from_origin_era5land('era5land', origin_563_basins_era5l)
    split_smap_result = split_basin_from_smap(smap_camels_sl_split_0)
    streamflow_result = rename_origin_streamflow(origin_biliu_str)
    split_forcing_result = concat_era5land_smap('era5land', split_era5land_result, split_smap_result)
    print('______________________________')
    print(split_forcing_result)
    print(split_forcing_result['sm_surface'].to_numpy())
    print('______________________________')
    final_forcing_ds = xr.open_dataset(hdscc.FS.open(final_forcing_biliu))
    print(final_forcing_ds)
    print(final_forcing_ds['sm_surface'].to_numpy())
    print('______________________________')
    # print(streamflow_result['streamflow'])
    predct = streamflow_result.time >= np.datetime64('2015-03-31 01:00:00')
    print(streamflow_result.sel(time=predct)['streamflow'])
    print('______________________________')
    final_str_arr = xr.open_dataset(hdscc.FS.open(final_forcing_str_biliu))['streamflow']
    print(final_str_arr)
    print(final_str_arr.to_numpy())


def split_basin_from_origin_era5land(data_name, file_name):
    # copy from https://github.com/iHeadWater/HydroDataCompiler/blob/main/notebooks/convert_concat_origin_data_to_train_dataset.ipynb
    ds = xr.open_dataset(file_name)
    ds = ds.rename({'time_start': 'time'})
    ds['time'] = ds['time'].astype('datetime64[ns]')
    ds = ds.astype('float32')
    ds['basin_id'] = ds['basin_id'].astype(str).str.zfill(8)
    # for basin_id in ds['basin_id'].values:
    # only split out 1 basin from original data file
    basin_id = '21401550'
    basin_ds = ds.sel(basin_id=basin_id)
    # basin_ds = basin_ds.drop('basin_id')
    output_file = f'{data_name}_basin/{data_name}_{basin_id}.nc'
    basin_ds.to_netcdf(output_file)
    return output_file


def split_basin_from_smap(file_path):
    # copy from https://github.com/iHeadWater/HydroDataCompiler/blob/main/notebooks/convert_concat_origin_data_to_train_dataset.ipynb
    ds = xr.open_dataset(file_path)
    ds = ds.rename({'time_start': 'time'})
    ds['time'] = ds['time'].astype('datetime64[ns]')
    ds['basin_id'] = ds['basin_id'].astype(str).str.zfill(8)
    ds = ds.astype('float32')
    ds = ds.drop_vars([var for var in ds.data_vars if var not in ['sm_surface', 'sm_rootzone']])

    data_shifted = ds.assign_coords(time=ds.time - pd.Timedelta(minutes=30))
    start_date = data_shifted['time'].values[0]
    end_date = data_shifted.time[-1].values
    hourly_time = pd.date_range(start=start_date, end=end_date, freq='h')
    data_hourly = data_shifted.reindex(time=hourly_time, method='ffill')
    # for basin_id in ds['basin_id'].values:
    # only split out 1 basin from original data file
    basin_id = '21401550'
    basin_ds = data_hourly.sel(basin_id=basin_id)
    basin_ds = basin_ds.drop('basin_id')
    output_file = f'smap_basin/smap_{basin_id}.nc'
    basin_ds.to_netcdf(output_file)
    return output_file


def concat_era5land_smap(data_name, file_path1, file_path2):
    basin_id = '21401550'
    if 's3' in file_path1:
        dataset1 = xr.open_dataset(hdscc.FS.open(file_path1))
    else:
        dataset1 = xr.open_dataset(file_path1)
    if 's3' in file_path2:
        dataset2 = xr.open_dataset(hdscc.FS.open(file_path1))
    else:
        dataset2 = xr.open_dataset(file_path2)
    dataset2_times = pd.to_datetime(dataset2.time.values)
    dataset1_times = pd.to_datetime(dataset1.time.values)
    common_times = pd.Index(dataset2_times).intersection(pd.Index(dataset1_times))

    dataset2_common = dataset2.sel(time=common_times)
    dataset1_common = dataset1.sel(time=common_times)
    combined_dataset = xr.merge([dataset2_common, dataset1_common])
    combined_dataset.to_netcdf(f'data_forcing_{data_name}/data_forcing_{basin_id}.nc')
    return combined_dataset


def rename_origin_streamflow(file_path):
    # copy from https://github.com/iHeadWater/HydroDataCompiler/blob/main/notebooks/convert_concat_origin_data_to_train_dataset.ipynb
    ds = xr.open_dataset(hdscc.FS.open(file_path))
    # ds = ds.rename({'TM': 'time', 'Q': 'streamflow'})
    ds = ds.astype('float32')
    basename = file_path.split('.')[-2]
    basin_id = basename.split('_')[-1]
    start_time = ds.time.min().values
    end_time = ds.time.max().values
    continuous_time_range = pd.date_range(start=start_time, end=end_time, freq='h')
    ds = ds.reindex(time=continuous_time_range)
    units_dict = {"streamflow": "mm/h"}
    units_json = json.dumps(units_dict)
    ds.attrs['units'] = units_json
    output_file = f'streamflow_basin/streamflow_{basin_id}.nc'
    ds.to_netcdf(output_file)
    return ds


'''
def plot_streamflow_arrays(sm):
    basin = '21401550'
    sm_surface = sm['streamflow']
    # 绘制曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(sm_surface.time, sm_surface, label='sm_surface')
    plt.xlabel('Time')
    plt.ylabel('Streamflow')
    plt.title(f'{basin} Streamflow Over Time')
    plt.show()
'''
