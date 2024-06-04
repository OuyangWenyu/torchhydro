import hydrodatasource.configs.config as hdscc
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray


def test_merge_minio_streamflow():
    basin_671_gdf = gpd.read_file("/ftproot/671_shp/671-hyd_na_dir_30s.shp")
    basin_671_gdf['STAID'] = basin_671_gdf['STAID'].apply(lambda x: x.split('-')[0].zfill(8))
    flow_files = hdscc.FS.glob(
        's3://stations-origin/zq_stations/hour_data/1h/usgs_datas_462_basins_after_2019/**')[1:]
    result_df = pd.DataFrame()
    for file in flow_files:
        hru_id = file.split('.')[0].split('_')[-1]
        if hru_id not in ['03049000', '03592718']:
            flow_df = pd.read_csv(hdscc.FS.open(file), engine='c', parse_dates=['datetime'])
        else:
            continue
        flow_df['site_no'] = flow_df['site_no'].astype(str).apply(lambda x: x.zfill(8))
        flow_df['datetime'] = flow_df['datetime'].dt.tz_localize(None)
        if '00060_discontinued' in flow_df.columns:
            flow_df = flow_df.rename(columns={'datetime': 'time_start', 'site_no': 'basin_id', '00060_discontinued': 'streamflow'})
        # 这种站只有水位，没有流量
        elif ('00060' not in flow_df.columns) & ('00060_discontinued' not in flow_df.columns):
            continue
        else:
            flow_df = flow_df.rename(columns={'datetime': 'time_start', 'site_no': 'basin_id', '00060': 'streamflow'})
        flow_df['streamflow'] = flow_df['streamflow'].astype(float)
        flow_df = flow_df[['basin_id', 'time_start', 'streamflow']]
        flow_df = flow_df.set_index('time_start').drop(columns=['basin_id'])
        flow_df = flow_df.resample('h').mean()
        hru_id_series = pd.DataFrame({'basin_id': np.repeat(hru_id, len(flow_df))})
        flow_df = pd.concat([flow_df.reset_index(), hru_id_series], axis=1)
        hru_area = basin_671_gdf['AREA'][basin_671_gdf['STAID'] == hru_id].to_list()[0]
        flow_df['streamflow'] = flow_df['streamflow'].apply(lambda x: x*3.6/(hru_area*35.31))
        camels_hourly_csv_path = f'/ftproot/camels_hourly/data/usgs_streamflow_csv/{hru_id}-usgs-hourly.csv'
        camels_hourly_df = pd.read_csv(camels_hourly_csv_path, engine='c')
        camels_hourly_df = camels_hourly_df.rename(columns={'date': 'time_start', 'QObs(mm/h)': 'streamflow'})
        camels_hourly_df = camels_hourly_df[camels_hourly_df['time_start'] >= '2015-01-01 00:00:00'].reset_index()
        camels_part_series = pd.DataFrame({'basin_id': np.repeat(hru_id, len(camels_hourly_df))})
        camels_part_df = pd.concat([camels_hourly_df, camels_part_series], axis=1)
        camels_part_df = camels_part_df[['basin_id', 'time_start', 'streamflow']]
        concat_df = pd.concat([camels_part_df, flow_df], axis=0)
        result_df = pd.concat([result_df, concat_df], axis=0)
    result_df['time_start'] = pd.to_datetime(result_df['time_start'])
    result_df = result_df.set_index(['time_start', 'basin_id'])
    result_df.to_csv('test_463_basins_2015_2023.csv')
    result_df = result_df[~result_df.index.duplicated()]
    result_ds = xarray.Dataset.from_dataframe(result_df)
    result_ds.to_netcdf('463_basins_2015_2023.nc')
