import xarray as xr
ts = xr.open_dataset("/Users/cylenlc/databackup/usgs-streamflow-nldas_hourly.nc")
print(ts)
print(ts["QObs(mm/d)"].attrs)   # 看径流的单位
print(ts["prcp(mm/day)"].attrs) # 看降水的单位
