import xarray as xr
data = xr.open_dataset("~/.cache/223_FloodEvents_38_repick_timeseries_1h_batch_sanxia_12020000_sanxia_60519980.nc")
print(data)