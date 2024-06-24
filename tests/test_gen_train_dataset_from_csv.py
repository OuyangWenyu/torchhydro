import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex
from numpy.lib.stride_tricks import as_strided as strided


def df_resample_cut(cut_df: DataFrame):
    # time_str = (step_cut_total_df['STEP'][step_cut_total_df['STCD'] == int(stcd)]).values[0]
    time_str = step_mode(cut_df)
    time_list = ['0 days 00:04:00', '0 days 00:05:00', '0 days 00:06:00', '0 days 00:12:00', '0 days 00:15:00',
                 '0 days 00:30:00', '0 days 01:00:00']
    if time_str in time_list:
        freq = '1h'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq).interpolate().dropna(how='all')
    elif time_str == '0 days 02:00:00':
        freq = '2h'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq).interpolate().dropna(how='all')
    elif time_str == '0 days 06:00:00':
        freq = '6h'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq, origin='end').interpolate()
        cut_df = cut_df.dropna(how='all')
    elif time_str == '1 days 00:00:00':
        freq = 'D'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq, origin='start').interpolate().dropna(how='all')
    else:
        freq = '3D'
    return cut_df, freq


def gen_train_test_xy(data_table: DataFrame, date_times: DatetimeIndex, pre_index: int, post_index: int):
    table, freq = df_resample_cut(data_table)
    # 以后除了水位以外，还可以有流量等变量
    true_variable_array = table['Z'][table.index.isin(date_times)].to_numpy()
    pre_shape = true_variable_array.shape[:-1] + (true_variable_array.shape[-1] - pre_index + 1, pre_index)
    post_shape = true_variable_array.shape[:-1] + (true_variable_array.shape[-1] - post_index + 1, post_index)
    # comma can't be ignored, otherwise will occur dimension mistake
    strides = true_variable_array.strides + (true_variable_array.strides[-1],)
    pre_window_result = strided(true_variable_array, shape=pre_shape, strides=strides)
    post_window_result = strided(true_variable_array, shape=post_shape, strides=strides)
    pre_train_df = pd.DataFrame(pre_window_result[:(pre_window_result.shape[0] - post_index - 1), :]).fillna(method='ffill')
    post_train_df = pd.DataFrame(post_window_result[pre_index+1:, :]).fillna(method='ffill')
    times_arr = ((date_times - pd.to_datetime('2000', format='%Y')) / np.timedelta64(1, 'h')).to_numpy()
    pre_train_df['index'] = times_arr[:(pre_window_result.shape[0] - post_index - 1)]
    post_train_df['index'] = times_arr[pre_index+1:post_window_result.shape[0]]
    return pre_train_df, post_train_df


def step_mode(csv_df):
    diffs = pd.to_datetime(csv_df['TM']).diff()
    return str(diffs.mode()[0])
