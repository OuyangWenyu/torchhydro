import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tbparse import SummaryReader
from matplotlib import cm, colors, pyplot as plt
from scipy.spatial.distance import cosine

from hydroutils import hydro_plot as hplt


def read_layer_name_from_tb_hist(hist_cols):
    layer_names = []
    for col in hist_cols:
        if "counts" in col:
            layer_name = col.split("/")[0]
            if layer_name not in layer_names:
                layer_names.append(layer_name)
    return layer_names


def epochs_hist_for_chosen_layer(epoch_interval, layer_name, df_hist):
    df = pd.DataFrame()
    all_epochs = df_hist.shape[0]
    limit_uppers = []
    limit_lowers = []
    for i in range(0, all_epochs, epoch_interval):
        limits = df_hist[layer_name + "/limits"][i]
        limit_uppers.append(limits.max())
        limit_lowers.append(limits.min())
    for i in range(0, all_epochs, epoch_interval):
        counts = df_hist[layer_name + "/counts"][i]
        limits = df_hist[layer_name + "/limits"][i]
        x, y = SummaryReader.histogram_to_bins(
            counts,
            limits,
            lower_bound=min(limit_lowers),
            upper_bound=max(limit_uppers),
            # n_bins=100,
        )
        df[i] = y
    df.index = x
    return df


def plot_layer_hist_for_basin_model_fold(
    model_name, chosen_layer_values, layers, bsize, cmap_str="Oranges"
):
    """_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    chosen_layer_values : _type_
        _description_
    layers : _type_
        _description_
    bsize : _type_
        _description_
    cmap_str : str, optional
        chose from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential, by default "Oranges"
    """
    project_dir = os.getcwd()
    result_dir = os.path.join(
        project_dir, "results", "tensorboard", "histograms", f"bsize{bsize}"
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for layer in layers:
        two_model_layers = chosen_layer_values[layer]
        try:
            df_lstm_show = two_model_layers[model_name]
        except KeyError:
            # if the model does not have this layer, skip
            continue
        lstm_x_lst = []
        lstm_y_lst = []
        lstm_dash_lines = []
        color_str = ""
        lw_lst = []
        alpha_lst = []
        cmap = cm.get_cmap(cmap_str)
        rgb_lst = []
        norm_color = colors.Normalize(vmin=0, vmax=df_lstm_show.shape[1])
        for i in df_lstm_show:
            lstm_x_lst.append(df_lstm_show.index.values)
            lstm_y_lst.append(df_lstm_show[i].values)
            lstm_dash_lines.append(True)
            color_str = color_str + "r"
            rgba = cmap(norm_color(i))
            rgb_lst.append(rgba)
            alpha_lst.append(0.5)
            lw_lst.append(0.5)
        # the first and last line should be solid, have dark color and wide width
        rgb_lst[0] = rgba
        lstm_dash_lines[-1] = False
        alpha_lst[-1] = 1
        alpha_lst[0] = 1
        lw_lst[-1] = 1
        lw_lst[0] = 1
        hplt.plot_ts(
            lstm_x_lst,
            lstm_y_lst,
            dash_lines=lstm_dash_lines,
            fig_size=(8, 4),
            xlabel="权值",
            ylabel="频数",
            # c_lst=color_str,
            c_lst=rgb_lst,
            linewidth=lw_lst,
            alpha=alpha_lst,
            leg_lst=None,
        )
        plt.savefig(
            os.path.join(
                result_dir,
                f"{basin_id}_fold{fold}_{model_name}_{layer}_hist.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )


def chosen_layer_in_layers(layers, chosen_layers):
    the_layers = []
    for layer in layers:
        the_layers.extend(
            layer for a_chosen_layer in chosen_layers if a_chosen_layer in layer
        )
    return the_layers


def get_latest_event_file(event_file_lst):
    """Get the latest event file in the current directory.

    Returns
    -------
    str
        The latest event file.
    """
    event_files = [Path(f) for f in event_file_lst]
    event_file_names_lst = [event_file.stem.split(".") for event_file in event_files]
    ctimes = [
        int(event_file_names[event_file_names.index("tfevents") + 1])
        for event_file_names in event_file_names_lst
    ]
    return event_files[ctimes.index(max(ctimes))]


def copy_latest_tblog_file(log_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        copy_lst = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.startswith("events")
        ]
        copy_file = get_latest_event_file(copy_lst)
        shutil.copy(copy_file, result_dir)


def read_tb_log(
    a_exp, best_batchsize, exp_example="gages", where_save="transfer_learning"
):
    """Copy a recent log file to the current directory and read the log file.

    Parameters
    ----------
    a_exp : _type_
        _description_
    best_batchsize : _type_
        _description_
    exp_example : str, optional
        _description_, by default "gages"
    where_save : str, optional
        A directory in "app" directory, by default "transfer_learning"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    project_dir = os.getcwd()
    result_dir = os.path.join(project_dir, "results")
    log_dir = os.path.join(
        result_dir,
        exp_example,
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log dir {log_dir} not found!")
    result_dir = os.path.join(
        result_dir,
        where_save,
        "results",
        "tensorboard",
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    copy_latest_tblog_file(log_dir, result_dir)
    scalar_file = os.path.join(result_dir, "scalars.csv")
    if not os.path.exists(scalar_file):
        reader = SummaryReader(result_dir)
        df_scalar = reader.scalars
        df_scalar.to_csv(scalar_file, index=False)
    else:
        df_scalar = pd.read_csv(scalar_file)

    # reader = SummaryReader(result_dir)
    histgram_file = os.path.join(result_dir, "histograms.pkl")
    if not os.path.exists(histgram_file):
        reader = SummaryReader(result_dir, pivot=True)
        df_histgram = reader.histograms
        # https://www.statology.org/pandas-save-dataframe/
        df_histgram.to_pickle(histgram_file)
    else:
        df_histgram = pd.read_pickle(histgram_file)
    return df_scalar, df_histgram


# plot param hist for each basin
def plot_param_hist_model(
    model_name,
    a_exp,
    batchsize,
    chosen_layer_for_hist,
):
    """plot paramter histogram for each basin

    Parameters
    ----------
    model_name: str
        name of a DL model
    a_exp : str
        the experiment name
    batchsize : int
        batch size
    chosen_layer_for_hist : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    chosen_layer_values = {layer: {} for layer in chosen_layer_for_hist}
    chosen_layer_values_consine = {layer: {} for layer in chosen_layer_for_hist}
    df_scalar, df_histgram = read_tb_log(a_exp, batchsize)
    hist_cols = df_histgram.columns.values
    model_layers = read_layer_name_from_tb_hist(hist_cols)
    chosen_layers = chosen_layer_in_layers(model_layers, chosen_layer_for_hist)
    k = 0
    for layer in chosen_layer_for_hist:
        if layer not in chosen_layers[k]:
            continue
        df_epochs_hist = epochs_hist_for_chosen_layer(10, chosen_layers[k], df_histgram)
        chosen_layer_values[layer][model_name] = df_epochs_hist
        chosen_layer_values_consine[layer][model_name] = 1 - cosine(
            df_epochs_hist[0], df_epochs_hist[90]
        )
        k = k + 1
    plot_layer_hist_for_basin_model_fold(
        model_name,
        chosen_layer_values,
        chosen_layer_for_hist,
        batchsize,
        cmap_str="Reds",
    )
    return chosen_layer_values, chosen_layer_values_consine


def merge_value(arrs_lst):
    arrs = np.array(arrs_lst)
    return np.mean(arrs, axis=0)
