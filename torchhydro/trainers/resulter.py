import os
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr
import fnmatch
from typing import Tuple
from functools import reduce
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tbparse import SummaryReader

from hydroutils.hydro_stat import stat_error
from hydrodatasource.utils.utils import streamflow_unit_conv

from torchhydro.configs.model_config import MODEL_PARAM_TEST_WAY
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.trainers.train_logger import save_model_params_log
from torchhydro.explainers.shap import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
    shap_summary_plot,
)
from torchhydro.trainers.train_utils import (
    calculate_and_record_metrics,
    cellstates_when_inference,
    read_pth_from_model_loader,
    model_infer,
)
from torchhydro.trainers.deep_hydro import DeepHydro


def set_unit_to_var(ds):
    units_dict = ds.attrs["units"]
    for var_name, units in units_dict.items():
        if var_name in ds:
            ds[var_name].attrs["units"] = units
    if "units" in ds.attrs:
        del ds.attrs["units"]
    return ds


class Resulter:
    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        self.result_dir = cfgs["data_cfgs"]["test_path"]
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    @property
    def pred_name(self):
        return f"epoch{str(self.chosen_trained_epoch)}flow_pred"

    @property
    def obs_name(self):
        return f"epoch{str(self.chosen_trained_epoch)}flow_obs"

    @property
    def chosen_trained_epoch(self):
        model_loader = self.cfgs["evaluation_cfgs"]["model_loader"]
        if model_loader["load_way"] == "specified":
            epoch_name = str(model_loader["test_epoch"])
        elif model_loader["load_way"] == "best":
            # NOTE: TO make it consistent with the name in case of model_loader["load_way"] == "pth", the name have to be "best_model.pth"
            epoch_name = "best_model.pth"
        elif model_loader["load_way"] == "latest":
            epoch_name = str(self.cfgs["training_cfgs"]["epochs"])
        elif model_loader["load_way"] == "pth":
            epoch_name = model_loader["pth_path"].split(os.sep)[-1]
        else:
            raise ValueError("Invalid load_way")
        return epoch_name

    def save_cfg(self, cfgs):
        # save the cfgs after training
        # update the cfgs with the latest one
        self.cfgs = cfgs
        param_file_exist = any(
            (
                fnmatch.fnmatch(file, "*.json")
                and "_stat" not in file  # statistics json file
                and "_dict" not in file  # data cache json file
            )
            for file in os.listdir(self.result_dir)
        )
        if not param_file_exist:
            # although we save params log during training, but sometimes we directly evaluate a model
            # so here we still save params log if param file does not exist
            # no param file was saved yet, here we save data and params setting
            save_model_params_log(cfgs, self.result_dir)

    def save_result(self, pred, obs):
        """
        save the pred value of testing period and obs value

        Parameters
        ----------
        pred
            predictions
        obs
            observations
        pred_name
            the file name of predictions
        obs_name
            the file name of observations

        Returns
        -------
        None
        """
        save_dir = self.result_dir
        flow_pred_file = os.path.join(save_dir, self.pred_name)
        flow_obs_file = os.path.join(save_dir, self.obs_name)
        pred = set_unit_to_var(pred)
        obs = set_unit_to_var(obs)
        pred.to_netcdf(flow_pred_file + ".nc")
        obs.to_netcdf(flow_obs_file + ".nc")

    def eval_result(self, preds_xr, obss_xr):
        # types of observations
        target_col = self.cfgs["data_cfgs"]["target_cols"]
        evaluation_metrics = self.cfgs["evaluation_cfgs"]["metrics"]
        basin_ids = self.cfgs["data_cfgs"]["object_ids"]
        test_path = self.cfgs["data_cfgs"]["test_path"]
        # Assume object_ids like ['changdian_61561']
        # fill_nan: "no" means ignoring the NaN value;
        #           "sum" means calculate the sum of the following values in the NaN locations.
        #           For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
        #           Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5].
        #           If it is a str, then all target vars use same fill_nan method;
        #           elif it is a list, each for a var
        fill_nan = self.cfgs["evaluation_cfgs"]["fill_nan"]
        #  Then evaluate the model metrics
        if type(fill_nan) is list and len(fill_nan) != len(target_col):
            raise ValueError("length of fill_nan must be equal to target_col's")
        for i, col in enumerate(target_col):
            eval_log = {}
            obs = obss_xr[col].to_numpy()
            pred = preds_xr[col].to_numpy()

            eval_log = calculate_and_record_metrics(
                obs,
                pred,
                evaluation_metrics,
                col,
                fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                eval_log,
            )
            # Create pandas DataFrames from eval_log for each target variable (e.g., streamflow)
            # Create a dictionary to hold the data for the DataFrame
            data = {}
            # Iterate over metrics in eval_log
            for metric, values in eval_log.items():
                # Remove 'of streamflow' (or similar) from the metric name
                clean_metric = metric.replace(f"of {col}", "").strip()

                # Add the cleaned metric to the data dictionary
                data[clean_metric] = values

            # Create a DataFrame using object_ids as the index and metrics as columns
            df = pd.DataFrame(data, index=basin_ids)

            # Define the output file name based on the target variable
            output_file = os.path.join(test_path, f"metric_{col}.csv")

            # Save the DataFrame to a CSV file
            df.to_csv(output_file, index_label="basin_id")

        # Finally, try to explain model behaviour using shap
        is_shap = self.cfgs["evaluation_cfgs"]["explainer"] == "shap"
        if is_shap:
            shap_summary_plot(self.model, self.traindataset, self.testdataset)
            # deep_explain_model_summary_plot(self.model, test_data)
            # deep_explain_model_heatmap(self.model, test_data)

    def _convert_streamflow_units(self, ds):
        """convert the streamflow units to m^3/s

        Parameters
        ----------
        pred : np.array
            predictions

        Returns
        -------
        """
        data_cfgs = self.cfgs["data_cfgs"]
        source_name = data_cfgs["source_cfgs"]["source_name"]
        source_path = data_cfgs["source_cfgs"]["source_path"]
        other_settings = data_cfgs["source_cfgs"].get("other_settings", {})
        data_source = data_sources_dict[source_name](source_path, **other_settings)
        basin_id = data_cfgs["object_ids"]
        # NOTE: all datasource should have read_area method
        basin_area = data_source.read_area(basin_id)
        target_unit = "m^3/s"
        # NOTE: the name of var flow should be streamflow
        var_flow = "streamflow"
        streamflow_ds = ds[[var_flow]]
        ds_ = streamflow_unit_conv(
            streamflow_ds, basin_area, target_unit=target_unit, inverse=True
        )
        new_ds = ds.copy(deep=True)
        new_ds[var_flow] = ds_[var_flow]
        return new_ds

    def load_result(self, convert_flow_unit=False) -> Tuple[np.array, np.array]:
        """load the pred value of testing period and obs value"""
        save_dir = self.result_dir
        pred_file = os.path.join(save_dir, self.pred_name + ".nc")
        obs_file = os.path.join(save_dir, self.obs_name + ".nc")
        pred = xr.open_dataset(pred_file)
        obs = xr.open_dataset(obs_file)
        if convert_flow_unit:
            pred = self._convert_streamflow_units(pred)
            obs = self._convert_streamflow_units(obs)
        return pred, obs

    def save_intermediate_results(self, **kwargs):
        """Load model weights and deal with some intermediate results"""
        is_cell_states = kwargs.get("is_cell_states", False)
        is_pbm_params = kwargs.get("is_pbm_params", False)
        cfgs = self.cfgs
        cfgs["training_cfgs"]["train_mode"] = False
        training_cfgs = cfgs["training_cfgs"]
        seq_first = training_cfgs["which_first_tensor"] == "sequence"
        if is_cell_states:
            # TODO: not support return_cell_states yet
            return cellstates_when_inference(seq_first, data_cfgs, pred)
        if is_pbm_params:
            self._save_pbm_params(cfgs, seq_first)

    def _save_pbm_params(self, cfgs, seq_first):
        training_cfgs = cfgs["training_cfgs"]
        model_loader = cfgs["evaluation_cfgs"]["model_loader"]
        model_pth_dir = cfgs["data_cfgs"]["test_path"]
        weight_path = read_pth_from_model_loader(model_loader, model_pth_dir)
        cfgs["model_cfgs"]["weight_path"] = weight_path
        cfgs["training_cfgs"]["device"] = [0] if torch.cuda.is_available() else [-1]
        deephydro = DeepHydro(cfgs)
        device = deephydro.device
        dl_model = deephydro.model.dl_model
        pb_model = deephydro.model.pb_model
        param_func = deephydro.model.param_func
        # TODO: check for dplnnmodule model
        param_test_way = deephydro.model.param_test_way
        test_dataloader = DataLoader(
            deephydro.testdataset,
            batch_size=training_cfgs["batch_size"],
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        )
        deephydro.model.eval()
        # here the batch is just an index of lookup table, so any batch size could be chosen
        params_lst = []
        with torch.no_grad():
            for xs, ys in test_dataloader:
                ys, gen = model_infer(seq_first, device, dl_model, xs[1], ys)
                # we set all params' values in [0, 1] and will scale them when forwarding
                if param_func == "clamp":
                    params_ = torch.clamp(gen, min=0.0, max=1.0)
                elif param_func == "sigmoid":
                    params_ = F.sigmoid(gen)
                else:
                    raise NotImplementedError(
                        "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
                    )
                # just get one-period values, here we use the final period's values
                params = params_[:, -1, :]
                params_lst.append(params)
        pb_params = reduce(lambda a, b: torch.cat((a, b), dim=0), params_lst)
        # trans tensor to pandas dataframe
        sites = deephydro.cfgs["data_cfgs"]["object_ids"]
        params_names = pb_model.params_names
        params_df = pd.DataFrame(
            pb_params.cpu().numpy(), columns=params_names, index=sites
        )
        save_param_file = os.path.join(
            model_pth_dir, f"pb_params_{int(time.time())}.csv"
        )
        params_df.to_csv(save_param_file, index_label="GAGE_ID")

    def read_tensorboard_log(self, **kwargs):
        """read tensorboard log files"""
        is_scalar = kwargs.get("is_scalar", False)
        is_histogram = kwargs.get("is_histogram", False)
        log_dir = self.cfgs["data_cfgs"]["test_path"]
        if is_scalar:
            scalar_file = os.path.join(log_dir, "tb_scalars.csv")
            if not os.path.exists(scalar_file):
                reader = SummaryReader(log_dir)
                df_scalar = reader.scalars
                df_scalar.to_csv(scalar_file, index=False)
            else:
                df_scalar = pd.read_csv(scalar_file)
        if is_histogram:
            histogram_file = os.path.join(log_dir, "tb_histograms.csv")
            if not os.path.exists(histogram_file):
                reader = SummaryReader(log_dir)
                df_histogram = reader.histograms
                df_histogram.to_csv(histogram_file, index=False)
            else:
                df_histogram = pd.read_csv(histogram_file)
        if is_scalar and is_histogram:
            return df_scalar, df_histogram
        elif is_scalar:
            return df_scalar
        elif is_histogram:
            return df_histogram

    # TODO: the following code is not finished yet
    def load_ensemble_result(
        self, save_dirs, test_epoch, flow_unit="m3/s", basin_areas=None
    ) -> Tuple[np.array, np.array]:
        """
        load ensemble mean value

        Parameters
        ----------
        save_dirs
        test_epoch
        flow_unit
            default is m3/s, if it is not m3/s, transform the results
        basin_areas
            if unit is mm/day it will be used, default is None

        Returns
        -------

        """
        preds = []
        obss = []
        for save_dir in save_dirs:
            pred_i, obs_i = self.load_result(save_dir, test_epoch)
            if pred_i.ndim == 3 and pred_i.shape[-1] == 1:
                pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
                obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            preds.append(pred_i)
            obss.append(obs_i)
        preds_np = np.array(preds)
        obss_np = np.array(obss)
        pred_mean = np.mean(preds_np, axis=0)
        obs_mean = np.mean(obss_np, axis=0)
        if flow_unit == "mm/day":
            if basin_areas is None:
                raise ArithmeticError("No basin areas we cannot calculate")
            basin_areas = np.repeat(basin_areas, obs_mean.shape[1], axis=0).reshape(
                obs_mean.shape
            )
            obs_mean = obs_mean * basin_areas * 1e-3 * 1e6 / 86400
            pred_mean = pred_mean * basin_areas * 1e-3 * 1e6 / 86400
        elif flow_unit == "m3/s":
            pass
        elif flow_unit == "ft3/s":
            obs_mean = obs_mean / 35.314666721489
            pred_mean = pred_mean / 35.314666721489
        return pred_mean, obs_mean

    def eval_ensemble_result(
        self,
        save_dirs,
        test_epoch,
        return_value=False,
        flow_unit="m3/s",
        basin_areas=None,
    ) -> Tuple[np.array, np.array]:
        """calculate statistics for ensemble results

        Parameters
        ----------
        save_dirs : _type_
            where the results save
        test_epoch : _type_
            we name the results files with the test_epoch
        return_value : bool, optional
            if True, return (inds_df, pred_mean, obs_mean), by default False
        flow_unit : str, optional
            arg for load_ensemble_result, by default "m3/s"
        basin_areas : _type_, optional
            arg for load_ensemble_result, by default None

        Returns
        -------
        Tuple[np.array, np.array]
            inds_df or (inds_df, pred_mean, obs_mean)
        """
        pred_mean, obs_mean = self.load_ensemble_result(
            save_dirs, test_epoch, flow_unit=flow_unit, basin_areas=basin_areas
        )
        inds = stat_error(obs_mean, pred_mean)
        inds_df = pd.DataFrame(inds)
        return (inds_df, pred_mean, obs_mean) if return_value else inds_df
