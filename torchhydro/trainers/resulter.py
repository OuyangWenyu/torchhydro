import numpy as np
import pandas as pd
import os
from typing import Tuple, Union
import fnmatch

from hydroutils.hydro_file import unserialize_numpy
from hydroutils.hydro_stat import stat_error

from torchhydro.trainers.train_logger import save_model_params_log
from torchhydro.explainers.shap import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
    shap_summary_plot,
)
from torchhydro.trainers.train_utils import calculate_and_record_metrics


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
            epoch_name = "best"
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
        eval_log = {}
        for i, col in enumerate(target_col):
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

        # Finally, try to explain model behaviour using shap
        is_shap = self.cfgs["evaluation_cfgs"]["explainer"] == "shap"
        if is_shap:
            shap_summary_plot(self.model, self.traindataset, self.testdataset)
            # deep_explain_model_summary_plot(self.model, test_data)
            # deep_explain_model_heatmap(self.model, test_data)

        return eval_log

    def load_result(self, not_only_1out=False) -> Tuple[np.array, np.array]:
        """load the pred value of testing period and obs value

        Parameters
        ----------
        not_only_1out : bool, optional
            Sometimes our model give multiple output and we will load all of them,
            then we set this parameter True, by default False

        Returns
        -------
        Tuple[np.array, np.array]
            _description_
        """
        save_dir = self.result_dir
        flow_pred_file = os.path.join(save_dir, self.pred_name)
        flow_obs_file = os.path.join(save_dir, self.obs_name)
        pred = unserialize_numpy(flow_pred_file)
        obs = unserialize_numpy(flow_obs_file)
        if not_only_1out:
            return pred, obs
        if obs.ndim == 3 and obs.shape[-1] == 1:
            if pred.shape[-1] != obs.shape[-1]:
                # TODO: for convenient, now we didn't process this special case for MTL
                pred = pred[:, :, 0]
            pred = pred.reshape(pred.shape[0], pred.shape[1])
            obs = obs.reshape(obs.shape[0], obs.shape[1])
        return pred, obs

    def stat_result_for1out(self, pred, obs, fill_nan):
        """
        show the statistics result for 1 output
        """
        inds = stat_error(obs, pred, fill_nan=fill_nan)
        inds_df = pd.DataFrame(inds)
        return inds_df, obs, pred

    def stat_result(
        self,
        save_dirs: str,
        test_epoch: int,
        return_value: bool = False,
        fill_nan: Union[str, list, tuple] = "no",
        unit="m3/s",
        basin_area=None,
        var_name=None,
    ) -> Tuple[pd.DataFrame, np.array, np.array]:
        """
        Show the statistics result

        Parameters
        ----------
        save_dirs : str
            where we read results
        test_epoch : int
            the epoch of test
        return_value : bool, optional
            if True, returen pred and obs data, by default False
        fill_nan : Union[str, list, tuple], optional
            how to deal with nan in obs, by default "no"
        unit : str, optional
            unit of flow, by default "m3/s"
            if m3/s, then didn't transform; else transform to m3/s

        Returns
        -------
        Tuple[pd.DataFrame, np.array, np.array]
            statistics results, 3-dim predicitons, 3-dim observations
        """
        pred, obs = self.load_result(save_dirs, test_epoch)
        if type(unit) is list:
            inds_df_lst = []
            pred_lst = []
            obs_lst = []
            for i in range(len(unit)):
                inds_df_, pred_, obs_ = self.stat_result_for1out(
                    var_name[i],
                    unit[i],
                    pred[:, :, i],
                    obs[:, :, i],
                    fill_nan[i],
                    basin_area=basin_area,
                )
                inds_df_lst.append(inds_df_)
                pred_lst.append(pred_)
                obs_lst.append(obs_)
            return inds_df_lst, pred_lst, obs_lst if return_value else inds_df_lst
        else:
            inds_df_, pred_, obs_ = self.stat_result_for1out(
                var_name, unit, pred, obs, fill_nan, basin_area=basin_area
            )
            return (inds_df_, pred_, obs_) if return_value else inds_df_

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

    def stat_ensemble_result(
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
