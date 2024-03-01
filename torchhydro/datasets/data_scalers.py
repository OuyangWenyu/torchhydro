import copy
import json
import os
import pickle as pkl
import shutil
import pint_xarray  # noqa: F401
import xarray as xr
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from hydroutils.hydro_stat import (
    cal_stat_prcp_norm,
    cal_stat_gamma,
    cal_stat,
    cal_4_stat_inds,
)

from torchhydro.datasets.data_utils import (
    _trans_norm,
    _prcp_norm,
    wrap_t_s_dict,
    unify_streamflow_unit,
)

SCALER_DICT = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
}


class ScalerHub(object):
    """
    A class for Scaler
    """

    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: np.array,
        constant_vars: np.array = None,
        data_cfgs: dict = None,
        is_tra_val_te: str = None,
        data_source: object = None,
        **kwargs,
    ):
        """
        Perform normalization

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            dynamic input variables
        constant_vars
            static input variables
        other_vars
            other required variables
        data_cfgs
            configs for reading data
        is_tra_val_te
            train, valid or test
        kwargs
            other optional parameters for ScalerHub
        """
        self.data_cfgs = data_cfgs
        norm_keys = ["target_vars", "relevant_vars", "constant_vars"]
        norm_dict = {}
        scaler_type = data_cfgs["scaler"]
        if scaler_type == "DapengScaler":
            gamma_norm_cols = data_cfgs["scaler_params"]["gamma_norm_cols"]
            prcp_norm_cols = data_cfgs["scaler_params"]["prcp_norm_cols"]
            pbm_norm = data_cfgs["scaler_params"]["pbm_norm"]
            scaler = DapengScaler(
                target_vars,
                relevant_vars,
                constant_vars,
                data_cfgs,
                is_tra_val_te,
                prcp_norm_cols=prcp_norm_cols,
                gamma_norm_cols=gamma_norm_cols,
                pbm_norm=pbm_norm,
                data_source=data_source,
            )
            x, y, c = scaler.load_data()
            self.target_scaler = scaler

        elif scaler_type in SCALER_DICT.keys():
            # TODO: not fully tested, espacially for pbm models
            all_vars = [target_vars, relevant_vars, constant_vars]
            for i in range(len(all_vars)):
                data_tmp = all_vars[i]
                scaler = SCALER_DICT[scaler_type]()
                if data_tmp.ndim == 3:
                    # for forcings and outputs
                    num_instances, num_time_steps, num_features = data_tmp.shape
                    data_tmp = data_tmp.to_numpy().reshape(-1, num_features)
                    save_file = os.path.join(
                        data_cfgs["test_path"], f"{norm_keys[i]}_scaler.pkl"
                    )
                    if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
                        data_norm = scaler.fit_transform(data_tmp)
                        # Save scaler in test_path for valid/test
                        with open(save_file, "wb") as outfile:
                            pkl.dump(scaler, outfile)
                    else:
                        if data_cfgs["stat_dict_file"] is not None:
                            shutil.copy(data_cfgs["stat_dict_file"], save_file)
                        if not os.path.isfile(save_file):
                            raise FileNotFoundError(
                                "Please genereate xx_scaler.pkl file"
                            )
                        with open(save_file, "rb") as infile:
                            scaler = pkl.load(infile)
                            data_norm = scaler.transform(data_tmp)
                    data_norm = data_norm.reshape(
                        num_instances, num_time_steps, num_features
                    )
                else:
                    # for attributes
                    save_file = os.path.join(
                        data_cfgs["test_path"], f"{norm_keys[i]}_scaler.pkl"
                    )
                    if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
                        data_norm = scaler.fit_transform(data_tmp)
                        # Save scaler in test_path for valid/test
                        with open(save_file, "wb") as outfile:
                            pkl.dump(scaler, outfile)
                    else:
                        if data_cfgs["stat_dict_file"] is not None:
                            shutil.copy(data_cfgs["stat_dict_file"], save_file)
                        assert os.path.isfile(save_file)
                        with open(save_file, "rb") as infile:
                            scaler = pkl.load(infile)
                            data_norm = scaler.transform(data_tmp)
                norm_dict[norm_keys[i]] = data_norm
                if i == 0:
                    self.target_scaler = scaler
            x = norm_dict["relevant_vars"]
            y = norm_dict["target_vars"]
            c = norm_dict["constant_vars"]
        else:
            raise NotImplementedError(
                "We don't provide this Scaler now!!! Please choose another one: DapengScaler or key in SCALER_DICT"
            )
        print("Finish Normalization\n")
        self.x = x
        self.y = y
        self.c = c


class DapengScaler(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: np.array,
        constant_vars: np.array,
        data_cfgs: dict,
        is_tra_val_te: str,
        other_vars: dict = None,
        prcp_norm_cols=None,
        gamma_norm_cols=None,
        pbm_norm=False,
        data_source: object = None,
    ):
        """
        The normalization and denormalization methods from Dapeng's 1st WRR paper.
        Some use StandardScaler, and some use special norm methods

        Parameters
        ----------
        target_vars
            output variables
        relevant_vars
            input dynamic variables
        constant_vars
            input static variables
        data_cfgs
            data parameter config in data source
        is_tra_val_te
            train/valid/test
        other_vars
            if more input are needed, list them in other_vars
        prcp_norm_cols
            data items which use _prcp_norm method to normalize
        gamma_norm_cols
            data items which use log(\sqrt(x)+.1) method to normalize
        pbm_norm
            if true, use pbm_norm method to normalize; the output of pbms is not normalized data, so its inverse is different.
        """
        if prcp_norm_cols is None:
            prcp_norm_cols = [
                "streamflow",
            ]
        if gamma_norm_cols is None:
            gamma_norm_cols = ["p_mean", "prcp", "temperature"]
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.data_other = other_vars
        self.prcp_norm_cols = prcp_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        # both prcp_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = gamma_norm_cols + prcp_norm_cols
        self.pbm_norm = pbm_norm
        self.data_source = data_source
        # save stat_dict of training period in test_path for valid/test
        stat_file = os.path.join(data_cfgs["test_path"], "dapengscaler_stat.json")
        # for testing sometimes such as pub cases, we need stat_dict_file from trained dataset
        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all()
            with open(stat_file, "w") as fp:
                json.dump(self.stat_dict, fp)
        else:
            # for valid/test, we need to load stat_dict from train
            if data_cfgs["stat_dict_file"] is not None:
                # we used a assigned stat file, typically for PUB exps
                shutil.copy(data_cfgs["stat_dict_file"], stat_file)
            assert os.path.isfile(stat_file)
            with open(stat_file, "r") as fp:
                self.stat_dict = json.load(fp)

    def inverse_transform(self, target_values):
        """
        Denormalization for output variables

        Parameters
        ----------
        target_values
            output variables

        Returns
        -------
        np.array
            denormalized predictions
        """
        stat_dict = self.stat_dict
        target_cols = self.data_cfgs["target_cols"]
        if self.pbm_norm:
            # for pbm's output, its unit is mm/day, so we don't need to recover its unit
            pred = target_values
        else:
            pred = _trans_norm(
                target_values,
                target_cols,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=False,
            )
            for i in range(len(self.data_cfgs["target_cols"])):
                var = self.data_cfgs["target_cols"][i]
                if var in self.prcp_norm_cols:
                    mean_prep = self.data_source.read_mean_prcp(
                        self.t_s_dict["sites_id"]
                    )
                    pred.loc[dict(variable=var)] = _prcp_norm(
                        pred.sel(variable=var).to_numpy(),
                        mean_prep.to_array().to_numpy().T,
                        to_norm=False,
                    )
        # add attrs for units
        pred.attrs.update(self.data_target.attrs)
        # trans to xarray dataset
        pred_ds = pred.to_dataset(dim="variable")

        return pred_ds

    def cal_stat_all(self):
        """
        Calculate statistics of outputs(streamflow etc), and inputs(forcing and attributes)

        Returns
        -------
        dict
            a dict with statistic values
        """
        # streamflow
        target_cols = self.data_cfgs["target_cols"]
        stat_dict = {}
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                mean_prep = self.data_source.read_mean_prcp(self.t_s_dict["sites_id"])
                stat_dict[var] = cal_stat_prcp_norm(
                    self.data_target.sel(variable=var).to_numpy(),
                    mean_prep.to_array().to_numpy().T,
                )
            elif var in self.gamma_norm_cols:
                stat_dict[var] = cal_stat_gamma(
                    self.data_target.sel(variable=var).to_numpy()
                )
            else:
                stat_dict[var] = cal_stat(self.data_target.sel(variable=var).to_numpy())

        # forcing
        forcing_lst = self.data_cfgs["relevant_cols"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var in self.gamma_norm_cols:
                stat_dict[var] = cal_stat_gamma(x.sel(variable=var).to_numpy())
            else:
                stat_dict[var] = cal_stat(x.sel(variable=var).to_numpy())

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_cfgs["constant_cols"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data.sel(variable=var).to_numpy())

        return stat_dict

    def get_data_obs(self, to_norm: bool = True) -> np.array:
        """
        Get observation values

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the output value for modeling
        """
        stat_dict = self.stat_dict
        data = self.data_target
        out = xr.full_like(data, np.nan)
        # if we don't set a copy() here, the attrs of data will be changed, which is not our wish
        out.attrs = copy.deepcopy(data.attrs)
        target_cols = self.data_cfgs["target_cols"]
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                mean_prep = self.data_source.read_mean_prcp(self.t_s_dict["sites_id"])
                out.loc[dict(variable=var)] = _prcp_norm(
                    data.sel(variable=var).to_numpy(),
                    mean_prep.to_array().to_numpy().T,
                    to_norm=True,
                )
                out.attrs["units"][var] = "dimensionless"
        out = _trans_norm(
            out,
            target_cols,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        return out

    def get_data_ts(self, to_norm=True) -> np.array:
        """
        Get dynamic input data

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the dynamic inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_cfgs["relevant_cols"]
        data = self.data_forcing
        data = _trans_norm(
            data, var_lst, stat_dict, log_norm_cols=self.log_norm_cols, to_norm=to_norm
        )
        return data

    def get_data_const(self, to_norm=True) -> np.array:
        """
        Attr data and normalization

        Parameters
        ----------
        rm_nan
            if true, fill NaN value with 0
        to_norm
            if true, perform normalization

        Returns
        -------
        np.array
            the static inputs for modeling
        """
        stat_dict = self.stat_dict
        var_lst = self.data_cfgs["constant_cols"]
        data = self.data_attr
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        return data

    def load_data(self):
        """
        Read data and perform normalization for DL models

        Returns
        -------
        tuple
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        x = self.get_data_ts()
        y = self.get_data_obs()
        c = self.get_data_const()
        return x, y, c


class GPM_GFS_Scaler(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: dict,
        constant_vars: np.array,
        data_gfs: dict,
        data_soil: dict,
        data_cfgs: dict,
        is_tra_val_te: str,
        prcp_norm_cols=None,
        gamma_norm_cols=None,
        pbm_norm=False,
        basin_data: object = None,
    ):
        """
        A class for scaling and normalizing Global Precipitation Measurement (GPM) and Global Forecast System (GFS) datasets.

        This class handles the preprocessing of GPM and GFS datasets for use in hydrological models. It includes functionalities
        for normalizing precipitation data, transforming other relevant and constant variables, and managing datasets for
        different user access levels. The class supports normalization and inverse normalization operations.

        Attributes:
        - data_target: Target variable data (e.g., observed streamflow or water levels).
        - data_forcing: Forcing data variables relevant to the model.
        - data_attr: Constant attribute data for the basins.
        - data_gfs: GFS forecast data.
        - data_soil: soil attributes data.
        - data_cfgs: Configuration data for the scaler.
        - t_s_dict: Dictionary for time and site information.
        - prcp_norm_cols: Columns to normalize using precipitation normalization.
        - gamma_norm_cols: Columns to normalize using gamma normalization.
        - pbm_norm: Boolean flag to determine if PBM normalization is used.
        - log_norm_cols: Columns to normalize using logarithmic normalization.
        - stat_dict: Dictionary storing statistics for normalization.

        Methods:
        - cal_stat_all: Calculates statistics for all variables.
        - get_data_obs: Retrieves and normalizes observation data.
        - get_data_ts: Retrieves and normalizes time series data.
        - get_data_gfs: Retrieves and normalizes GFS data.
        - get_data_soil: Retrieves and normalizes soil data.
        - get_data_const: Retrieves and normalizes constant data.
        - load_data: Loads and processes all data types (observations, time series, GFS, constants).
        - inverse_transform: Applies inverse transformation to bring data back to original scale.
        - GPM_GFS_cal_stat_prcp_norm: Calculates statistics for precipitation normalized data.
        - GPM_GFS_cal_stat_gamma: Calculates statistics for gamma normalized data.
        - GPM_GFS_prcp_norm: Normalizes or denormalizes precipitation data.

        The class is designed to support different types of data handling and normalization strategies based on user roles
        (trainer or tester) and the specific requirements of hydrological modeling.
        """

        prcp_norm_cols = [
            "waterlevel",
            "streamflow",
        ]

        gamma_norm_cols = [
            "tp",
            "dswrf",
            "pwat",
            "2r",
            "2sh",
            "2t",
            "tcc",
            "10u",
            "10v",
        ]
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_gfs = data_gfs
        self.data_soil = data_soil
        self.data_cfgs = data_cfgs
        self.t_s_dict = wrap_t_s_dict(data_cfgs, is_tra_val_te)
        self.prcp_norm_cols = prcp_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        self.pbm_norm = pbm_norm
        # both prcp_norm_cols and gamma_norm_cols use log(\sqrt(x)+.1) method to normalize
        self.log_norm_cols = gamma_norm_cols + prcp_norm_cols
        self.basin_data = basin_data
        stat_file = os.path.join(data_cfgs["test_path"], "GPM_GFS_Scaler_stat.json")

        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all()
            with open(stat_file, "w") as fp:
                json.dump(self.stat_dict, fp)
        else:
            if data_cfgs["stat_dict_file"] is not None:
                if not os.path.exists(data_cfgs["test_path"]):
                    os.makedirs(data_cfgs["test_path"])
                shutil.copy(data_cfgs["stat_dict_file"], stat_file)
            assert os.path.isfile(stat_file)
            with open(stat_file, "r") as fp:
                self.stat_dict = json.load(fp)

    def cal_stat_all(self):
        """
        Calculates statistics for all variables in the dataset for scaling and normalization purposes.

        This method processes various types of data including target variables, rainfall forcing data, GFS forcing data,
        and constant attributes. It calculates statistics for each variable based on the specified normalization methods
        and stores them in a dictionary.

        Steps:
        1. Process target variables: For each target variable specified in the configuration, calculate statistics using
        the appropriate normalization method (e.g., precipitation normalization).
        2. Process rainfall forcing data: For each rainfall-related variable, calculate statistics using gamma normalization.
        3. Process GFS forcing data: For variables related to GFS data, apply gamma normalization and calculate statistics.
        4. Process constant attributes: Calculate basic statistics for constant attribute variables.

        Returns:
        A dictionary containing calculated statistics for each variable. The dictionary keys are the variable names, and
        the values are the corresponding statistical measures needed for normalization.

        This method supports different types of normalization, such as precipitation normalization and gamma normalization,
        tailored to the specific requirements of the dataset and modeling objectives.
        """
        stat_dict = {}

        # y
        target_cols = self.data_cfgs["target_cols"]
        y = self.data_target
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                mean_prep = self.basin_data.read_MP(
                    self.t_s_dict["sites_id"],
                    self.data_cfgs["attributes_path"],
                    self.data_cfgs["user"],
                )

                stat_dict[var] = self.GPM_GFS_cal_stat_prcp_norm(
                    y.sel(variable=var).to_numpy(),
                    mean_prep.to_numpy(),
                )

        # rainfall_forcing
        rainfall = self.data_cfgs["relevant_cols"][0]
        x = self.data_forcing
        for k in range(len(rainfall)):
            var = rainfall[k]
            if var in self.gamma_norm_cols:
                stat_dict[var] = self.GPM_GFS_cal_stat_gamma(x, var)

        # gfs_forcing
        gfs_lst = self.data_cfgs["relevant_cols"][1]
        if gfs_lst != ["None"]:
            data_gfs = self.data_gfs
            for k in range(len(gfs_lst)):
                var = gfs_lst[k]
                if var in self.gamma_norm_cols:
                    stat_dict[var] = self.GPM_GFS_cal_stat_gamma(data_gfs, var)

        # soil_attributes
        soil_lst = self.data_cfgs["relevant_cols"][2]
        if soil_lst != ["None"]:
            data_soil = self.data_soil
            for k in range(len(soil_lst)):
                var = soil_lst[k]
                stat_dict[var] = self.GPM_GFS_cal_stat_gamma(data_soil, var)

        # const attribute
        attr_lst = self.data_cfgs["constant_cols"]
        if attr_lst:
            attr_data = self.data_attr
            for k in range(len(attr_lst)):
                var = attr_lst[k]
                stat_dict[var] = cal_stat(attr_data.sel(variable=var).to_numpy())
        return stat_dict

    def get_data_obs(self, to_norm: bool = True) -> np.array:
        """
        Retrieves and optionally normalizes observation data for target variables.

        This method processes the target data and applies normalization based on the specified methods for each variable.
        It supports different normalization methods, including precipitation normalization and logarithmic normalization.

        Parameters:
        - to_norm: Boolean flag indicating whether to normalize the data (default is True).

        Steps:
        1. Initialize an output dataset with the same structure as the target data but filled with NaN values.
        2. Iterate over each target variable:
        - For precipitation-related variables, apply precipitation normalization using mean precipitation data.
        - For other variables, apply the normalization method as specified in the statistical dictionary.
        3. Optionally normalize the data using the transformation function `_trans_norm` based on the `to_norm` flag.

        Returns:
        An xarray Dataset containing the processed (and optionally normalized) target data. The data structure is similar
        to the input data, but with values transformed according to the specified normalization methods.

        This method is an integral part of data preprocessing, preparing observation data for use in hydrological models
        by ensuring that the data is in a suitable format and scale.
        """
        stat_dict = self.stat_dict
        data = self.data_target
        out = xr.full_like(data, np.nan)
        out.attrs = copy.deepcopy(data.attrs)
        target_cols = self.data_cfgs["target_cols"]
        for i in range(len(target_cols)):
            var = target_cols[i]
            if var in self.prcp_norm_cols:
                mean_prep = self.basin_data.read_MP(
                    self.t_s_dict["sites_id"],
                    self.data_cfgs["attributes_path"],
                    self.data_cfgs["user"],
                )

                out.loc[dict(variable=var)] = self.GPM_GFS_prcp_norm(
                    data.sel(variable=var).to_numpy(),
                    mean_prep.to_numpy(),
                    to_norm=True,
                )
                out.attrs["units"][var] = "dimensionless"
        out = _trans_norm(
            out,
            target_cols,
            stat_dict,
            log_norm_cols=self.log_norm_cols,
            to_norm=to_norm,
        )
        return out

    def get_data_ts(self, to_norm=True) -> dict:
        """
        Retrieves and optionally normalizes time series data for relevant variables.

        This method processes the time series (forcing) data for each basin and applies normalization as specified.
        It uses a transformation function to normalize the data based on the given statistical dictionary and variable list.

        Parameters:
        - to_norm: Boolean flag indicating whether to normalize the data (default is True).

        Process:
        1. Iterates through each basin in the time series data.
        2. Applies the normalization transformation (`_trans_norm`) to the data for each basin.
        This includes both standard and logarithmic normalization, based on the variable types and configuration settings.
        3. Updates the data dictionary with the transformed data for each basin.

        Returns:
        A dictionary where keys are basin IDs, and values are the corresponding xarray Datasets of processed (and optionally normalized) time series data for each basin.

        This method is crucial for preparing time series data for hydrological modeling, ensuring consistency and comparability
        across different basins and variables.
        """
        stat_dict = self.stat_dict
        var_list = self.data_cfgs["relevant_cols"][0]
        data = self.data_forcing
        for id, basin in data.items():
            basin = _trans_norm(
                basin,
                var_list,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=to_norm,
            )
            data[id] = basin
        return data

    def get_data_gfs(self, to_norm=True) -> dict:
        """
        Retrieves and optionally normalizes Global Forecast System (GFS) data for each basin.

        This method processes the GFS data, which is a type of forcing data, for each basin. It applies normalization transformations based on specified statistical parameters and variable lists. The method is designed to handle GFS data variables, preparing them for use in hydrological modeling.

        Parameters:
        - to_norm: Boolean flag indicating whether to normalize the data (default is True).

        Process:
        1. Iterates through each basin in the GFS data.
        2. For each basin, applies the normalization transformation (`_trans_norm`) to the GFS data.
        This transformation is based on the statistical dictionary, variable list, and logarithmic normalization settings.
        3. Updates the GFS data dictionary with the transformed data for each basin.

        Returns:
        A dictionary where keys are basin IDs, and values are xarray Datasets of processed (and optionally normalized) GFS data for each basin.

        This method ensures that GFS data is appropriately scaled and normalized for integration into hydrological models, enhancing the consistency and accuracy of the modeling process.
        """
        stat_dict = self.stat_dict
        var_list = self.data_cfgs["relevant_cols"][1]
        data = self.data_gfs
        for id, basin in data.items():
            basin = _trans_norm(
                basin,
                var_list,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=to_norm,
            )
            data[id] = basin
        return data

    def get_data_soil(self, to_norm=True) -> dict:
        """
        Retrieves and optionally normalizes soil attributes data for each basin.

        This method processes the soil attributes data, which is a type of forcing data, for each basin. It applies normalization transformations based on specified statistical parameters and variable lists. The method is designed to handle soil attributes data variables, preparing them for use in hydrological modeling.

        Parameters:
        - to_norm: Boolean flag indicating whether to normalize the data (default is True).

        Process:
        1. Iterates through each basin in the soil attributes data.
        2. For each basin, applies the normalization transformation (`_trans_norm`) to the soil attributes data.
        This transformation is based on the statistical dictionary, variable list, and logarithmic normalization settings.
        3. Updates the soil attributes data dictionary with the transformed data for each basin.

        Returns:
        A dictionary where keys are basin IDs, and values are xarray Datasets of processed (and optionally normalized) soil attributes data for each basin.

        This method ensures that soil attributes data is appropriately scaled and normalized for integration into hydrological models, enhancing the consistency and accuracy of the modeling process.
        """
        stat_dict = self.stat_dict
        var_list = self.data_cfgs["relevant_cols"][2]
        data = self.data_soil
        for id, basin in data.items():
            basin = _trans_norm(
                basin,
                var_list,
                stat_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=to_norm,
            )
            data[id] = basin
        return data

    def get_data_const(self, to_norm=True) -> np.array:
        """
        Retrieves and optionally normalizes constant attribute data.

        This method processes constant attribute data, which typically includes static features or characteristics of the basins. It applies normalization based on the statistical parameters provided, ensuring that the constant data is prepared for use in hydrological modeling.

        Parameters:
        - to_norm: Boolean flag indicating whether to normalize the data (default is True).

        Process:
        1. Retrieves the statistical dictionary and list of constant variables from the class attributes.
        2. Applies the normalization transformation (`_trans_norm`) to the constant attribute data.
        This transformation is guided by the statistical dictionary and variable list, and it's based on the `to_norm` parameter.
        3. Returns the processed data, which is either normalized or kept in its original scale based on the `to_norm` flag.

        Returns:
        An xarray Dataset or NumPy array (depending on the data structure) containing the processed constant attribute data.

        The normalization of constant data is important for maintaining consistency across various datasets in hydrological models, especially when these attributes are used in conjunction with other time-varying datasets.
        """
        stat_dict = self.stat_dict
        var_lst = self.data_cfgs["constant_cols"]
        data = self.data_attr
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        return data

    def load_data(self):
        """
        Loads and consolidates various types of data for hydrological modeling.

        This method is a comprehensive function that calls other methods to retrieve and process different datasets, including time series data, observation data, constant attributes, and GFS data. It organizes these datasets into a structured format suitable for use in modeling processes.

        Process:
        1. Retrieves time series data (forcing data) using the `get_data_ts` method.
        2. Retrieves observation data (target variables) using the `get_data_obs` method.
        3. Retrieves constant attribute data using the `get_data_const` method, if constant columns are specified in the configuration.
        4. Retrieves GFS data using the `get_data_gfs` method, if relevant GFS columns are specified.
        5. Retrieves soil attributes data using the 'get_data_soil' method, if relevant soil columns are specified.

        Returns:
        A tuple containing:
        - x: Time series data for each basin.
        - y: Observation data for target variables.
        - c: Constant attribute data for each basin (or None if not applicable).
        - g: GFS data for each basin (or None if not relevant).
        - s: soil attributes data for each basin (or None if not revevant)

        This method simplifies the process of data preparation by providing a single entry point to gather all necessary datasets for hydrological modeling, ensuring that they are properly normalized and ready for analysis.
        """
        x = self.get_data_ts()
        y = self.get_data_obs()
        c = self.get_data_const() if self.data_cfgs["constant_cols"] else None
        g = (
            self.get_data_gfs()
            if self.data_cfgs["relevant_cols"][1] != ["None"]
            else None
        )
        s = (
            self.get_data_soil()
            if self.data_cfgs["relevant_cols"][2] != ["None"]
            else None
        )
        return x, y, c, g, s

    def inverse_transform(self, target_values):
        """
        Applies the inverse transformation to the target values to convert them back to their original scale.

        This method is used to reverse the normalization applied to the target data, making the predictions interpretable in their original context. It is particularly useful for transforming model output back to its original scale after prediction.

        Parameters:
        - target_values: xarray Dataset or NumPy array containing the normalized target values to be inverse transformed.

        Process:
        1. Checks if PBM normalization is used. If so, keeps the predictions as is.
        2. Otherwise, applies the inverse normalization transformation using `_trans_norm` to the target values.
        3. For precipitation-related variables, further processes the data using precipitation normalization methods.
        4. Updates the attributes of the predictions to match the original target data attributes.
        5. Converts the xarray Dataset to a Pint-quantified dataset for unit handling.

        Returns:
        An xarray Dataset containing the inverse transformed data, scaled back to its original units and dimensions.

        This method is essential for interpreting model predictions in real-world units and scales, especially in applications where understanding the absolute scale of predictions is crucial.
        """
        star_dict = self.stat_dict
        target_cols = self.data_cfgs["target_cols"]
        if self.pbm_norm:
            pred = target_values
        else:
            pred = _trans_norm(
                target_values,
                target_cols,
                star_dict,
                log_norm_cols=self.log_norm_cols,
                to_norm=False,
            )
            for i in range(len(self.data_cfgs["target_cols"])):
                var = self.data_cfgs["target_cols"][i]
                if var in self.prcp_norm_cols:
                    mean_prep = self.basin_data.read_MP(
                        self.t_s_dict["sites_id"],
                        self.data_cfgs["attributes_path"],
                        self.data_cfgs["user"],
                    )

                    pred.loc[dict(variable=var)] = self.GPM_GFS_prcp_norm(
                        pred.sel(variable=var).to_numpy(),
                        mean_prep.to_numpy(),
                        to_norm=False,
                    )

            pred.attrs.update(self.data_target.attrs)
            pred_ds = pred.to_dataset(dim="variable")
            pred_ds = pred_ds.pint.quantify(pred_ds.attrs["units"])
            return pred_ds

    def GPM_GFS_cal_stat_prcp_norm(self, x, meanprep):
        """
        Calculates statistical parameters for precipitation-normalized data.

        This method is designed to process and normalize precipitation data by adjusting it relative to a mean precipitation value. It is useful for standardizing precipitation measurements across different locations or time periods, making them comparable.

        Parameters:
        - x: An array of observed precipitation values that need to be normalized.
        - meanprep: An array of mean precipitation values used for normalization.

        Process:
        1. Creates a temporary array (`tempprep`) by replicating the mean precipitation values to match the dimensions of `x`.
        2. Normalizes the precipitation values (`x`) by dividing them by the corresponding mean precipitation values (`tempprep`).
        3. Calculates and returns statistical parameters for the normalized data using the `cal_stat_gamma` function.

        Returns:
        Statistical parameters of the normalized precipitation data, useful for understanding the distribution and variation of precipitation after normalization.

        This method is a part of the data preprocessing steps in hydrological models, ensuring that precipitation data from different sources or time periods is normalized for consistency and comparability.
        """
        tempprep = np.tile(meanprep, (x.shape[0], 1))
        flowua = x / tempprep
        return cal_stat_gamma(flowua)

    def GPM_GFS_cal_stat_gamma(self, x, var):
        """
        Calculates statistical parameters for data using a gamma distribution normalization approach.

        This method processes data from multiple basins, normalizes it using a transformation suitable for gamma distributions, and then calculates statistical indicators. It is particularly useful for variables where the data distribution is skewed or where a gamma distribution is a good fit.

        Parameters:
        - x: A dictionary or collection of data arrays from multiple basins.
        - var: The specific variable for which the statistics are to be calculated.

        Process:
        1. Iterates through each basin's data, selecting and flattening the array for the specified variable.
        2. Removes NaN values and combines data from all basins into a single array.
        3. Applies a logarithmic transformation suitable for gamma distributions to the combined data.
        4. Removes any NaN values that may arise post-transformation.
        5. Calculates and returns statistical indicators using the `cal_4_stat_inds` function.

        Returns:
        A set of statistical indicators (like mean, variance, etc.) for the transformed data, which are essential for understanding the normalized distribution of the specified variable.

        This method is a key component of data preprocessing in hydrological models, particularly for variables that are better represented by a gamma distribution, enhancing the robustness and accuracy of the modeling process.
        """
        combined = np.array([])
        for basin in x.values():
            a = basin.sel(variable=var).to_numpy()
            a = a.flatten()
            a = a[~np.isnan(a)]
            combined = np.hstack((combined, a))
        b = np.log10(np.sqrt(combined) + 0.1)
        b = b[~np.isnan(b)]
        return cal_4_stat_inds(b)

    def GPM_GFS_prcp_norm(
        self, x: np.array, mean_prep: np.array, to_norm: bool
    ) -> np.array:
        """
        Applies or reverses precipitation normalization to the given data array.

        This method normalizes or denormalizes precipitation data based on mean precipitation values. Normalization is applied by dividing the original data by the mean precipitation values, and denormalization is done by multiplying the normalized data by the mean precipitation values.

        Parameters:
        - x: A NumPy array containing precipitation data to be normalized or denormalized.
        - mean_prep: A NumPy array of mean precipitation values used for normalization or denormalization.
        - to_norm: A boolean flag indicating whether to normalize (True) or denormalize (False) the data.

        Returns:
        A NumPy array of either normalized or denormalized precipitation data, depending on the value of 'to_norm'.

        The normalization process is critical for adjusting precipitation data to a common scale, especially when combining or comparing data across different regions or time periods. Conversely, denormalization is useful for converting the data back to its original scale, particularly for interpretation or presentation purposes.
        """
        tempprep = np.tile(mean_prep, (x.shape[0], 1))
        return x / tempprep if to_norm else x * tempprep


class Muti_Basin_GPM_GFS_SCALER(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: dict,
        constant_vars: np.array = None,
        data_gfs: dict = None,
        data_soil: dict = None,
        data_cfgs: dict = None,
        is_tra_val_te: str = None,
        basin_data: object = None,
        **kwargs,
    ):
        """
        A class for handling the scaling and normalization of data across multiple basins using the GPM and GFS datasets.

        This class is designed to initialize and manage a specific scaler (e.g., GPM_GFS_Scaler) for multiple basin datasets.
        It configures the scaler based on provided data configurations and parameters, and applies the scaling operations to the
        target variables, relevant variables, and constant variables of the data.

        Attributes:
        - data_cfgs: Configuration dictionary specifying scaler settings and parameters.
        - x, y, c, g, s: Scaled and normalized data for time series (x), observations (y), constant attributes (c), GFS data (g), and soil attributes data.
        - target_scaler: An instance of the scaler class (e.g., GPM_GFS_Scaler_2) used for data normalization and scaling.

        Methods:
        - __init__: Initializes the scaler with the provided data and configurations.

        The class supports different types of data scaling and normalization, determined by the 'scaler' type specified in the
        data configurations. It is tailored for hydrological modeling applications involving multiple basins and complex datasets.

        Usage:
        - Initialize the class with target variables, relevant variables, optional constant variables, optional GFS data,
        data configurations, and the train/validation/test flag.
        - Access the normalized data through the class attributes.
        """

        self.data_cfgs = data_cfgs
        scaler_type = data_cfgs["scaler"]
        if scaler_type == "GPM_GFS_Scaler":
            gamma_norm_cols = data_cfgs["scaler_params"]["gamma_norm_cols"]
            prcp_norm_cols = data_cfgs["scaler_params"]["prcp_norm_cols"]
            pbm_norm = data_cfgs["scaler_params"]["pbm_norm"]
            scaler = GPM_GFS_Scaler(
                target_vars=target_vars,
                relevant_vars=relevant_vars,
                constant_vars=constant_vars,
                data_gfs=data_gfs,
                data_soil=data_soil,
                data_cfgs=data_cfgs,
                is_tra_val_te=is_tra_val_te,
                prcp_norm_cols=prcp_norm_cols,
                gamma_norm_cols=gamma_norm_cols,
                pbm_norm=pbm_norm,
                basin_data=basin_data,
            )
            self.x, self.y, self.c, self.g, self.s = scaler.load_data()
            self.target_scaler = scaler
        print("Finish Normalization\n")


# Todo
class Muti_Basin_Batch_Loading_SCALER(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: dict,
        data_cfgs: dict = None,
        minio_cfgs: dict = None,
        t_s_dict: dict = None,
        is_tra_val_te: str = None,
        **kwargs,
    ):
        self.data_cfgs = data_cfgs
        scaler_type = data_cfgs["scaler"]
        if scaler_type == "GPM_GFS_Scaler":
            gamma_norm_cols = data_cfgs["scaler_params"]["gamma_norm_cols"]
            prcp_norm_cols = data_cfgs["scaler_params"]["prcp_norm_cols"]
            scaler = GPM_GFS_Scaler_3(
                target_vars=target_vars,
                relevant_vars=relevant_vars,
                data_cfgs=data_cfgs,
                minio_cfgs=minio_cfgs,
                t_s_dict=t_s_dict,
                is_tra_val_te=is_tra_val_te,
                prcp_norm_cols=prcp_norm_cols,
                gamma_norm_cols=gamma_norm_cols,
            )
        print("Finish Normalization\n")


# Todo
class GPM_GFS_Scaler_3(object):
    def __init__(
        self,
        target_vars: np.array,
        relevant_vars: dict,
        data_cfgs: dict,
        minio_cfgs: dict,
        t_s_dict: dict,
        is_tra_val_te: str,
        prcp_norm_cols=None,
        gamma_norm_cols=None,
    ):
        prcp_norm_cols = [
            "waterlevel",
            "streamflow",
        ]

        gamma_norm_cols = [
            "tp",
        ]
        self.t_s_dict = t_s_dict
        self.data_target = target_vars
        self.data_forcing = relevant_vars
        self.data_cfgs = data_cfgs
        self.minio_cfgs = minio_cfgs
        self.prcp_norm_cols = prcp_norm_cols
        self.gamma_norm_cols = gamma_norm_cols
        self.log_norm_cols = gamma_norm_cols + prcp_norm_cols
        if is_tra_val_te == "train" and data_cfgs["stat_dict_file"] is None:
            self.stat_dict = self.cal_stat_all()

    def cal_stat_all(self):
        stat_dict = {}

        x = self.data_forcing
        forcing_lst = self.data_cfgs["relevant_cols"]
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var in self.gamma_norm_cols:
                stat_dict[var] = self.GPM_GFS_cal_stat_gamma(x, var)

        y = self.data_target
        target_cols = self.data_cfgs["target_cols"]
        for k in range(len(target_cols)):
            var = target_cols[k]
            if var in self.prcp_norm_cols:
                meanprep = self.read_mean_prcp(self.t_s_dict["sites_id"])
                stat_dict[var] = self.GPM_GFS_cal_stat_prcp_norm(
                    y.sel(variable=var).to_numpy(),
                    meanprep.to_array().to_numpy(),
                )
        return stat_dict

    def cal_5_stat_inds(self, b):
        p10 = np.percentile(b, 10).astype(float)
        p90 = np.percentile(b, 90).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        if std < 0.001:
            std = 1
        return [p10, p90, mean, std, len(b)]

    def GPM_GFS_cal_stat_prcp_norm(self, y, meanprep):
        tempprep = np.tile(meanprep, (y.shape[0], 1))
        flowua = y / tempprep
        a = flowua.flatten()
        b = a[~np.isnan(a)]
        b = np.log10(np.sqrt(b) + 0.1)
        return self.cal_5_stat_inds(b)

    def GPM_GFS_cal_stat_gamma(self, x, var):
        combined = np.array([])
        for basin in x.values():
            a = basin.sel(variable=var).to_numpy()
            a = a.flatten()
            a = a[~np.isnan(a)]
            combined = np.hstack((combined, a))
        b = np.log10(np.sqrt(combined) + 0.1)
        b = b[~np.isnan(b)]
        return self.cal_5_stat_inds(b)

    def GPM_GFS_prcp_norm(
        self, x: np.array, mean_prep: np.array, to_norm: bool
    ) -> np.array:
        tempprep = np.tile(mean_prep, (x.shape[0], 1))
        return x / tempprep if to_norm else x * tempprep

    def read_mean_prcp(self, gage_id_lst):
        pass
