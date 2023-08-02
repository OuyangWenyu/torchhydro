"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-07-29 11:42:01
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: \HydroTL\hydrotl\models\time_model.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch import nn
import json
import os
from datetime import datetime
from hydrodataset import HydroDataset

from torchhydro.datasets.data_dict import dataloaders_dict
from torchhydro.models.model_dict_function import (
    pytorch_model_dict,
    pytorch_model_wrapper_dict,
)
from torchhydro.models.model_utils import get_the_device


class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations
    of models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train
    and validation at this point.
    """

    def __init__(self, model_base: str, data_source: HydroDataset, params: Dict):
        """
        Parameters
        ----------
        model_base
            name of the model
        data_source
            the digital twin of a data_source in reality
        params
            parameters for initializing the model
        """
        self.params = params
        if "weight_path" in params["model_params"]:
            self.model = self.load_model(
                model_base,
                params["model_params"],
                params["model_params"]["weight_path"],
            )
        else:
            self.model = self.load_model(model_base, params["model_params"])
        self.training = self.make_data_load(data_source, params["data_params"], "train")
        if params["data_params"]["t_range_valid"] is not None:
            self.validation = self.make_data_load(
                data_source, params["data_params"], "valid"
            )
        self.test_data = self.make_data_load(data_source, params["data_params"], "test")

    @abstractmethod
    def load_model(
        self, model_base: str, model_params: Dict, weight_path=None
    ) -> object:
        """
        Get a time series forecast model and it varies based on the underlying framework used

        Parameters
        ----------
        model_base
            name of the model
        model_params
            model parameters
        weight_path
            where we put model's weights

        Returns
        -------
        object
            a time series forecast model
        """
        raise NotImplementedError

    @abstractmethod
    def make_data_load(
        self, data_source: HydroDataset, params: Dict, loader_type: str
    ) -> object:
        """
        Initializes a data loader based on the provided data_source.

        Parameters
        ----------
        data_source
            a class for a given data source
        params
            parameters for loading data source
        loader_type
            train or valid or test

        Returns
        -------
        object
            a class for loading data from data source
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_path: str, epoch: int):
        """
        Saves a model to a specific path along with a configuration report
        of the parameters and data info.
        """
        raise NotImplementedError


class PyTorchForecast(TimeSeriesModel):
    """
    The class for PyTorch time series forecast model
    """

    def __init__(
        self, model_base: str, data_source_model: HydroDataset, params_dict: Dict
    ):
        """
        Parameters
        ----------
        model_base
            name of model we gonna use; chosen from pytorch_model_dict in model_dict_function.py
        data_source_model
            data source where we read data from
        params_dict
            parameters set for the model
        """
        self.device_num = params_dict["training_params"]["device"]
        self.device = get_the_device(self.device_num)
        super().__init__(model_base, data_source_model, params_dict)
        print(f"Torch is using {str(self.device)}")

    def load_model(
        self, model_base: str, model_params: Dict, weight_path: str = None, strict=True
    ):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Parameters
        ----------
        model_base
            name of the model
        model_params
            model parameters
        weight_path
            where we put model's weights
        strict
            whether to strictly enforce that the keys in 'state_dict` match the keys returned by this module's
            'torch.nn.Module.state_dict` function; its default: ``True``
        Returns
        -------
        object
            model in pytorch_model_dict in model_dict_function.py
        """
        if model_base not in pytorch_model_dict:
            raise Exception(
                f"Error the model {model_base} was not found in the model dict. Please add it."
            )
        model = pytorch_model_dict[model_base](**model_params["model_param"])
        if weight_path is not None:
            # if the model has been trained
            strict = False
            checkpoint = torch.load(weight_path, map_location=self.device)
            if "weight_path_add" in model_params:
                if "excluded_layers" in model_params["weight_path_add"]:
                    # delete some layers from source model if we don't need them
                    excluded_layers = model_params["weight_path_add"]["excluded_layers"]
                    for layer in excluded_layers:
                        del checkpoint[layer]
                    print("sucessfully deleted layers")
                else:
                    print("directly loading identically-named layers of source model")
            if "tl_tag" in model.__dict__ and model.tl_tag:
                # it means target model's structure is different with source model's
                # when model.tl_tag is true.
                # our transfer learning model now only support one whole part -- tl_part
                model.tl_part.load_state_dict(checkpoint, strict=strict)
            else:
                # directly load model's weights
                model.load_state_dict(checkpoint, strict=strict)
            print("Weights sucessfully loaded")
        if torch.cuda.device_count() > 1 and len(self.device_num) > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            which_first_tensor = self.params["training_params"]["which_first_tensor"]
            sequece_first = which_first_tensor == "sequence"
            parallel_dim = 1 if sequece_first else 0
            model = nn.DataParallel(model, device_ids=self.device_num, dim=parallel_dim)
        model.to(self.device)
        if (
            weight_path is not None
            and "weight_path_add" in model_params
            and "freeze_params" in model_params["weight_path_add"]
        ):
            freeze_params = model_params["weight_path_add"]["freeze_params"]
            for param in freeze_params:
                if "tl_tag" in model.__dict__:
                    exec(f"model.tl_part.{param}.requires_grad = False")
                else:
                    exec(f"model.{param}.requires_grad = False")
        if ("model_wrapper" in list(model_params.keys())) and (
            model_params["model_wrapper"] is not None
        ):
            wrapper_name = model_params["model_wrapper"]
            wrapper_params = model_params["model_wrapper_param"]
            model = pytorch_model_wrapper_dict[wrapper_name](model, **wrapper_params)
        return model

    def save_model(self, final_path: str, epoch: int) -> None:
        """
        Function to save a model to a given file path

        Parameters
        ----------
        final_path
            where we save the model
        epoch
            the epoch when we save the model

        Returns
        -------
        None
        """
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_name = f"{time_stamp}_model.pth"
        params_name = f"{time_stamp}.json"
        model_save_path = os.path.join(final_path, model_name)
        params_save_path = os.path.join(final_path, f"{time_stamp}.json")
        if torch.cuda.device_count() > 1 and len(self.device_num) > 1:
            torch.save(self.model.module.state_dict(), model_save_path)
        else:
            torch.save(self.model.state_dict(), model_save_path)
        with open(params_save_path, "w+") as p:
            json.dump(self.params, p)

    def make_data_load(
        self, data_source_model: HydroDataset, data_params: Dict, loader_type: str
    ):
        """
        Initializes a data loader based on the provided data_source.

        Parameters
        ----------
        data_source_model
            the model for reading data from data source
        data_params
            parameters for loading data
        loader_type
            train or valid or test

        Returns
        -------
        object
            an object initializing from class in dataloaders_dict in data_dict.py
        """
        the_loader = data_params["data_loader"]
        if the_loader in list(dataloaders_dict.keys()):
            loader = dataloaders_dict[the_loader](
                data_source_model, data_params, loader_type
            )
        else:
            raise Exception(
                "Error the data model "
                + str(data_source_model)
                + " was not found in the model dict. Please add it."
            )
        return loader
