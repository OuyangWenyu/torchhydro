"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-09-24 21:01:22
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: \torchhydro\torchhydro\trainers\deep_hydro.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict
import numpy as np
import torch
from torch import nn
import json
import os
from datetime import datetime
from hydrodataset import HydroDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchhydro.datasets.data_sets import KuaiSampler

from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.models.model_dict_function import (
    pytorch_criterion_dict,
    pytorch_model_dict,
    pytorch_model_wrapper_dict,
    pytorch_opt_dict,
)
from torchhydro.models.model_utils import get_the_device
from torchhydro.trainers.train_utils import (
    EarlyStopper,
    evaluate_validation,
    compute_validation,
    torch_single_train,
)
from torchhydro.trainers.train_logger import TrainLogger


class DeepHydroInterface(ABC):
    """
    An abstract class used to handle different configurations
    of hydrological deep learning models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train and validation at this point.
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
        self.training = self.make_dataset(data_source, params["data_params"], "train")
        if params["data_params"]["t_range_valid"] is not None:
            self.validation = self.make_dataset(
                data_source, params["data_params"], "valid"
            )
        self.test_data = self.make_dataset(data_source, params["data_params"], "test")

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
    def make_dataset(
        self, data_source: HydroDataset, params: Dict, loader_type: str
    ) -> object:
        """
        Initializes a pytorch dataset based on the provided data_source.

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
            a dataset class loading data from data source
        """
        raise NotImplementedError

    @abstractmethod
    def model_train(self):
        """
        Train the model
        """
        raise NotImplementedError


class DeepHydro(DeepHydroInterface):
    """
    The Base Trainer class for Hydrological Deep Learning models
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
            raise NotImplementedError(
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

    def make_dataset(
        self, data_source_model: HydroDataset, data_params: Dict, loader_type: str
    ):
        """
        Initializes a pytorch dataset based on the provided data_source.

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
            an object initializing from class in datasets_dict in data_dict.py
        """
        dataset = data_params["dataset"]
        if dataset in list(datasets_dict.keys()):
            loader = datasets_dict[dataset](data_source_model, data_params, loader_type)
        else:
            raise NotImplementedError(
                "Error the dataset "
                + str(dataset)
                + " was not found in the dataset dict. Please add it."
            )
        return loader

    def model_train(self) -> None:
        """train a hydrological DL model"""
        # A dictionary of the necessary parameters for training
        training_params = self.params["training_params"]
        # The file path to load model weights from; defaults to "model_save"
        model_filepath = self.params["data_params"]["test_path"]
        data_params = self.params["data_params"]
        es = None
        if "early_stopping" in self.params:
            es = EarlyStopper(self.params["early_stopping"]["patience"])
        criterion = self._get_loss_func(training_params)
        opt = self._get_optimizer(training_params)
        lr_scheduler = training_params["lr_scheduler"]
        max_epochs = training_params["epochs"]
        start_epoch = training_params["start_epoch"]
        # use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader, validation_data_loader = self._get_dataloader(
            training_params, data_params
        )
        logger = TrainLogger(model_filepath, self.params, opt)
        for epoch in range(start_epoch, max_epochs + 1):
            with logger.log_epoch_train(epoch) as train_logs:
                if lr_scheduler is not None and epoch in lr_scheduler.keys():
                    # now we only support manual setting lr scheduler
                    for param_group in opt.param_groups:
                        param_group["lr"] = lr_scheduler[epoch]
                total_loss, n_iter_ep = torch_single_train(
                    self.model,
                    opt,
                    criterion,
                    data_loader,
                    device=self.device,
                    which_first_tensor=training_params["which_first_tensor"],
                )
                train_logs["train_loss"] = total_loss
                train_logs["model"] = self.model

            valid_loss = None
            valid_metrics = None
            if data_params["t_range_valid"] is not None:
                with logger.log_epoch_valid(epoch) as valid_logs:
                    valid_obss_np, valid_preds_np, valid_loss = compute_validation(
                        self.model,
                        criterion,
                        validation_data_loader,
                        device=self.device,
                        which_first_tensor=training_params["which_first_tensor"],
                    )
                    evaluation_metrics = self.params["evaluate_params"]["metrics"]
                    fill_nan = self.params["evaluate_params"]["fill_nan"]
                    target_col = self.params["data_params"]["target_cols"]
                    valid_metrics = evaluate_validation(
                        validation_data_loader,
                        valid_preds_np,
                        valid_obss_np,
                        evaluation_metrics,
                        fill_nan,
                        target_col,
                    )
                    valid_logs["valid_loss"] = valid_loss
                    valid_logs["valid_metrics"] = valid_metrics
            logger.save_session_param(
                epoch, total_loss, n_iter_ep, valid_loss, valid_metrics
            )
            logger.save_model_and_params(self.model, epoch, self.params)
            if es and not es.check_loss(self.model, valid_loss):
                print("Stopping model now")
                self.model.load_state_dict(torch.load("checkpoint.pth"))
                break

        logger.tb.close()

    def _get_optimizer(self, training_params):
        params_in_opt = self.model.parameters()
        return pytorch_opt_dict[training_params["optimizer"]](
            params_in_opt, **training_params["optim_params"]
        )

    def _get_loss_func(self, training_params):
        criterion_init_params = {}
        if "criterion_params" in training_params:
            loss_param = training_params["criterion_params"]
            if loss_param is not None:
                for key in loss_param.keys():
                    if key == "loss_funcs":
                        criterion_init_params[key] = pytorch_criterion_dict[
                            loss_param[key]
                        ]()
                    else:
                        criterion_init_params[key] = loss_param[key]
        return pytorch_criterion_dict[training_params["criterion"]](
            **criterion_init_params
        )

    def _get_dataloader(self, training_params, data_params):
        worker_num = 0
        pin_memory = False
        if "num_workers" in training_params:
            worker_num = training_params["num_workers"]
            print(f"using {str(worker_num)} workers")
        if "pin_memory" in training_params:
            pin_memory = training_params["pin_memory"]
            print(f"Pin memory set to {str(pin_memory)}")
        train_dataset = self.training
        sampler = None
        if data_params["sampler"] is not None:
            # now we only have one special sampler from Kuai Fang's Deep Learning papers
            batch_size = data_params["batch_size"]
            rho = data_params["forecast_history"]
            warmup_length = data_params["warmup_length"]
            ngrid = train_dataset.y.basin.size
            nt = train_dataset.y.time.size
            sampler = KuaiSampler(
                train_dataset,
                batch_size=batch_size,
                warmup_length=warmup_length,
                rho=rho,
                ngrid=ngrid,
                nt=nt,
            )
        data_loader = DataLoader(
            train_dataset,
            batch_size=training_params["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=worker_num,
            pin_memory=pin_memory,
            timeout=0,
        )
        if data_params["t_range_valid"] is not None:
            valid_dataset = self.validation
            validation_data_loader = DataLoader(
                valid_dataset,
                batch_size=training_params["batch_size"],
                shuffle=False,
                num_workers=worker_num,
                pin_memory=pin_memory,
                timeout=0,
            )

        return data_loader, validation_data_loader


class FedLearnHydro(DeepHydro):
    """Federated Learning Hydrological DL model"""

    def __init__(
        self, model_base: str, data_source_model: HydroDataset, params_dict: Dict
    ):
        super().__init__(model_base, data_source_model, params_dict)

    def model_train(self) -> None:
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, momentum=0.5
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-4
            )

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
