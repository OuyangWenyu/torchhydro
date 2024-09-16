"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:15:48
LastEditTime: 2024-09-16 10:19:34
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: \torchhydro\torchhydro\trainers\deep_hydro.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import copy
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from torchhydro.configs.config import update_nested_dict
from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.sampler import (
    KuaiSampler,
    fl_sample_basin,
    fl_sample_region,
    HydroSampler,
)
from torchhydro.models.model_dict_function import (
    pytorch_criterion_dict,
    pytorch_model_dict,
    pytorch_opt_dict,
)
from torchhydro.models.model_utils import get_the_device
from torchhydro.trainers.train_logger import TrainLogger
from torchhydro.trainers.train_utils import (
    EarlyStopper,
    average_weights,
    denormalize4eval,
    evaluate_validation,
    compute_validation,
    model_infer,
    read_pth_from_model_loader,
    torch_single_train,
    calculate_and_record_metrics,
)


class DeepHydroInterface(ABC):
    """
    An abstract class used to handle different configurations
    of hydrological deep learning models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train and validation at this point.
    """

    def __init__(self, cfgs: Dict):
        """
        Parameters
        ----------
        cfgs
            configs for initializing DeepHydro
        """

        self._cfgs = cfgs

    @property
    def cfgs(self):
        """all configs"""
        return self._cfgs

    @property
    def weight_path(self):
        """weight path"""
        return self._cfgs["model_cfgs"]["weight_path"]

    @weight_path.setter
    def weight_path(self, weight_path):
        self._cfgs["model_cfgs"]["weight_path"] = weight_path

    @abstractmethod
    def load_model(self, mode="train") -> object:
        """Get a Hydro DL model"""
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self, is_tra_val_te: str) -> object:
        """
        Initializes a pytorch dataset.

        Parameters
        ----------
        is_tra_val_te
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

    @abstractmethod
    def model_evaluate(self):
        """
        Evaluate the model
        """
        raise NotImplementedError


class DeepHydro(DeepHydroInterface):
    """
    The Base Trainer class for Hydrological Deep Learning models
    """

    def __init__(
        self,
        cfgs: Dict,
        pre_model=None,
    ):
        """
        Parameters
        ----------
        cfgs
            configs for the model
        pre_model
            a pre-trained model, if it is not None,
            we will use its weights to initialize this model
            by default None
        """
        super().__init__(cfgs)
        self.device_num = cfgs["training_cfgs"]["device"]
        self.device = get_the_device(self.device_num)
        self.pre_model = pre_model
        self.model = self.load_model()
        if cfgs["training_cfgs"]["train_mode"]:
            self.traindataset = self.make_dataset("train")
            if cfgs["data_cfgs"]["t_range_valid"] is not None:
                self.validdataset = self.make_dataset("valid")
        self.testdataset: BaseDataset = self.make_dataset("test")
        print(f"Torch is using {str(self.device)}")

    def load_model(self, mode="train"):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Returns
        -------
        object
            model in pytorch_model_dict in model_dict_function.py
        """
        if mode == "infer":
            self.weight_path = self._get_trained_model()
        elif mode != "train":
            raise ValueError("Invalid mode; must be 'train' or 'infer'")
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        if model_name not in pytorch_model_dict:
            raise NotImplementedError(
                f"Error the model {model_name} was not found in the model dict. Please add it."
            )
        if self.pre_model is not None:
            model = self._load_pretrain_model()
        elif self.weight_path is not None:
            # load model from pth file (saved weights and biases)
            model = self._load_model_from_pth()
        else:
            model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
            # model_data = torch.load(weight_path)
            # model.load_state_dict(model_data)
        if torch.cuda.device_count() > 1 and len(self.device_num) > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            which_first_tensor = self.cfgs["training_cfgs"]["which_first_tensor"]
            sequece_first = which_first_tensor == "sequence"
            parallel_dim = 1 if sequece_first else 0
            model = nn.DataParallel(model, device_ids=self.device_num, dim=parallel_dim)
        model.to(self.device)
        return model

    def _load_pretrain_model(self):
        """load a pretrained model as the initial model"""
        return self.pre_model

    def _load_model_from_pth(self):
        weight_path = self.weight_path
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
        checkpoint = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        print("Weights sucessfully loaded")
        return model

    def make_dataset(self, is_tra_val_te: str):
        """
        Initializes a pytorch dataset.

        Parameters
        ----------
        is_tra_val_te
            train or valid or test

        Returns
        -------
        object
            an object initializing from class in datasets_dict in data_dict.py
        """
        data_cfgs = self.cfgs["data_cfgs"]
        dataset_name = data_cfgs["dataset"]

        if dataset_name in list(datasets_dict.keys()):
            dataset = datasets_dict[dataset_name](data_cfgs, is_tra_val_te)
        else:
            raise NotImplementedError(
                f"Error the dataset {str(dataset_name)} was not found in the dataset dict. Please add it."
            )
        return dataset

    def model_train(self) -> None:
        """train a hydrological DL model"""
        # A dictionary of the necessary parameters for training
        training_cfgs = self.cfgs["training_cfgs"]
        # The file path to load model weights from; defaults to "model_save"
        model_filepath = self.cfgs["data_cfgs"]["test_path"]
        data_cfgs = self.cfgs["data_cfgs"]
        es = None
        if training_cfgs["early_stopping"]:
            es = EarlyStopper(training_cfgs["patience"])
        criterion = self._get_loss_func(training_cfgs)
        opt = self._get_optimizer(training_cfgs)
        scheduler = self._get_scheduler(training_cfgs, opt)
        max_epochs = training_cfgs["epochs"]
        start_epoch = training_cfgs["start_epoch"]
        # use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader, validation_data_loader = self._get_dataloader(
            training_cfgs, data_cfgs
        )
        logger = TrainLogger(model_filepath, self.cfgs, opt)
        for epoch in range(start_epoch, max_epochs + 1):
            with logger.log_epoch_train(epoch) as train_logs:
                total_loss, n_iter_ep = torch_single_train(
                    self.model,
                    opt,
                    criterion,
                    data_loader,
                    device=self.device,
                    which_first_tensor=training_cfgs["which_first_tensor"],
                )
                train_logs["train_loss"] = total_loss
                train_logs["model"] = self.model

            valid_loss = None
            valid_metrics = None
            if data_cfgs["t_range_valid"] is not None:
                with logger.log_epoch_valid(epoch) as valid_logs:
                    valid_loss, valid_metrics = self._1epoch_valid(
                        training_cfgs, criterion, validation_data_loader, valid_logs
                    )

            self._scheduler_step(training_cfgs, scheduler, valid_loss)
            logger.save_session_param(
                epoch, total_loss, n_iter_ep, valid_loss, valid_metrics
            )
            logger.save_model_and_params(self.model, epoch, self.cfgs)
            if es and not es.check_loss(
                self.model,
                valid_loss,
                self.cfgs["data_cfgs"]["test_path"],
            ):
                print("Stopping model now")
                break
        # logger.plot_model_structure(self.model)
        logger.tb.close()

        # return the trained model weights and bias and the epoch loss
        return self.model.state_dict(), sum(logger.epoch_loss) / len(logger.epoch_loss)

    def _get_scheduler(self, training_cfgs, opt):
        lr_scheduler_cfg = training_cfgs["lr_scheduler"]

        if "lr" in lr_scheduler_cfg and "lr_factor" not in lr_scheduler_cfg:
            scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
        elif isinstance(lr_scheduler_cfg, dict) and all(
            isinstance(epoch, int) for epoch in lr_scheduler_cfg
        ):
            scheduler = LambdaLR(
                opt, lr_lambda=lambda epoch: lr_scheduler_cfg.get(epoch, 1.0)
            )
        elif "lr_factor" in lr_scheduler_cfg and "lr_patience" not in lr_scheduler_cfg:
            scheduler = ExponentialLR(opt, gamma=lr_scheduler_cfg["lr_factor"])
        elif "lr_factor" in lr_scheduler_cfg:
            scheduler = ReduceLROnPlateau(
                opt,
                mode="min",
                factor=lr_scheduler_cfg["lr_factor"],
                patience=lr_scheduler_cfg["lr_patience"],
            )
        else:
            raise ValueError("Invalid lr_scheduler configuration")

        return scheduler

    def _scheduler_step(self, training_cfgs, scheduler, valid_loss):
        lr_scheduler_cfg = training_cfgs["lr_scheduler"]
        required_keys = {"lr_factor", "lr_patience"}
        if required_keys.issubset(lr_scheduler_cfg.keys()):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

    def _1epoch_valid(
        self, training_cfgs, criterion, validation_data_loader, valid_logs
    ):
        valid_obss_np, valid_preds_np, valid_loss = compute_validation(
            self.model,
            criterion,
            validation_data_loader,
            device=self.device,
            which_first_tensor=training_cfgs["which_first_tensor"],
        )
        valid_logs["valid_loss"] = valid_loss
        if self.cfgs["evaluation_cfgs"]["calc_metrics"]:
            target_col = self.cfgs["data_cfgs"]["target_cols"]
            valid_metrics = evaluate_validation(
                validation_data_loader,
                valid_preds_np,
                valid_obss_np,
                self.cfgs["evaluation_cfgs"],
                target_col,
            )
            valid_logs["valid_metrics"] = valid_metrics
            return valid_loss, valid_metrics
        return valid_loss, None

    def _get_trained_model(self):
        model_loader = self.cfgs["evaluation_cfgs"]["model_loader"]
        model_pth_dir = self.cfgs["data_cfgs"]["test_path"]
        return read_pth_from_model_loader(model_loader, model_pth_dir)

    def model_evaluate(self) -> Tuple[Dict, np.array, np.array]:
        """
        A function to evaluate a model, called at end of training.

        Returns
        -------
        tuple[dict, np.array, np.array]
            eval_log, denormalized predictions and observations
        """
        self.model = self.load_model(mode="infer")
        preds_xr, obss_xr = self.inference()
        return preds_xr, obss_xr

    def inference(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """infer using trained model and unnormalized results"""
        data_cfgs = self.cfgs["data_cfgs"]
        training_cfgs = self.cfgs["training_cfgs"]
        evaluation_cfgs = self.cfgs["evaluation_cfgs"]
        device = get_the_device(self.cfgs["training_cfgs"]["device"])

        ngrid = self.testdataset.ngrid
        if data_cfgs["sampler"] == "HydroSampler":
            test_num_samples = self.testdataset.num_samples
            test_dataloader = DataLoader(
                self.testdataset,
                batch_size=test_num_samples // ngrid,
                shuffle=False,
                drop_last=False,
                timeout=0,
            )
        else:
            test_dataloader = DataLoader(
                self.testdataset,
                batch_size=training_cfgs["batch_size"],
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
            )
        seq_first = training_cfgs["which_first_tensor"] == "sequence"
        self.model.eval()
        # here the batch is just an index of lookup table, so any batch size could be chosen
        test_preds = []
        obss = []
        with torch.no_grad():
            for xs, ys in test_dataloader:
                # here the a batch doesn't mean a basin; it is only an index in lookup table
                # for NtoN mode, only basin is index in lookup table, so the batch is same as basin
                # for Nto1 mode, batch is only an index
                ys, pred = model_infer(seq_first, device, self.model, xs, ys)
                test_preds.append(pred.cpu().numpy())
                obss.append(ys.cpu().numpy())
            pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
            obs = reduce(lambda x, y: np.vstack((x, y)), obss)
        if pred.ndim == 2:
            # TODO: check
            # the ndim is 2 meaning we use an Nto1 mode
            # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
            # params of reshape should be (basin size, time length)
            pred = pred.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
            obs = obs.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)

        if not evaluation_cfgs["long_seq_pred"]:
            target_len = len(data_cfgs["target_cols"])
            prec_window = data_cfgs["prec_window"]
            batch_size = test_dataloader.batch_size
            if evaluation_cfgs["rolling"]:
                forecast_length = data_cfgs["forecast_length"]
                pred = pred[:, prec_window:, :].reshape(
                    ngrid, batch_size, forecast_length, target_len
                )
                obs = obs[:, prec_window:, :].reshape(
                    ngrid, batch_size, forecast_length, target_len
                )

                pred = pred[:, ::forecast_length, :, :]
                obs = obs[:, ::forecast_length, :, :]

                pred = np.concatenate(pred, axis=0).reshape(ngrid, -1, target_len)
                obs = np.concatenate(obs, axis=0).reshape(ngrid, -1, target_len)

                pred = pred[:, :batch_size, :]
                obs = obs[:, :batch_size, :]
            else:
                pred = pred[:, prec_window, :].reshape(ngrid, batch_size, target_len)
                obs = obs[:, prec_window, :].reshape(ngrid, batch_size, target_len)
            pred_xr, obs_xr = denormalize4eval(
                test_dataloader, pred, obs, long_seq_pred=False
            )
            fill_nan = evaluation_cfgs["fill_nan"]
            eval_log = {}
            for i, col in enumerate(data_cfgs["target_cols"]):
                obs = obs_xr[col].to_numpy()
                pred = pred_xr[col].to_numpy()
                eval_log = calculate_and_record_metrics(
                    obs,
                    pred,
                    evaluation_cfgs["metrics"],
                    col,
                    fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                    eval_log,
                )
            test_log = f" Best Metric {eval_log}"
            print(test_log)
        else:
            pred_xr, obs_xr = denormalize4eval(test_dataloader, pred, obs)
        return pred_xr, obs_xr

    def _get_optimizer(self, training_cfgs):
        params_in_opt = self.model.parameters()
        return pytorch_opt_dict[training_cfgs["optimizer"]](
            params_in_opt, **training_cfgs["optim_params"]
        )

    def _get_loss_func(self, training_cfgs):
        criterion_init_params = {}
        if "criterion_params" in training_cfgs:
            loss_param = training_cfgs["criterion_params"]
            if loss_param is not None:
                for key in loss_param.keys():
                    if key == "loss_funcs":
                        criterion_init_params[key] = pytorch_criterion_dict[
                            loss_param[key]
                        ]()
                    else:
                        criterion_init_params[key] = loss_param[key]
        return pytorch_criterion_dict[training_cfgs["criterion"]](
            **criterion_init_params
        )

    def _get_dataloader(self, training_cfgs, data_cfgs):
        worker_num = 0
        pin_memory = False
        if "num_workers" in training_cfgs:
            worker_num = training_cfgs["num_workers"]
            print(f"using {str(worker_num)} workers")
        if "pin_memory" in training_cfgs:
            pin_memory = training_cfgs["pin_memory"]
            print(f"Pin memory set to {str(pin_memory)}")
        train_dataset: BaseDataset = self.traindataset
        sampler = None
        if data_cfgs["sampler"] is not None:
            # now we only have one special sampler from Kuai Fang's Deep Learning papers
            batch_size = data_cfgs["batch_size"]
            rho = data_cfgs["forecast_history"]
            warmup_length = data_cfgs["warmup_length"]
            horizon = data_cfgs["forecast_length"]
            ngrid = train_dataset.ngrid
            nt = train_dataset.nt
            if data_cfgs["sampler"] == "HydroSampler":
                sampler = HydroSampler(train_dataset)
            elif data_cfgs["sampler"] == "KuaiSampler":
                sampler = KuaiSampler(
                    train_dataset,
                    batch_size=batch_size,
                    warmup_length=warmup_length,
                    rho_horizon=rho + horizon,
                    ngrid=ngrid,
                    nt=nt,
                )
            elif data_cfgs["sampler"] == "DistSampler":
                sampler = DistributedSampler(train_dataset)
            else:
                raise NotImplementedError("This sampler not implemented yet")
        data_loader = DataLoader(
            train_dataset,
            batch_size=training_cfgs["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=worker_num,
            pin_memory=pin_memory,
            timeout=0,
        )
        if data_cfgs["t_range_valid"] is not None:
            valid_dataset: BaseDataset = self.validdataset
            batch_size_valid = training_cfgs["batch_size"]
            if data_cfgs["sampler"] == "HydroSampler":
                # for HydroSampler when evaluating, we need to set new batch size
                batch_size_valid = valid_dataset.num_samples // ngrid
            validation_data_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size_valid,
                shuffle=False,
                num_workers=worker_num,
                pin_memory=pin_memory,
                timeout=0,
            )
            return data_loader, validation_data_loader

        return data_loader, None


class FedLearnHydro(DeepHydro):
    """Federated Learning Hydrological DL model"""

    def __init__(self, cfgs: Dict):
        super().__init__(cfgs)
        # a user group which is a dict where the keys are the user index
        # and the values are the corresponding data for each of those users
        train_dataset = self.traindataset
        fl_hyperparam = self.cfgs["model_cfgs"]["fl_hyperparam"]
        # sample training data amongst users
        if fl_hyperparam["fl_sample"] == "basin":
            # Sample a basin for a user
            user_groups = fl_sample_basin(train_dataset)
        elif fl_hyperparam["fl_sample"] == "region":
            # Sample a region for a user
            user_groups = fl_sample_region(train_dataset)
        else:
            raise NotImplementedError()
        self.user_groups = user_groups

    @property
    def num_users(self):
        """number of users in federated learning"""
        return len(self.user_groups)

    def model_train(self) -> None:
        # BUILD MODEL
        global_model = self.model

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        print_every = 2

        training_cfgs = self.cfgs["training_cfgs"]
        model_cfgs = self.cfgs["model_cfgs"]
        max_epochs = training_cfgs["epochs"]
        start_epoch = training_cfgs["start_epoch"]
        fl_hyperparam = model_cfgs["fl_hyperparam"]
        # total rounds in a FL system is max_epochs
        for epoch in tqdm(range(start_epoch, max_epochs + 1)):
            local_weights, local_losses = [], []
            print(f"\n | Global Training Round : {epoch} |\n")

            global_model.train()
            m = max(int(fl_hyperparam["fl_frac"] * self.num_users), 1)
            # randomly select m users, they will be the clients in this round
            idxs_users = np.random.choice(range(self.num_users), m, replace=False)

            for idx in idxs_users:
                # each user will be used to train the model locally
                # user_gourps[idx] means the idx of dataset for a user
                user_cfgs = self._get_a_user_cfgs(idx)
                local_model = DeepHydro(
                    user_cfgs,
                    pre_model=copy.deepcopy(global_model),
                )
                w, loss = local_model.model_train()
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc = []
            global_model.eval()
            for c in range(self.num_users):
                one_user_cfg = self._get_a_user_cfgs(c)
                local_model = DeepHydro(
                    one_user_cfg,
                    pre_model=global_model,
                )
                acc, _, _ = local_model.model_evaluate()
                list_acc.append(acc)
            values = [list(d.values())[0][0] for d in list_acc]
            filtered_values = [v for v in values if not np.isnan(v)]
            train_accuracy.append(sum(filtered_values) / len(filtered_values))

            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f" \nAvg Training Stats after {epoch+1} global rounds:")
                print(f"Training Loss : {np.mean(np.array(train_loss))}")
                print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))

    def _get_a_user_cfgs(self, idx):
        """To get a user's configs for local training"""
        user = self.user_groups[idx]

        # update data_cfgs
        # Use defaultdict to collect dates for each basin
        basin_dates = defaultdict(list)

        for _, (basin, time) in user.items():
            basin_dates[basin].append(time)

        # Initialize a list to store distinct basins
        basins = []

        # for each basin, we can find its date range
        date_ranges = {}
        for basin, times in basin_dates.items():
            basins.append(basin)
            date_ranges[basin] = (np.min(times), np.max(times))
        # get the longest date range
        longest_date_range = max(date_ranges.values(), key=lambda x: x[1] - x[0])
        # transform the date range of numpy data into string
        longest_date_range = [
            np.datetime_as_string(dt, unit="D") for dt in longest_date_range
        ]
        user_cfgs = copy.deepcopy(self.cfgs)
        # update data_cfgs
        update_nested_dict(
            user_cfgs, ["data_cfgs", "t_range_train"], longest_date_range
        )
        # for local training in FL, we don't need a validation set
        update_nested_dict(user_cfgs, ["data_cfgs", "t_range_valid"], None)
        # for local training in FL, we don't need a test set, but we should set one to avoid error
        update_nested_dict(user_cfgs, ["data_cfgs", "t_range_test"], longest_date_range)
        update_nested_dict(user_cfgs, ["data_cfgs", "object_ids"], basins)

        # update training_cfgs
        # we also need to update some training params for local training from FL settings
        update_nested_dict(
            user_cfgs,
            ["training_cfgs", "epochs"],
            user_cfgs["model_cfgs"]["fl_hyperparam"]["fl_local_ep"],
        )
        update_nested_dict(
            user_cfgs,
            ["evaluation_cfgs", "test_epoch"],
            user_cfgs["model_cfgs"]["fl_hyperparam"]["fl_local_ep"],
        )
        # don't need to save model weights for local training
        update_nested_dict(
            user_cfgs,
            ["training_cfgs", "save_epoch"],
            None,
        )
        # there are two settings for batch size in configs, we need to update both of them
        update_nested_dict(
            user_cfgs,
            ["training_cfgs", "batch_size"],
            user_cfgs["model_cfgs"]["fl_hyperparam"]["fl_local_bs"],
        )
        update_nested_dict(
            user_cfgs,
            ["data_cfgs", "batch_size"],
            user_cfgs["model_cfgs"]["fl_hyperparam"]["fl_local_bs"],
        )

        # update model_cfgs finally
        # For local model, its model_type is Normal
        update_nested_dict(user_cfgs, ["model_cfgs", "model_type"], "Normal")
        update_nested_dict(
            user_cfgs,
            ["model_cfgs", "fl_hyperparam"],
            None,
        )
        return user_cfgs


class TransLearnHydro(DeepHydro):
    def __init__(self, cfgs: Dict, pre_model=None):
        super().__init__(cfgs, pre_model)

    def load_model(self, mode="train"):
        """Load model for transfer learning"""
        model_cfgs = self.cfgs["model_cfgs"]
        if self.weight_path is None and self.pre_model is None:
            raise NotImplementedError(
                "For transfer learning, we need a pre-trained model"
            )
        model = super().load_model(mode)
        if (
            "weight_path_add" in model_cfgs
            and "freeze_params" in model_cfgs["weight_path_add"]
        ):
            freeze_params = model_cfgs["weight_path_add"]["freeze_params"]
            for param in freeze_params:
                exec(f"model.{param}.requires_grad = False")
        return model

    def _load_model_from_pth(self):
        weight_path = self.weight_path
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
        checkpoint = torch.load(weight_path, map_location=self.device)
        if "weight_path_add" in model_cfgs:
            if "excluded_layers" in model_cfgs["weight_path_add"]:
                # delete some layers from source model if we don't need them
                excluded_layers = model_cfgs["weight_path_add"]["excluded_layers"]
                for layer in excluded_layers:
                    del checkpoint[layer]
                print("sucessfully deleted layers")
            else:
                print("directly loading identically-named layers of source model")
        model.load_state_dict(checkpoint, strict=False)
        print("Weights sucessfully loaded")
        return model


class MultiTaskHydro(DeepHydro):
    def __init__(self, cfgs: Dict, pre_model=None):
        super().__init__(cfgs, pre_model)

    def _get_optimizer(self, training_cfgs):
        params_in_opt = self.model.parameters()
        if training_cfgs["criterion"] == "UncertaintyWeights":
            # log_var = torch.zeros((1,), requires_grad=True)
            log_vars = [
                torch.zeros((1,), requires_grad=True, device=self.device)
                for _ in range(training_cfgs["multi_targets"])
            ]
            params_in_opt = list(self.model.parameters()) + log_vars
        return pytorch_opt_dict[training_cfgs["optimizer"]](
            params_in_opt, **training_cfgs["optim_params"]
        )

    def _get_loss_func(self, training_cfgs):
        if "criterion_params" in training_cfgs:
            loss_param = training_cfgs["criterion_params"]
            if loss_param is not None:
                criterion_init_params = {
                    key: (
                        pytorch_criterion_dict[loss_param[key]]()
                        if key == "loss_funcs"
                        else loss_param[key]
                    )
                    for key in loss_param.keys()
                }
        if training_cfgs["criterion"] == "MultiOutWaterBalanceLoss":
            # TODO: hard code for streamflow and ET
            stat_dict = self.traindataset.target_scaler.stat_dict
            stat_dict_keys = list(stat_dict.keys())
            q_name = np.intersect1d(
                [
                    "usgsFlow",
                    "streamflow",
                    "Q",
                    "qobs",
                ],
                stat_dict_keys,
            )[0]
            et_name = np.intersect1d(
                [
                    "ET",
                    "LE",
                    "GPP",
                    "Ec",
                    "Es",
                    "Ei",
                    "ET_water",
                    # sum pf ET components in PML V2
                    "ET_sum",
                ],
                stat_dict_keys,
            )[0]
            q_mean = self.training.target_scaler.stat_dict[q_name][2]
            q_std = self.training.target_scaler.stat_dict[q_name][3]
            et_mean = self.training.target_scaler.stat_dict[et_name][2]
            et_std = self.training.target_scaler.stat_dict[et_name][3]
            means = [q_mean, et_mean]
            stds = [q_std, et_std]
            criterion_init_params["means"] = means
            criterion_init_params["stds"] = stds
        return pytorch_criterion_dict[training_cfgs["criterion"]](
            **criterion_init_params
        )


class DistributedDeepHydro(MultiTaskHydro):
    def __init__(self, world_size, cfgs: Dict):
        super().__init__(cfgs, cfgs["model_cfgs"]["weight_path"])
        self.world_size = world_size

    def setup(self, rank):
        os.environ["MASTER_ADDR"] = self.cfgs["training_cfgs"]["master_addr"]
        os.environ["MASTER_PORT"] = self.cfgs["training_cfgs"]["port"]
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        self.device = torch.device(rank)
        self.rank = rank

    def cleanup(self):
        dist.destroy_process_group()

    def load_model(self, mode="train"):
        if mode == "infer":
            if self.weight_path is None:
                # for continue training
                self.weight_path = self._get_trained_model()
        elif mode != "train":
            raise ValueError("Invalid mode; must be 'train' or 'infer'")
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        if model_name not in pytorch_model_dict:
            raise NotImplementedError(
                f"Error the model {model_name} was not found in the model dict. Please add it."
            )
        if self.pre_model is not None:
            return self._load_pretrain_model()
        elif self.weight_path is not None:
            # load model from pth file (saved weights and biases)
            return self._load_model_from_pth()
        else:
            return pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])

    def model_train(self):
        model = self.load_model().to(self.device)
        self.model = DDP(model, device_ids=[self.rank])
        training_cfgs = self.cfgs["training_cfgs"]
        # The file path to load model weights from; defaults to "model_save"
        model_filepath = self.cfgs["data_cfgs"]["test_path"]
        data_cfgs = self.cfgs["data_cfgs"]
        es = None
        if training_cfgs["early_stopping"]:
            es = EarlyStopper(training_cfgs["patience"])
        criterion = self._get_loss_func(training_cfgs)
        opt = self._get_optimizer(training_cfgs)
        scheduler = self._get_scheduler(training_cfgs, opt)
        max_epochs = training_cfgs["epochs"]
        start_epoch = training_cfgs["start_epoch"]
        # use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader, validation_data_loader = self._get_dataloader(
            training_cfgs, data_cfgs
        )
        logger = TrainLogger(model_filepath, self.cfgs, opt)
        for epoch in range(start_epoch, max_epochs + 1):
            data_loader.sampler.set_epoch(epoch)
            with logger.log_epoch_train(epoch) as train_logs:
                total_loss, n_iter_ep = torch_single_train(
                    self.model,
                    opt,
                    criterion,
                    data_loader,
                    device=self.device,
                    which_first_tensor=training_cfgs["which_first_tensor"],
                )
                train_logs["train_loss"] = total_loss
                train_logs["model"] = self.model

            valid_loss = None
            valid_metrics = None
            if data_cfgs["t_range_valid"] is not None:
                with logger.log_epoch_valid(epoch) as valid_logs:
                    valid_loss, valid_metrics = self._1epoch_valid(
                        training_cfgs, criterion, validation_data_loader, valid_logs
                    )

            self._scheduler_step(training_cfgs, scheduler, valid_loss)
            logger.save_session_param(
                epoch, total_loss, n_iter_ep, valid_loss, valid_metrics
            )
            logger.save_model_and_params(self.model, epoch, self.cfgs)
            if es and not es.check_loss(
                self.model,
                valid_loss,
                self.cfgs["data_cfgs"]["test_path"],
            ):
                print("Stopping model now")
                break
        # if self.rank == 0:
        # logging.log(1, f"Training complete in: {time.time() - start_time:.2f} seconds"

    def run(self):
        self.setup(self.rank)
        self.model_train()
        self.model_evaluate()
        self.cleanup()


def train_worker(rank, world_size, cfgs):
    trainer = DistributedDeepHydro(world_size, cfgs)
    trainer.rank = rank
    trainer.run()


model_type_dict = {
    "Normal": DeepHydro,
    "FedLearn": FedLearnHydro,
    "TransLearn": TransLearnHydro,
    "MTL": MultiTaskHydro,
    "DDP_MTL": DistributedDeepHydro,
}
