"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-09-25 19:09:12
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: /torchhydro/torchhydro/trainers/deep_hydro.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC, abstractmethod
import copy
from typing import Dict
import numpy as np
import torch
from torch import nn
from hydrodataset import HydroDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchhydro.datasets.sampler import KuaiSampler, fl_sample_basin, fl_sample_region
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
    average_weights,
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

    def __init__(self, data_source: HydroDataset, cfgs: Dict):
        """
        Parameters
        ----------
        data_source
            the digital twin of a data_source in reality
        cfgs
            configs for initializing DeepHydro
        """
        self.cfgs = cfgs
        if "weight_path" in cfgs["model_cfgs"]:
            self.model = self.load_model(
                cfgs["model_cfgs"],
                cfgs["model_cfgs"]["weight_path"],
            )
        else:
            self.model = self.load_model(cfgs["model_cfgs"])
        self.traindataset = self.make_dataset(data_source, cfgs["data_cfgs"], "train")
        if cfgs["data_cfgs"]["t_range_valid"] is not None:
            self.validdataset = self.make_dataset(
                data_source, cfgs["data_cfgs"], "valid"
            )
        self.testdataset = self.make_dataset(data_source, cfgs["data_cfgs"], "test")

    @abstractmethod
    def load_model(self, model_cfgs: Dict, weight_path=None) -> object:
        """
        Get a time series forecast model and it varies based on the underlying framework used

        Parameters
        ----------
        model_cfgs
            model configs
        weight_path
            where we put model's weights

        Returns
        -------
        object
            a Hydro DL model
        """
        raise NotImplementedError

    @abstractmethod
    def make_dataset(
        self, data_source: HydroDataset, data_cfgs: Dict, loader_type: str
    ) -> object:
        """
        Initializes a pytorch dataset based on the provided data_source.

        Parameters
        ----------
        data_source
            a class for a given data source
        data_cfgs
            configs for loading data source
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

    def __init__(self, data_source: HydroDataset, cfgs: Dict):
        """
        Parameters
        ----------
        data_source
            data source where we read data from
        params_dict
            parameters set for the model
        """
        self.device_num = cfgs["training_cfgs"]["device"]
        self.device = get_the_device(self.device_num)
        super().__init__(data_source, cfgs)
        print(f"Torch is using {str(self.device)}")

    def load_model(self, model_cfgs: Dict, weight_path: str = None, strict=True):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Parameters
        ----------
        model_cfgs
            model configs
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
        model_name = model_cfgs["model_name"]
        if model_name not in pytorch_model_dict:
            raise NotImplementedError(
                f"Error the model {model_name} was not found in the model dict. Please add it."
            )
        model = pytorch_model_dict[model_name](**model_cfgs["model_param"])
        if weight_path is not None:
            # if the model has been trained
            strict = False
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
            which_first_tensor = self.cfgs["training_cfgs"]["which_first_tensor"]
            sequece_first = which_first_tensor == "sequence"
            parallel_dim = 1 if sequece_first else 0
            model = nn.DataParallel(model, device_ids=self.device_num, dim=parallel_dim)
        model.to(self.device)
        if (
            weight_path is not None
            and "weight_path_add" in model_cfgs
            and "freeze_params" in model_cfgs["weight_path_add"]
        ):
            freeze_params = model_cfgs["weight_path_add"]["freeze_params"]
            for param in freeze_params:
                if "tl_tag" in model.__dict__:
                    exec(f"model.tl_part.{param}.requires_grad = False")
                else:
                    exec(f"model.{param}.requires_grad = False")
        if ("model_wrapper" in list(model_cfgs.keys())) and (
            model_cfgs["model_wrapper"] is not None
        ):
            wrapper_name = model_cfgs["model_wrapper"]
            wrapper_params = model_cfgs["model_wrapper_param"]
            model = pytorch_model_wrapper_dict[wrapper_name](model, **wrapper_params)
        return model

    def make_dataset(
        self, data_source: HydroDataset, data_cfgs: Dict, loader_type: str
    ):
        """
        Initializes a pytorch dataset based on the provided data_source.

        Parameters
        ----------
        data_source: Hydrodataset
            data_source read data from data source
        data_cfgs
            configs for loading data
        loader_type
            train or valid or test

        Returns
        -------
        object
            an object initializing from class in datasets_dict in data_dict.py
        """
        dataset_name = data_cfgs["dataset"]
        if dataset_name in list(datasets_dict.keys()):
            dataset = datasets_dict[dataset_name](data_source, data_cfgs, loader_type)
        else:
            raise NotImplementedError(
                "Error the dataset "
                + str(dataset_name)
                + " was not found in the dataset dict. Please add it."
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
        if "early_stopping" in self.cfgs:
            es = EarlyStopper(self.cfgs["early_stopping"]["patience"])
        criterion = self._get_loss_func(training_cfgs)
        opt = self._get_optimizer(training_cfgs)
        lr_scheduler = training_cfgs["lr_scheduler"]
        max_epochs = training_cfgs["epochs"]
        start_epoch = training_cfgs["start_epoch"]
        # use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader, validation_data_loader = self._get_dataloader(
            training_cfgs, data_cfgs
        )
        logger = TrainLogger(model_filepath, self.cfgs, opt)
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
                    which_first_tensor=training_cfgs["which_first_tensor"],
                )
                train_logs["train_loss"] = total_loss
                train_logs["model"] = self.model

            valid_loss = None
            valid_metrics = None
            if data_cfgs["t_range_valid"] is not None:
                with logger.log_epoch_valid(epoch) as valid_logs:
                    valid_obss_np, valid_preds_np, valid_loss = compute_validation(
                        self.model,
                        criterion,
                        validation_data_loader,
                        device=self.device,
                        which_first_tensor=training_cfgs["which_first_tensor"],
                    )
                    evaluation_metrics = self.cfgs["evaluation_cfgs"]["metrics"]
                    fill_nan = self.cfgs["evaluation_cfgs"]["fill_nan"]
                    target_col = self.cfgs["data_cfgs"]["target_cols"]
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
            logger.save_model_and_params(self.model, epoch, self.cfgs)
            if es and not es.check_loss(self.model, valid_loss):
                print("Stopping model now")
                self.model.load_state_dict(torch.load("checkpoint.pth"))
                break

        logger.tb.close()

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
        train_dataset = self.traindataset
        sampler = None
        if data_cfgs["sampler"] is not None:
            # now we only have one special sampler from Kuai Fang's Deep Learning papers
            batch_size = data_cfgs["batch_size"]
            rho = data_cfgs["forecast_history"]
            warmup_length = data_cfgs["warmup_length"]
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
            batch_size=training_cfgs["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=worker_num,
            pin_memory=pin_memory,
            timeout=0,
        )
        if data_cfgs["t_range_valid"] is not None:
            valid_dataset = self.validdataset
            validation_data_loader = DataLoader(
                valid_dataset,
                batch_size=training_cfgs["batch_size"],
                shuffle=False,
                num_workers=worker_num,
                pin_memory=pin_memory,
                timeout=0,
            )

        return data_loader, validation_data_loader


class FedLearnHydro(DeepHydro):
    """Federated Learning Hydrological DL model"""

    def __init__(self, data_source: HydroDataset, params_dict: Dict):
        super().__init__(data_source, params_dict)
        # a user group which is a dict where the keys are the user index
        # and the values are the corresponding data for each of those users
        train_dataset = self.traindataset
        fl_params = self.cfgs["model_cfgs"]["fl_params"]
        # sample training data amongst users
        if fl_params["fl_sample"] == "basin":
            # Sample a basin for a user
            user_groups = fl_sample_basin(train_dataset)
        else:
            if fl_params["fl_sample"] != "region":
                raise NotImplementedError()
            else:
                # Sample a region for a user
                user_groups = fl_sample_region(train_dataset)
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
        fl_params = model_cfgs["fl_params"]
        # total rounds in a FL system is max_epochs
        for epoch in tqdm(range(start_epoch, max_epochs + 1)):
            local_weights, local_losses = [], []
            print(f"\n | Global Training Round : {epoch} |\n")

            global_model.train()
            m = max(int(fl_params["fl_frac"] * self.num_users), 1)
            # randomly select m users, they will be the clients in this round
            idxs_users = np.random.choice(range(self.num_users), m, replace=False)

            for idx in idxs_users:
                # each user will be used to train the model locally
                # user_gourps[idx] means the idx of dataset for a user
                self.user_groups[idx]
                local_model = DeepHydro(
                    model_cfgs["model_name"],
                    self.data_source,
                    self.cfgs,
                )
                w, loss = local_model.model_train(
                    model=copy.deepcopy(global_model), global_round=epoch
                )
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(self.num_users):
                local_model = DeepHydro(
                    model_cfgs["model_name"],
                    self.data_source,
                    self.cfgs,
                )
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc) / len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f" \nAvg Training Stats after {epoch+1} global rounds:")
                print(f"Training Loss : {np.mean(np.array(train_loss))}")
                print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, valid_dataset)

        print(f" \n Results after {args.epochs} global rounds of training:")
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name = "../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".format(
            args.dataset,
            args.model,
            args.epochs,
            args.frac,
            args.iid,
            args.local_ep,
            args.local_bs,
        )

        with open(file_name, "wb") as f:
            pickle.dump([train_loss, train_accuracy], f)

        print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))


model_type_dict = {
    "Normal": DeepHydro,
    "FedLearn": FedLearnHydro,
}
