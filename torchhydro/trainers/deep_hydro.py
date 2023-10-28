"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-10-28 13:22:06
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: \torchhydro\torchhydro\trainers\deep_hydro.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from functools import reduce
import os
from typing import Dict, Tuple
from hydroutils.hydro_stat import stat_error
import numpy as np
import torch
from torch import nn
from hydrodataset import HydroDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchhydro.explainers.shap import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
)
from torchhydro.configs.config import update_nested_dict
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
    denormalize4eval,
    evaluate_validation,
    compute_validation,
    model_infer,
    torch_single_train,
    cellstates_when_inference,
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
        self._data_source = data_source
        self._cfgs = cfgs

    @property
    def data_source(self):
        """data source"""
        return self._data_source

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
    def load_model(self) -> object:
        """Get a Hydro DL model"""
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self, is_tra_val_te: str) -> object:
        """
        Initializes a pytorch dataset based on the provided data_source.

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
        data_source: HydroDataset,
        cfgs: Dict,
        pre_model=None,
    ):
        """
        Parameters
        ----------
        data_source
            data source where we read data from
        cfgs
            configs for the model
        pre_model
            a pre-trained model, if it is not None,
            we will use its weights to initialize this model
            by default None
        """
        self.device_num = cfgs["training_cfgs"]["device"]
        self.device = get_the_device(self.device_num)
        self.pre_model = pre_model
        super().__init__(data_source, cfgs)
        self.model = self.load_model()
        self.traindataset = self.make_dataset("train")
        if cfgs["data_cfgs"]["t_range_valid"] is not None:
            self.validdataset = self.make_dataset("valid")
        self.testdataset = self.make_dataset("test")
        print(f"Torch is using {str(self.device)}")

    def load_model(self):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Returns
        -------
        object
            model in pytorch_model_dict in model_dict_function.py
        """
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
        Initializes a pytorch dataset based on the provided data_source.

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
        data_source = self.data_source
        if dataset_name in list(datasets_dict.keys()):
            dataset = datasets_dict[dataset_name](data_source, data_cfgs, is_tra_val_te)
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
        # return the trained model weights and bias and the epoch loss
        return self.model.state_dict(), sum(logger.epoch_loss) / len(logger.epoch_loss)

    def model_evaluate(self) -> Tuple[Dict, np.array, np.array]:
        """
        A function to evaluate a model, called at end of training.

        Returns
        -------
        tuple[dict, np.array, np.array]
            eval_log, denormalized predictions and observations
        """
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
        # save result here
        eval_log = {}

        # test the trained model
        test_epoch = self.cfgs["evaluation_cfgs"]["test_epoch"]
        train_epoch = self.cfgs["training_cfgs"]["epochs"]
        if test_epoch != train_epoch:
            # Generally we use same epoch for train and test, but sometimes not
            # TODO: better refactor this part, because sometimes we save multi models for multi hyperparameters
            model_filepath = self.cfgs["data_cfgs"]["test_path"]
            self.model = self.load_model(
                self.cfgs["model_cfgs"],
                weight_path=os.path.join(
                    model_filepath, f"model_Ep{str(test_epoch)}.pth"
                ),
            )
        preds_xr, obss_xr, test_data = self.inference()
        #  Then evaluate the model metrics
        if type(fill_nan) is list and len(fill_nan) != len(target_col):
            raise ValueError("length of fill_nan must be equal to target_col's")
        for i in range(len(target_col)):
            obs_xr = obss_xr[list(obss_xr.data_vars.keys())[i]]
            pred_xr = preds_xr[list(preds_xr.data_vars.keys())[i]]
            if type(fill_nan) is str:
                inds = stat_error(
                    obs_xr.to_numpy(),
                    pred_xr.to_numpy(),
                    fill_nan,
                )
            else:
                inds = stat_error(
                    obs_xr.to_numpy(),
                    pred_xr.to_numpy(),
                    fill_nan[i],
                )
            for evaluation_metric in evaluation_metrics:
                eval_log[f"{evaluation_metric} of {target_col[i]}"] = inds[
                    evaluation_metric
                ]

        # Finally, try to explain model behaviour using shap
        is_shap = self.cfgs["evaluation_cfgs"]["explainer"] == "shap"
        if is_shap:
            deep_explain_model_summary_plot(self.model, test_data)
            deep_explain_model_heatmap(self.model, test_data)

        return eval_log, preds_xr, obss_xr

    def inference(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """infer using trained model and unnormalized results"""
        data_cfgs = self.cfgs["data_cfgs"]
        training_cfgs = self.cfgs["training_cfgs"]
        device = get_the_device(self.cfgs["training_cfgs"]["device"])
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
                ys, output = model_infer(seq_first, device, self.model, xs, ys)
                test_preds.append(output.cpu().numpy())
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
        # TODO: not support return_cell_states yet
        return_cell_state = False
        if return_cell_state:
            return cellstates_when_inference(seq_first, data_cfgs, pred)
        pred_xr, obs_xr = denormalize4eval(test_dataloader, pred, obs)
        return pred_xr, obs_xr, self.testdataset

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

        return data_loader, None


class FedLearnHydro(DeepHydro):
    """Federated Learning Hydrological DL model"""

    def __init__(self, data_source: HydroDataset, cfgs: Dict):
        super().__init__(data_source, cfgs)
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
                    self.data_source,
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
                    self.data_source,
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
    def __init__(self, data_source: HydroDataset, cfgs: Dict, pre_model=None):
        super().__init__(data_source, cfgs, pre_model)

    def load_model(self):
        """Load model for transfer learning"""
        model_cfgs = self.cfgs["model_cfgs"]
        if self.weight_path is None and self.pre_model is None:
            raise NotImplementedError(
                "For transfer learning, we need a pre-trained model"
            )
        model = super().load_model()
        if (
            "weight_path_add" in model_cfgs
            and "freeze_params" in model_cfgs["weight_path_add"]
        ):
            freeze_params = model_cfgs["weight_path_add"]["freeze_params"]
            for param in freeze_params:
                exec(f"model.{param}.requires_grad = False")
        if ("model_wrapper" in list(model_cfgs.keys())) and (
            model_cfgs["model_wrapper"] is not None
        ):
            wrapper_name = model_cfgs["model_wrapper"]
            wrapper_params = model_cfgs["model_wrapper_param"]
            model = pytorch_model_wrapper_dict[wrapper_name](model, **wrapper_params)
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


model_type_dict = {
    "Normal": DeepHydro,
    "FedLearn": FedLearnHydro,
    "TransLearn": TransLearnHydro,
}
