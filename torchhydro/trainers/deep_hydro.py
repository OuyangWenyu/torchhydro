"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:15:48
LastEditTime: 2025-07-13 16:25:31
LastEditors: Wenyu Ouyang
Description: HydroDL model class
FilePath: \torchhydro\torchhydro\trainers\deep_hydro.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import bisect
import copy
import inspect
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple
import os

import numpy as np
import xarray as xr
import torch
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import shap

from torchhydro.configs.config import update_nested_dict
from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.datasets.data_sets import (
    BaseDataset,
    prebatched_collate_fn,
    static_attr_batch_collate_fn,
)
from torchhydro.datasets.sampler import (
    fl_sample_basin,
    fl_sample_region,
    data_sampler_dict,
)
from torchhydro.models.model_dict_function import (
    pytorch_criterion_dict,
    pytorch_model_dict,
    pytorch_opt_dict,
)
from torchhydro.trainers.train_logger import TrainLogger
from torchhydro.trainers.train_utils import (
    EarlyStopper,
    average_weights,
    evaluate_validation,
    compute_validation,
    model_infer,
    read_pth_from_model_loader,
    torch_single_train,
    get_preds_to_be_eval,
    varied_length_collate_fn,
    gnn_collate_fn,
)
from torchhydro.trainers.fabric_wrapper import create_fabric_wrapper


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
        # Initialize fabric based on configuration
        self.fabric = create_fabric_wrapper(cfgs.get("training_cfgs", {}))
        self.pre_model = pre_model
        self.model = self.fabric.setup_module(self.load_model())
        if cfgs["training_cfgs"]["train_mode"]:
            self.traindataset = self.make_dataset("train")
            if cfgs["data_cfgs"]["t_range_valid"] is not None:
                self.validdataset = self.make_dataset("valid")
        self.testdataset: BaseDataset = self.make_dataset("test")

    @property
    def device(self):
        """Get the device from fabric wrapper"""
        return self.fabric._device

    def load_model(self, mode="train"):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Returns
        -------
        object
            model in pytorch_model_dict in model_dict_function.py
        """
        if mode == "infer":
            if self.weight_path is None or self.cfgs["model_cfgs"]["continue_train"]:
                # if no weight path is provided
                # or weight file is provided but continue train again,
                # we will use the trained model in the new case_dir directory
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
            return self._load_model_from_pth()
        else:
            return pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])

    def _load_pretrain_model(self):
        """load a pretrained model as the initial model"""
        return self.pre_model

    def _load_model_from_pth(self):
        weight_path = self.weight_path
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
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
            dataset = datasets_dict[dataset_name](self.cfgs, is_tra_val_te)
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
        model_filepath = self.cfgs["data_cfgs"]["case_dir"]
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
                    non_blocking_transfer=training_cfgs.get(
                        "non_blocking_transfer",
                        bool(training_cfgs.get("pin_memory", False)),
                    ),
                    cuda_prefetch=bool(training_cfgs.get("cuda_prefetch", False)),
                    cuda_prefetch_batches=int(
                        training_cfgs.get("cuda_prefetch_batches", 2)
                    ),
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
                self.cfgs["data_cfgs"]["case_dir"],
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
            target_lr = lr_scheduler_cfg["lr"]
            scheduler = LambdaLR(opt, lr_lambda=lambda epoch: target_lr / opt.param_groups[0]['lr'])
            # scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
        elif isinstance(lr_scheduler_cfg, dict) and all(
            isinstance(epoch, int) for epoch in lr_scheduler_cfg
        ):
            # piecewise constant learning rate
            epochs = sorted(lr_scheduler_cfg.keys())
            values = [lr_scheduler_cfg[e] for e in epochs]

            def lr_lambda(epoch):
                idx = bisect.bisect_right(epochs, epoch) - 1
                return 1.0 if idx < 0 else values[idx]

            scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
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
        if (
            self.cfgs["training_cfgs"]["valid_batch_mode"] == "test"
            and self.cfgs["training_cfgs"]["calc_metrics"]
        ):
            # NOTE: Now we only evaluate the metrics for test-mode validation
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
        model_pth_dir = self.cfgs["data_cfgs"]["case_dir"]
        return read_pth_from_model_loader(model_loader, model_pth_dir)

    def model_evaluate(self):
        """
        Evaluate the model, and perform SHAP analysis if needed.

        Returns
        -------
        preds_xr, obss_xr
        """
        self.model = self.load_model(mode="infer").to(self.device)

        data_cfgs = self.cfgs["data_cfgs"]
        training_cfgs = self.cfgs["training_cfgs"]

        # Create dataloader in advance as it is used for both inference and SHAP
        test_dataloader = self._get_dataloader(training_cfgs, data_cfgs, mode="infer")

        # Decide whether to perform SHAP analysis based on configuration
        eval_cfgs = self.cfgs.get("evaluation_cfgs", {})
        shap_cfgs = eval_cfgs.get("shap_cfgs", {})

        # Inference returns X_all for SHAP analysis only if enabled
        if shap_cfgs.get("enable", False):
            pred_xr, obs_xr, X_all = self.inference(test_dataloader, return_inputs=True)
            print("SHAP analysis enabled.")
        else:
            pred_xr, obs_xr = self.inference(test_dataloader, return_inputs=False)
            X_all = None
            print("SHAP analysis disabled.")

        if shap_cfgs.get("enable", False):
            self.run_shap_analysis(
                X_all=X_all,
                save_dir=shap_cfgs.get("save_dir", "./shap_results"),
                max_sample=shap_cfgs.get("max_sample", 3000),
                background_size=shap_cfgs.get("background_size", 100),
            )

        return pred_xr, obs_xr

    def inference(self, dataloader, return_inputs=True):
        self.model.eval()

        preds = []
        obss = []
        X_all = []

        with torch.no_grad():
            for batch in dataloader:
                # Handle Seq2Seq format: batch[0] might be a list [encoder_input, decoder_input]
                if isinstance(batch[0], list):
                    # Seq2Seq format: move each element to device and transpose to (seq, batch, feature)
                    x = [inp.permute(1, 0, 2).to(self.device) for inp in batch[0]]
                    # if x[0].shape[0] == 0:
                    # Handle empty batch case
                    # continue
                    pred = self.model(*x)
                    # Transpose prediction back to (batch, seq, feature)
                    pred = pred.permute(1, 0, 2)

                else:
                    # Traditional format: single tensor
                    x = batch[0].to(self.device)
                    # --- PERMUTE TO SEQ-FIRST IF NEEDED ---
                    if (
                        self.cfgs["training_cfgs"].get("which_first_tensor")
                        == "sequence"
                        and x.ndim == 3
                    ):
                        x = x.transpose(0, 1)

                    pred = self.model(x)

                    # --- TRANSPOSE BACK TO BATCH-FIRST ---
                    if (
                        self.cfgs["training_cfgs"].get("which_first_tensor")
                        == "sequence"
                        and isinstance(pred, torch.Tensor)
                        and pred.ndim == 3
                    ):
                        pred = pred.transpose(0, 1)

                y = batch[1].to(self.device).cpu()

                # --- DO NOT TRANSPOSE LABELS ---
                # Labels from _flood_event_collate_fn or default stack are already (Batch, Seq, Feat).
                # Transposing them here would make them (Seq, Batch, Feat), which breaks concatenation.

                # Handle models that return dict (e.g., Seq2Seq with attention)
                if isinstance(pred, dict):
                    # If this was a sequence-first dict, permute its tensors
                    if (
                        self.cfgs["training_cfgs"].get("which_first_tensor")
                        == "sequence"
                    ):
                        for k, v in pred.items():
                            if isinstance(v, torch.Tensor) and v.ndim == 3:
                                # Avoid double-transposition if it starts with 'f' (frequency branches)
                                # and we already know where it came from. Actually, safer to check shape.
                                # seq_first: (Seq, Batch, Feat). Batch size is usually smaller than Seq len.
                                # But in inference, Seq len (Flood events) can be smaller too.
                                # Let's assume if it came through model(x) it is Seq-first.
                                pred[k] = v.transpose(0, 1)
                    if "f1" in pred:
                        # Use f1 for 1-hour scale predictions
                        pred = pred["f1"]
                    # elif 'f0' in pred:
                    # Fallback to f0
                    # pred = pred['f0']
                    else:
                        # Generic handling for other dict outputs
                        pred_tensor = None
                        for key in ["predictions", "output", "pred", "logits"]:
                            if key in pred and isinstance(pred[key], torch.Tensor):
                                pred_tensor = pred[key]
                                break
                        if pred_tensor is None:
                            pred_tensor = next(iter(pred.values()))
                        pred = pred_tensor

                preds.append(pred.cpu())
                obss.append(y.cpu())

                if return_inputs:
                    # Handle both formats for X_all
                    if isinstance(x, list):
                        # For Seq2Seq, store the encoder input (first element)
                        x_for_shap = x[0]
                    else:
                        x_for_shap = x

                    # Ensure SHAP input is Batch-first
                    if (
                        self.cfgs["training_cfgs"].get("which_first_tensor")
                        == "sequence"
                        and x_for_shap.ndim == 3
                    ):
                        x_for_shap = x_for_shap.transpose(0, 1)

                    X_all.append(x_for_shap.cpu())

        # Convert to numpy
        # Pad sequence dimension independently for preds and obss
        if len(preds) > 0:
            max_p_s = max(p.shape[1] for p in preds)
            max_o_s = max(o.shape[1] for o in obss)

            padded_preds = []
            padded_obss = []

            for p, o in zip(preds, obss):
                if p.shape[1] < max_p_s:
                    pad_p = torch.full(
                        (p.shape[0], max_p_s - p.shape[1], p.shape[2]), np.nan
                    )
                    p = torch.cat([p, pad_p], dim=1)
                if o.shape[1] < max_o_s:
                    pad_o = torch.full(
                        (o.shape[0], max_o_s - o.shape[1], o.shape[2]), np.nan
                    )
                    o = torch.cat([o, pad_o], dim=1)
                padded_preds.append(p)
                padded_obss.append(o)

            pred_np = torch.cat(padded_preds, 0).numpy()
            obs_np = torch.cat(padded_obss, 0).numpy()
        else:
            pred_np = np.array([])
            obs_np = np.array([])

        # Convert to xarray ---- KEEP THIS (TorchHydro requirement)
        obs_xr, pred_xr = get_preds_to_be_eval(
            dataloader, self.cfgs["evaluation_cfgs"], pred_np, obs_np
        )

        if return_inputs:
            return pred_xr, obs_xr, X_all
        else:
            return pred_xr, obs_xr

    def run_shap_analysis(
        self, X_all, save_dir, max_sample=3000, feature_names=None, background_size=100
    ):
        import torch
        import numpy as np
        import shap
        import matplotlib.pyplot as plt
        import os

        print("Running FAST SHAP analysis...")

        eval_cfgs = self.cfgs.get("evaluation_cfgs", {}) or {}
        shap_cfgs = eval_cfgs.get("shap_cfgs", {}) or {}
        sequence_enable = bool(shap_cfgs.get("sequence_enable", False))
        plot_mode = shap_cfgs.get("plot_mode")
        if not isinstance(plot_mode, str):
            plot_mode = "all"
        plot_mode = plot_mode.lower()

        plot_fast = True
        plot_seq = True
        plot_hybrid = True
        plot_split = False

        if plot_mode == "split_only":
            plot_fast = False
            plot_seq = False
            plot_hybrid = False
            plot_split = True
            sequence_enable = True

        # ======================================================
        # 1. Optimize Memory & Diversity: Sample sparsely from MANY batches
        # ======================================================
        # X_all is a list of tensors, each (B, T, F)

        if not X_all:
            print("[WARNING] No data for SHAP analysis.")
            return

        # Get dimensions from first batch
        _, T, F_total = X_all[0].shape

        # We need max_sample samples.
        # To ensure spatial diversity (static attributes variation), we must sample from many batches.
        # Strategy: Randomly select N batches, then sample k items from each batch.

        import random

        n_batches = len(X_all)

        # We want to sample from at least min(50, n_batches) batches to cover different basins
        target_batch_count = min(50, n_batches)

        # How many samples per batch?
        # We want total ~ max_sample * 2 (safety margin)
        samples_per_batch = max(1, (max_sample * 2) // target_batch_count)

        print(
            f"[INFO] Sampling strategy: {target_batch_count} batches, ~{samples_per_batch} samples/batch"
        )

        batch_indices = list(range(n_batches))
        random.shuffle(batch_indices)
        selected_batch_indices = batch_indices[:target_batch_count]

        collected_subsets = []

        for idx in selected_batch_indices:
            batch = X_all[idx]  # (B, T, F)
            B = batch.shape[0]

            # Flatten this batch to (B*T, F)
            batch_flat = batch.reshape(-1, F_total)
            M_batch = batch_flat.shape[0]

            # Sample randomly from this batch
            if M_batch > samples_per_batch:
                # Random indices
                indices = torch.randperm(M_batch)[:samples_per_batch]
                batch_subset = batch_flat[indices]
            else:
                batch_subset = batch_flat

            collected_subsets.append(batch_subset)

        # Concatenate subsets
        X_subset = torch.cat(collected_subsets, dim=0)  # (N_total_subset, F)

        # Handle NaNs: Replace NaNs with 0 (mean of normalized data)
        if torch.isnan(X_subset).any():
            print(
                "[WARNING] Input data contains NaNs. Filling with 0 for SHAP analysis."
            )
            X_subset = torch.nan_to_num(X_subset, nan=0.0)

        print(f"[INFO] Collected subset samples: {X_subset.shape[0]}")

        # ======================================================
        # 2. Flatten to 2D: Already flattened in loop
        # ======================================================
        X_flat = X_subset
        M = X_flat.shape[0]
        print(f"[INFO] Flattened subset samples: {M}")

        # ======================================================
        # 3. Sample input data for acceleration
        # ======================================================
        # Ensure we don't sample more than we have
        sample_size = min(max_sample, M)
        if sample_size < M:
            # We already shuffled and sampled, but let's do one final random selection to be sure
            idx = np.random.choice(M, sample_size, replace=False)
            X_sample = X_flat[idx]
        else:
            X_sample = X_flat

        X_sample_np = X_sample.cpu().numpy()  # numpy，shape=(sample_size, F)

        print(f"[INFO] Using {sample_size} samples for SHAP")

        # ======================================================
        # 4. Select background for SHAP
        # ======================================================
        # background_size is passed as argument
        if M > background_size:
            idx_bg = np.random.choice(M, background_size, replace=False)
            background = X_flat[idx_bg]
        else:
            background = X_flat

        background_np = background.cpu().numpy()
        print(f"[INFO] Using {background.shape[0]} samples as SHAP background")

        # ======================================================
        # 5. Wrap the model: SamplingExplainer uses a python function
        # ======================================================
        # Move model to CPU to avoid MPS bugs with SHAP/LSTM
        original_device = self.device
        shap_device = torch.device("cpu")
        print(
            f"[INFO] Moving model to {shap_device} for SHAP analysis to avoid MPS errors..."
        )
        self.model.to(shap_device)
        self.model.eval()

        def model_predict(x_np):
            """
            Args:
                x_np: numpy array with shape (batch, F).

            Returns:
                numpy array with shape (batch, 1).

            Note:
                Converts numpy array to torch.Tensor, feeds it to the model,
                and reshapes the output to (batch, 1).
            """
            x_tensor = torch.tensor(
                x_np, dtype=torch.float32, device=shap_device
            )  # shape=(B,F)

            # SimpleLSTM (and default nn.LSTM) expects (Time, Batch, Features) if batch_first=False
            # We treat SHAP samples as independent time steps (Time=1)
            x_tensor = x_tensor.unsqueeze(0)  # (1, B, F)

            if torch.isnan(x_tensor).any():
                print(f"[WARNING] SHAP input x_tensor contains NaNs!")

            with torch.no_grad():
                out = self.model(x_tensor)  # (1, B, 1)

            if torch.isnan(out).any():
                print(f"[WARNING] SHAP model output contains NaNs!")
                # Optional: Replace output NaNs with 0 to allow SHAP to continue (but this hides model issues)
                # out = torch.nan_to_num(out, nan=0.0)

            # --- SamplingExplainer requires 2D output ---
            if out.shape[0] == 1:
                out = out.squeeze(0)
            else:
                print(
                    f"[WARNING] Unexpected output shape {out.shape} in SHAP. Squeezing dim 0."
                )
                out = out.squeeze(0)

            # Debug output range
            # print(f"[DEBUG] SHAP model output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

            return out.cpu().numpy()

        # ======================================================
        # 6. Run SHAP
        # ======================================================
        print(f"[INFO] Computing SHAP values...")
        explainer = shap.SamplingExplainer(model_predict, background_np)

        # Increased nsamples for stability
        shap_values = explainer.shap_values(X_sample_np, nsamples=200)

        # Check shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take the first output channel

        print(f"[INFO] SHAP shape = {np.array(shap_values).shape}")

        # Handle NaNs in SHAP values
        if np.isnan(shap_values).any():
            nan_count = np.isnan(shap_values).sum()
            total_count = shap_values.size
            print(
                f"[WARNING] SHAP values contain {nan_count}/{total_count} NaNs! Filling with 0."
            )
            shap_values = np.nan_to_num(shap_values, nan=0.0)

        # ======================================================
        # 7. Construct feature names
        # ======================================================
        if feature_names is None:
            # Attempt to retrieve real feature names from configuration and stat file
            feature_names = self._get_feature_names(F_total)

            if feature_names is not None:
                print(f"[INFO] Using reconstructed real feature names: {feature_names}")
            else:
                print(
                    f"[WARNING] Could not reconstruct feature names. Using default names."
                )
                data_cfgs = self.cfgs.get("data_cfgs", {})
                relevant_cols = data_cfgs.get("relevant_cols", [])
                feature_names = [f"DYNAMIC_{i}" for i in range(len(relevant_cols))] + [
                    f"STATIC_{i}" for i in range(F_total - len(relevant_cols))
                ]

        # ======================================================
        # 8. Output plots
        # ======================================================
        os.makedirs(save_dir, exist_ok=True)

        shap_values_fast = np.array(shap_values)
        X_fast_np = X_sample_np

        if plot_fast:
            plt.figure()
            shap.summary_plot(
                shap_values,
                X_sample_np,
                feature_names=feature_names,
                max_display=100,
                show=False,
            )
            plt.savefig(f"{save_dir}/shap_summary_fast.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(
                shap_values,
                X_sample_np,
                feature_names=feature_names,
                plot_type="bar",
                max_display=100,
                show=False,
            )
            plt.savefig(f"{save_dir}/shap_bar_fast.png", dpi=300, bbox_inches="tight")
            plt.close()

        print(f"[INFO] SHAP analysis saved to {save_dir}")

        if sequence_enable:
            training_cfgs = self.cfgs.get("training_cfgs", {}) or {}
            warmup_length = training_cfgs.get("warmup_length")
            hindcast_length = training_cfgs.get("hindcast_length")
            forecast_length = training_cfgs.get("forecast_length")

            seq_len = shap_cfgs.get("sequence_length")
            if not isinstance(seq_len, int):
                if (
                    isinstance(warmup_length, int)
                    and isinstance(hindcast_length, int)
                    and isinstance(forecast_length, int)
                ):
                    seq_len = warmup_length + hindcast_length + forecast_length
                else:
                    seq_len = T

            seq_max_sample = shap_cfgs.get("sequence_max_sample")
            if not isinstance(seq_max_sample, int):
                seq_max_sample = min(128, max_sample)
            seq_max_sample = max(1, seq_max_sample)

            seq_background_size = shap_cfgs.get("sequence_background_size")
            if not isinstance(seq_background_size, int):
                seq_background_size = min(32, background_size)
            seq_background_size = max(1, seq_background_size)

            seq_nsamples = shap_cfgs.get("sequence_nsamples")
            if not isinstance(seq_nsamples, int):
                seq_nsamples = 50

            time_agg = shap_cfgs.get("time_agg")
            if not isinstance(time_agg, str):
                time_agg = "abs_sum"
            time_agg = time_agg.lower()

            which_first_tensor = (
                self.cfgs.get("training_cfgs", {}).get("which_first_tensor") or "batch"
            )

            import random

            n_batches = len(X_all)
            target_batch_count = min(50, n_batches)
            seqs_per_batch = max(1, (seq_max_sample * 2) // max(1, target_batch_count))
            batch_indices = list(range(n_batches))
            random.shuffle(batch_indices)
            selected_batch_indices = batch_indices[:target_batch_count]

            seq_list = []
            for idx in selected_batch_indices:
                batch = X_all[idx]  # (B, T, F)
                if not isinstance(batch, torch.Tensor) or batch.ndim != 3:
                    continue
                B = int(batch.shape[0])
                if B <= 0:
                    continue
                take = min(B, seqs_per_batch)
                seq_indices = torch.randperm(B)[:take]
                for sidx in seq_indices:
                    seq = batch[sidx]  # (T, F)
                    if not isinstance(seq, torch.Tensor) or seq.ndim != 2:
                        continue
                    T_seq = int(seq.shape[0])
                    if T_seq <= 0:
                        continue
                    if T_seq >= seq_len:
                        seq_use = seq[-seq_len:]
                    else:
                        pad_len = seq_len - T_seq
                        pad = torch.zeros(
                            (pad_len, F_total), dtype=seq.dtype, device=seq.device
                        )
                        seq_use = torch.cat([pad, seq], dim=0)
                    seq_list.append(seq_use)

            if not seq_list:
                print("[WARNING] sequence_enable=True but no sequences collected; skipping.")
            else:
                X_seq = torch.stack(seq_list, dim=0)  # (N, seq_len, F)
                if torch.isnan(X_seq).any():
                    X_seq = torch.nan_to_num(X_seq, nan=0.0)

                if X_seq.shape[0] > seq_max_sample:
                    X_seq = X_seq[torch.randperm(X_seq.shape[0])[:seq_max_sample]]

                N_seq = int(X_seq.shape[0])
                bg_n = min(seq_background_size, N_seq)
                if bg_n < N_seq:
                    idx_bg = np.random.choice(N_seq, bg_n, replace=False)
                    background_seq = X_seq[idx_bg]
                else:
                    background_seq = X_seq

                print(
                    f"[INFO] SHAP sequence enabled: N={N_seq}, T={seq_len}, F={F_total}, bg={int(background_seq.shape[0])}, nsamples={seq_nsamples}, time_agg={time_agg}"
                )

                self.model.to(original_device)
                self.model.eval()

                try:
                    from torch.nn.modules.rnn import RNNBase
                except Exception:
                    RNNBase = ()

                rnn_modules = []
                rnn_training_states = []
                for m in self.model.modules():
                    if isinstance(m, RNNBase):
                        rnn_modules.append(m)
                        rnn_training_states.append(m.training)
                        m.train()

                class _ShapSeqWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, x):
                        if which_first_tensor == "sequence":
                            x = x.transpose(0, 1)
                        out = self.model(x)
                        if isinstance(out, dict):
                            if "f1" in out:
                                out = out["f1"]
                            else:
                                out = next(iter(out.values()))
                        if isinstance(out, torch.Tensor) and out.ndim == 3:
                            out = out[-1] if which_first_tensor == "sequence" else out[:, -1]
                        if isinstance(out, torch.Tensor) and out.ndim == 1:
                            out = out.unsqueeze(1)
                        return out

                wrapper = _ShapSeqWrapper(self.model).to(original_device)
                background_t = background_seq.to(original_device).float()
                sample_t = X_seq.to(original_device).float()

                try:
                    with torch.backends.cudnn.flags(enabled=False):
                        explainer = shap.GradientExplainer(wrapper, background_t)
                        shap_values = explainer.shap_values(sample_t, nsamples=seq_nsamples)
                finally:
                    for m, st in zip(rnn_modules, rnn_training_states):
                        m.train(st)

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.array(shap_values)
                if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
                    shap_values = shap_values[..., 0]
                if np.isnan(shap_values).any():
                    shap_values = np.nan_to_num(shap_values, nan=0.0)

                if time_agg == "sum":
                    shap_values_feat = shap_values.sum(axis=1)
                else:
                    shap_values_feat = np.abs(shap_values).sum(axis=1)

                X_feat = sample_t.detach().cpu().numpy().mean(axis=1)
                suffix = "sum" if time_agg == "sum" else "abs_sum"

                if plot_seq:
                    plt.figure()
                    shap.summary_plot(
                        shap_values_feat,
                        X_feat,
                        feature_names=feature_names,
                        max_display=100,
                        show=False,
                    )
                    plt.savefig(
                        f"{save_dir}/shap_summary_seq_{suffix}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                    plt.figure()
                    shap.summary_plot(
                        shap_values_feat,
                        X_feat,
                        feature_names=feature_names,
                        plot_type="bar",
                        max_display=100,
                        show=False,
                    )
                    plt.savefig(
                        f"{save_dir}/shap_bar_seq_{suffix}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                    print(f"[INFO] SHAP sequence plots saved to {save_dir}")

                data_cfgs = self.cfgs.get("data_cfgs", {}) or {}
                relevant_cols = data_cfgs.get("relevant_cols", []) or []
                static_start = int(len(relevant_cols))
                static_start = max(0, min(static_start, F_total))

                hybrid_n = int(min(512, shap_values_fast.shape[0], shap_values_feat.shape[0]))
                if hybrid_n > 0 and static_start < F_total:
                    if shap_values_fast.shape[0] > hybrid_n:
                        idx_fast = np.random.choice(shap_values_fast.shape[0], hybrid_n, replace=False)
                    else:
                        idx_fast = np.arange(shap_values_fast.shape[0])

                    if shap_values_feat.shape[0] > hybrid_n:
                        idx_seq = np.random.choice(shap_values_feat.shape[0], hybrid_n, replace=False)
                    else:
                        idx_seq = np.arange(shap_values_feat.shape[0])

                    hybrid_shap = shap_values_fast[idx_fast].copy()
                    hybrid_x = X_fast_np[idx_fast].copy()

                    x_seq_last = sample_t.detach().cpu().numpy()[:, -1, :]
                    seq_shap_sel = shap_values_feat[idx_seq]
                    seq_x_sel = x_seq_last[idx_seq]

                    hybrid_shap[:, static_start:] = seq_shap_sel[:, static_start:]
                    hybrid_x[:, static_start:] = seq_x_sel[:, static_start:]

                    hybrid_dynamic_mode = shap_cfgs.get("hybrid_dynamic_mode")
                    if not isinstance(hybrid_dynamic_mode, str):
                        hybrid_dynamic_mode = "fast"
                    hybrid_dynamic_mode = hybrid_dynamic_mode.lower()

                    dyn_seq_idx = []
                    if hybrid_dynamic_mode == "seq_all":
                        dyn_seq_idx = list(range(static_start))
                    elif hybrid_dynamic_mode == "seq_bucket0":
                        feature_buckets = (
                            (self.cfgs.get("model_cfgs", {}) or {})
                            .get("model_hyperparam", {})
                            .get("feature_buckets")
                        )
                        if isinstance(feature_buckets, (list, tuple)) and len(
                            feature_buckets
                        ) >= static_start:
                            dyn_seq_idx = [
                                i
                                for i in range(static_start)
                                if int(feature_buckets[i]) == 0
                            ]

                    extra_seq_features = shap_cfgs.get("hybrid_dynamic_seq_features")
                    if isinstance(extra_seq_features, (list, tuple)) and static_start > 0:
                        name_to_idx = {
                            str(n): i for i, n in enumerate(feature_names[:static_start])
                        }
                        for fn in extra_seq_features:
                            idx = name_to_idx.get(str(fn))
                            if idx is not None:
                                dyn_seq_idx.append(idx)

                    if dyn_seq_idx:
                        dyn_seq_idx = sorted(set(dyn_seq_idx))
                        hybrid_shap[:, dyn_seq_idx] = seq_shap_sel[:, dyn_seq_idx]
                        hybrid_x[:, dyn_seq_idx] = seq_x_sel[:, dyn_seq_idx]

                    if plot_hybrid:
                        plt.figure()
                        shap.summary_plot(
                            hybrid_shap,
                            hybrid_x,
                            feature_names=feature_names,
                            max_display=100,
                            show=False,
                        )
                        plt.savefig(
                            f"{save_dir}/shap_summary_hybrid.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()

                        plt.figure()
                        shap.summary_plot(
                            hybrid_shap,
                            hybrid_x,
                            feature_names=feature_names,
                            plot_type="bar",
                            max_display=100,
                            show=False,
                        )
                        plt.savefig(
                            f"{save_dir}/shap_bar_hybrid.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()

                        print(f"[INFO] SHAP hybrid plots saved to {save_dir}")

                    if plot_split:
                        feature_buckets = (
                            (self.cfgs.get("model_cfgs", {}) or {})
                            .get("model_hyperparam", {})
                            .get("feature_buckets")
                        )
                        fb = None
                        if isinstance(feature_buckets, (list, tuple)):
                            if len(feature_buckets) >= F_total:
                                fb = [int(x) for x in feature_buckets[:F_total]]
                            elif len(feature_buckets) == static_start:
                                fb = [int(x) for x in feature_buckets] + [0] * (
                                    F_total - static_start
                                )

                        if fb is None:
                            low_idx = list(range(static_start, F_total))
                            high_idx = list(range(0, static_start))
                        else:
                            low_idx = [i for i in range(F_total) if int(fb[i]) == 0]
                            high_idx = [i for i in range(F_total) if int(fb[i]) > 0]

                        if low_idx:
                            low_shap = hybrid_shap[:, low_idx]
                            low_x = hybrid_x[:, low_idx]
                            low_names = [feature_names[i] for i in low_idx]

                            plt.figure()
                            shap.summary_plot(
                                low_shap,
                                low_x,
                                feature_names=low_names,
                                max_display=100,
                                show=False,
                            )
                            plt.savefig(
                                f"{save_dir}/shap_summary_hybrid_lowfreq.png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                            plt.close()

                            plt.figure()
                            shap.summary_plot(
                                low_shap,
                                low_x,
                                feature_names=low_names,
                                plot_type="bar",
                                max_display=100,
                                show=False,
                            )
                            plt.savefig(
                                f"{save_dir}/shap_bar_hybrid_lowfreq.png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                            plt.close()

                        if high_idx:
                            high_shap = hybrid_shap[:, high_idx]
                            high_x = hybrid_x[:, high_idx]
                            high_names = [feature_names[i] for i in high_idx]

                            plt.figure()
                            shap.summary_plot(
                                high_shap,
                                high_x,
                                feature_names=high_names,
                                max_display=100,
                                show=False,
                            )
                            plt.savefig(
                                f"{save_dir}/shap_summary_hybrid_highfreq.png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                            plt.close()

                            plt.figure()
                            shap.summary_plot(
                                high_shap,
                                high_x,
                                feature_names=high_names,
                                plot_type="bar",
                                max_display=100,
                                show=False,
                            )
                            plt.savefig(
                                f"{save_dir}/shap_bar_hybrid_highfreq.png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                            plt.close()

                        print(f"[INFO] SHAP hybrid high/low frequency split plots saved to {save_dir}")

                        clean_extra = shap_cfgs.get("clean_extra")
                        if not isinstance(clean_extra, bool):
                            clean_extra = True
                        if clean_extra:
                            keep = {
                                "shap_bar_hybrid_lowfreq.png",
                                "shap_bar_hybrid_highfreq.png",
                                "shap_summary_hybrid_lowfreq.png",
                                "shap_summary_hybrid_highfreq.png",
                            }
                            for fn in os.listdir(save_dir):
                                if not (fn.startswith("shap_") and fn.endswith(".png")):
                                    continue
                                if fn in keep:
                                    continue
                                try:
                                    os.remove(os.path.join(save_dir, fn))
                                except Exception:
                                    pass

        # Restore model to original device
        self.model.to(original_device)
        print(f"[INFO] Model restored to {original_device}")

    def _get_feature_names(self, F_total):
        """
        Reconstruct feature names, handling One-Hot Encoding and extra features (e.g. embeddings) found in stat_dict.
        """
        import json

        data_cfgs = self.cfgs.get("data_cfgs", {})
        relevant_cols = data_cfgs.get("relevant_cols", [])
        constant_cols = data_cfgs.get("constant_cols", [])
        target_cols = data_cfgs.get("target_cols", [])

        # Check for One-Hot Encoding
        scaler_params = data_cfgs.get("scaler_params", {})
        one_hot_cols = scaler_params.get("one_hot_cols", [])

        # Try to load stat_dict
        # Priority 1: Directory of the weight path (best_model.pth)
        stat_dict = {}
        stat_file_loaded = False

        search_paths = []
        if self.weight_path:
            search_paths.append(
                os.path.join(
                    os.path.dirname(self.weight_path), "dapengscaler_stat.json"
                )
            )
        if "case_dir" in data_cfgs:
            search_paths.append(
                os.path.join(data_cfgs["case_dir"], "dapengscaler_stat.json")
            )

        for stat_file in search_paths:
            if os.path.exists(stat_file):
                try:
                    with open(stat_file, "r") as fp:
                        stat_dict = json.load(fp)
                    print(f"[INFO] Loaded stat file from {stat_file}")
                    stat_file_loaded = True
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load stat file {stat_file}: {e}")

        if not stat_file_loaded:
            print(
                "[WARNING] Could not load dapengscaler_stat.json from any expected location."
            )

        feature_names = []

        # 1. Dynamic features (relevant_cols)
        feature_names.extend(relevant_cols)

        # 2. Static features (constant_cols) from config
        # We need to track which keys we have used to identify "extra" ones later
        used_keys = set(relevant_cols) | set(target_cols)

        for col in constant_cols:
            used_keys.add(col)
            if col in one_hot_cols:
                # Expand One-Hot columns
                categories = stat_dict.get(f"{col}_categories", [])
                if categories:
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
                else:
                    # Fallback
                    feature_names.append(f"{col}_UNKNOWN")
            else:
                feature_names.append(col)

        # 3. Check for missing features (e.g. AlphaEarth embeddings)
        if len(feature_names) < F_total and stat_dict:
            print(
                f"[INFO] Feature name count ({len(feature_names)}) < Model Input ({F_total}). Searching for extra features in stat_dict..."
            )

            extra_candidates = []
            for key in stat_dict.keys():
                # Skip known columns
                if key in used_keys:
                    continue
                # Skip One-Hot category metadata
                if key.endswith("_categories"):
                    continue
                # Skip One-Hot original columns if they were expanded (already handled by checking used_keys? No, one_hot_cols are in constant_cols, so in used_keys)

                # Add to candidates
                extra_candidates.append(key)

            # Sort candidates to ensure deterministic order
            # We assume extra features (like embeddings) are appended in alphabetical or numerical order
            extra_candidates.sort()

            # Try appending extra candidates
            # We only append enough to match F_total if possible, or just all of them and see

            # Heuristic: If we need exactly N more features, and we found M extra candidates.
            needed = F_total - len(feature_names)

            if len(extra_candidates) >= needed:
                # If we have enough candidates, take them.
                # If we have MORE than needed, it's ambiguous. But usually embeddings are distinct.
                # Let's assume all valid extra candidates are part of the input.

                # However, stat_dict might contain other auxiliary variables.
                # If we match exactly, great.
                if len(extra_candidates) == needed:
                    print(
                        f"[INFO] Found exactly {needed} extra features. Appending them."
                    )
                    feature_names.extend(extra_candidates)
                else:
                    print(
                        f"[WARNING] Found {len(extra_candidates)} extra candidates, but needed {needed}. Appending first {needed} sorted keys."
                    )
                    feature_names.extend(extra_candidates[:needed])
            else:
                print(
                    f"[WARNING] Found {len(extra_candidates)} extra candidates, but needed {needed}. Appending all available."
                )
                feature_names.extend(extra_candidates)

        # Final check
        if len(feature_names) != F_total:
            print(
                f"[WARNING] Reconstructed feature names length ({len(feature_names)}) does not match model input ({F_total})."
            )
            return None

        return feature_names

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
                    if key in ("loss_funcs", "eval_loss_func"):
                        criterion_init_params[key] = pytorch_criterion_dict[
                            loss_param[key]
                        ]()
                    else:
                        criterion_init_params[key] = loss_param[key]
        return pytorch_criterion_dict[training_cfgs["criterion"]](
            **criterion_init_params
        )

    def _flood_event_collate_fn(self, batch):
        """自定义的洪水事件 collate 函数，确保所有样本长度一致"""

        # Check if this is Seq2Seq format: ([x1, x2, ...], y)
        if isinstance(batch[0][0], list):
            # Seq2Seq format: use default collate for list of tensors
            from torch.utils.data.dataloader import default_collate

            return default_collate(batch)

        # 找到这个批次中最长的序列长度
        max_len = max(tensor_data[0].shape[0] for tensor_data in batch)

        # 调整所有样本到相同长度
        processed_batch = []
        for tensor_data in batch:
            # 获取x和y（假设tensor_data[0]是x，tensor_data[1]是y）
            x = tensor_data[0]
            y = tensor_data[1] if len(tensor_data) > 1 else None

            current_len = x.shape[0]
            if current_len < max_len:
                # 使用 NaN 填充 x
                padding_x = torch.full(
                    (max_len - current_len, x.shape[1]), np.nan, dtype=x.dtype
                )
                padded_x = torch.cat([x, padding_x], dim=0)

                # 如果有y，也进行填充
                if y is not None:
                    padding_y = torch.full(
                        (max_len - current_len, y.shape[1]), np.nan, dtype=y.dtype
                    )
                    padded_y = torch.cat([y, padding_y], dim=0)
                else:
                    padded_y = None
            else:
                # 如果更长则截断
                padded_x = x[:max_len]
                padded_y = y[:max_len] if y is not None else None

            if padded_y is not None:
                processed_batch.append((padded_x, padded_y))
            else:
                processed_batch.append(padded_x)

        # 根据数据结构返回堆叠后的结果
        if len(processed_batch) > 0 and isinstance(processed_batch[0], tuple):
            return (
                torch.stack([x for x, _ in processed_batch], 0),
                torch.stack([y for _, y in processed_batch], 0),
            )
        else:
            return torch.stack(processed_batch, 0)

    def _get_dataloader(self, training_cfgs, data_cfgs, mode="train"):
        if mode == "infer":
            _collate_fn = None
            # Use GNN collate function for GNN/graph datasets in inference mode
            if (
                hasattr(self.testdataset, "__class__")
                and (
                    "GNN" in self.testdataset.__class__.__name__
                    or "TgHydroDataset" in self.testdataset.__class__.__name__
                )
            ):
                _collate_fn = gnn_collate_fn
            # 使用自定义的 collate 函数处理 FloodEventDataset
            elif (
                hasattr(self.testdataset, "__class__")
                and "FloodEvent" in self.testdataset.__class__.__name__
            ):
                _collate_fn = self._flood_event_collate_fn
            return DataLoader(
                self.testdataset,
                batch_size=training_cfgs["batch_size"],
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
                collate_fn=_collate_fn,
            )
        worker_num = 0
        pin_memory = False
        if "num_workers" in training_cfgs:
            worker_num = training_cfgs["num_workers"]
            print(f"using {str(worker_num)} workers")
        if "pin_memory" in training_cfgs:
            pin_memory = training_cfgs["pin_memory"]
            print(f"Pin memory set to {str(pin_memory)}")
        dataloader_extra_kwargs = {}
        if worker_num > 0:
            if "persistent_workers" in training_cfgs:
                dataloader_extra_kwargs["persistent_workers"] = training_cfgs[
                    "persistent_workers"
                ]
            if "prefetch_factor" in training_cfgs:
                dataloader_extra_kwargs["prefetch_factor"] = training_cfgs[
                    "prefetch_factor"
                ]
            multiprocessing_context = training_cfgs.get("multiprocessing_context")
            if multiprocessing_context not in (None, "auto"):
                dataloader_extra_kwargs["multiprocessing_context"] = (
                    multiprocessing_context
                )
            if "in_order" in inspect.signature(DataLoader).parameters:
                dataloader_extra_kwargs["in_order"] = bool(
                    training_cfgs.get("dataloader_in_order", False)
                )
        sampler = self._get_sampler(data_cfgs, training_cfgs, self.traindataset)
        batch_sampler = sampler if getattr(sampler, "is_batch_sampler", False) else None
        sampler_arg = None if batch_sampler is not None else sampler
        _collate_fn = None
        if training_cfgs["variable_length_cfgs"]["use_variable_length"]:
            _collate_fn = varied_length_collate_fn
        # Use GNN collate function for GNN/graph datasets
        elif (
            hasattr(self.traindataset, "__class__")
            and (
                "GNN" in self.traindataset.__class__.__name__
                or "TgHydroDataset" in self.traindataset.__class__.__name__
            )
        ):
            _collate_fn = gnn_collate_fn

        use_batch_level_dataloader = bool(
            training_cfgs.get("batch_level_dataloader", False)
        )
        if hasattr(self.traindataset, "_return_static_attrs_separately"):
            self.traindataset._return_static_attrs_separately = False
        if hasattr(self.traindataset, "_batch_level_dataloader_enabled"):
            self.traindataset._batch_level_dataloader_enabled = False
        train_supports_prebatched = bool(
            getattr(self.traindataset, "supports_prebatched_getitems", False)
        )
        if (
            (sampler is None or batch_sampler is not None)
            and _collate_fn is None
            and (self.traindataset.__class__ is BaseDataset or train_supports_prebatched)
        ):
            if (use_batch_level_dataloader or train_supports_prebatched) and hasattr(
                self.traindataset, "_get_batch_level_items"
            ):
                self.traindataset._batch_level_dataloader_enabled = True
                _collate_fn = prebatched_collate_fn
            elif self.traindataset.__class__ is BaseDataset:
                self.traindataset._return_static_attrs_separately = True
                _collate_fn = static_attr_batch_collate_fn
        elif use_batch_level_dataloader:
            warnings.warn(
                "batch_level_dataloader=True was ignored because this "
                "dataset uses a custom sampler/collate path or is not the "
                "standard BaseDataset."
            )
        loader_kwargs = {
            "num_workers": worker_num,
            "pin_memory": pin_memory,
            "timeout": (
                float(training_cfgs.get("dataloader_timeout", 0.0))
                if worker_num > 0
                else 0
            ),
            "collate_fn": _collate_fn,
            **dataloader_extra_kwargs,
        }
        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs.update(
                {
                    "batch_size": training_cfgs["batch_size"],
                    "shuffle": sampler_arg is None,
                    "sampler": sampler_arg,
                }
            )
        data_loader = DataLoader(self.traindataset, **loader_kwargs)
        if data_cfgs["t_range_valid"] is not None:
            # Use the same collate function for validation dataset
            _val_collate_fn = None
            if training_cfgs["variable_length_cfgs"]["use_variable_length"]:
                _val_collate_fn = varied_length_collate_fn
            elif (
                hasattr(self.validdataset, "__class__")
                and (
                    "GNN" in self.validdataset.__class__.__name__
                    or "TgHydroDataset" in self.validdataset.__class__.__name__
                )
            ):
                _val_collate_fn = gnn_collate_fn
            if hasattr(self.validdataset, "_return_static_attrs_separately"):
                self.validdataset._return_static_attrs_separately = False
            if hasattr(self.validdataset, "_batch_level_dataloader_enabled"):
                self.validdataset._batch_level_dataloader_enabled = False
            valid_supports_prebatched = bool(
                getattr(self.validdataset, "supports_prebatched_getitems", False)
            )
            if (
                _val_collate_fn is None
                and (self.validdataset.__class__ is BaseDataset or valid_supports_prebatched)
            ):
                if (use_batch_level_dataloader or valid_supports_prebatched) and hasattr(
                    self.validdataset, "_get_batch_level_items"
                ):
                    self.validdataset._batch_level_dataloader_enabled = True
                    _val_collate_fn = prebatched_collate_fn
                elif self.validdataset.__class__ is BaseDataset:
                    self.validdataset._return_static_attrs_separately = True
                    _val_collate_fn = static_attr_batch_collate_fn
            elif use_batch_level_dataloader:
                warnings.warn(
                    "batch_level_dataloader=True was ignored for validation "
                    "because this dataset uses a custom collate path or is "
                    "not the standard BaseDataset."
                )

            validation_data_loader = DataLoader(
                self.validdataset,
                batch_size=training_cfgs["batch_size"],
                shuffle=False,
                num_workers=worker_num,
                pin_memory=pin_memory,
                timeout=(
                    float(training_cfgs.get("dataloader_timeout", 0.0))
                    if worker_num > 0
                    else 0
                ),
                collate_fn=_val_collate_fn,
                **dataloader_extra_kwargs,
            )
            return data_loader, validation_data_loader

        return data_loader, None

    def _get_sampler(self, data_cfgs, training_cfgs, train_dataset):
        """
        return data sampler based on the provided configuration and training dataset.

        Parameters
        ----------
        data_cfgs : dict
            Configuration dictionary containing parameters for data sampling. Expected keys are:
            - "sampler": dict, containing:
            - "name": str, name of the sampler to use.
            - "sampler_hyperparam": dict, optional hyperparameters for the sampler.
        training_cfgs: dict
            Configuration dictionary containing parameters for training. Expected keys are:
            - "batch_size": int, size of each batch.
        train_dataset : Dataset
            The training dataset object which contains the data to be sampled. Expected attributes are:
            - ngrid: int, number of grids in the dataset.
            - nt: int, number of time steps in the dataset.
            - rho: int, length of the input sequence.
            - warmup_length: int, length of the warmup period.
            - horizon: int, length of the forecast horizon.

        Returns
        -------
        sampler_class
            An instance of the specified sampler class, initialized with the provided dataset and hyperparameters.

        Raises
        ------
        NotImplementedError
            If the specified sampler name is not found in the `data_sampler_dict`.
        """
        if data_cfgs["sampler"] is None:
            return None
        batch_size = training_cfgs["batch_size"]
        rho = train_dataset.rho
        warmup_length = train_dataset.warmup_length
        horizon = train_dataset.horizon
        ngrid = train_dataset.ngrid
        nt = train_dataset.nt
        sampler_name = data_cfgs["sampler"]
        if sampler_name not in data_sampler_dict:
            raise NotImplementedError(f"Sampler {sampler_name} not implemented yet")
        sampler_class = data_sampler_dict[sampler_name]
        sampler_hyperparam = dict(data_cfgs.get("sampler_hyperparam") or {})
        if sampler_name == "KuaiSampler":
            sampler_hyperparam |= {
                "batch_size": batch_size,
                "warmup_length": warmup_length,
                "rho_horizon": rho + horizon,
                "ngrid": ngrid,
                "nt": nt,
            }
        elif sampler_name == "WindowLenBatchSampler":
            sampler_hyperparam |= {
                "batch_size": batch_size,
            }
        elif sampler_name == "CloudZarrChunkBatchSampler":
            sampler_hyperparam |= {
                "batch_size": batch_size,
            }

        return sampler_class(train_dataset, **sampler_hyperparam)


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
        if mode == "train":
            model = super().load_model(mode)
        elif mode == "infer":
            self.weight_path = self._get_trained_model()
            model = self._load_model_from_pth()
            model.to(self.device)
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
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
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


model_type_dict = {
    "Normal": DeepHydro,
    "FedLearn": FedLearnHydro,
    "TransLearn": TransLearnHydro,
    "MTL": MultiTaskHydro,
}
