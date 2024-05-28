"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-05-04 11:30:00
LastEditors: Wenyu Ouyang
Description: Training function for DL models
FilePath: \torchhydro\torchhydro\trainers\train_logger.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from contextlib import contextmanager
from datetime import datetime
import json
import os
import time
from hydroutils import hydro_file
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def save_model(model, model_file, gpu_num=1):
    try:
        if torch.cuda.device_count() > 1 and gpu_num > 1:
            torch.save(model.module.state_dict(), model_file)
        else:
            torch.save(model.state_dict(), model_file)
    except RuntimeError:
        torch.save(model.module.state_dict(), model_file)


def save_model_params_log(params, params_log_path):
    time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
    params_log_file = os.path.join(params_log_path, f"{time_stamp}.json")
    hydro_file.serialize_json(params, params_log_file)


class TrainLogger:
    def __init__(self, model_filepath, params, opt):
        self.training_cfgs = params["training_cfgs"]
        self.data_cfgs = params["data_cfgs"]
        self.evaluation_cfgs = params["evaluation_cfgs"]
        self.model_cfgs = params["model_cfgs"]
        self.opt = opt
        self.training_save_dir = model_filepath
        self.tb = SummaryWriter(self.training_save_dir)
        self.session_params = []
        self.train_time = []
        # log loss for each epoch
        self.epoch_loss = []

    def save_session_param(
        self, epoch, total_loss, n_iter_ep, valid_loss=None, valid_metrics=None
    ):
        if valid_loss is None or valid_metrics is None:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "iter_num": n_iter_ep,
            }
        else:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "validation_loss": str(valid_loss),
                "validation_metric": valid_metrics,
                "iter_num": n_iter_ep,
            }
        epoch_params["train_time"] = self.train_time[epoch - 1]
        self.session_params.append(epoch_params)

    @contextmanager
    def log_epoch_train(self, epoch):
        start_time = time.time()
        logs = {}
        # here content in the with block will be performed
        yield logs
        total_loss = logs["train_loss"]
        elapsed_time = time.time() - start_time
        lr = self.opt.param_groups[0]["lr"]
        log_str = "Epoch {} Loss {:.4f} time {:.2f} lr {}".format(
            epoch, total_loss, elapsed_time, lr
        )
        print(log_str)
        model = logs["model"]
        print(model)
        self.tb.add_scalar("Loss", total_loss, epoch)
        # self.plot_hist_img(model, epoch)
        self.train_time.append(log_str)
        self.epoch_loss.append(total_loss)

    @contextmanager
    def log_epoch_valid(self, epoch):
        logs = {}
        yield logs
        valid_loss = logs["valid_loss"]
        if self.evaluation_cfgs["calc_metrics"]:
            valid_metrics = logs["valid_metrics"]
            val_log = "Epoch {} Valid Loss {:.4f} Valid Metric {}".format(
                epoch, valid_loss, valid_metrics
            )
            print(val_log)
            self.tb.add_scalar("ValidLoss", valid_loss, epoch)
            target_col = self.data_cfgs["target_cols"]
            evaluation_metrics = self.evaluation_cfgs["metrics"]
            for i in range(len(target_col)):
                for evaluation_metric in evaluation_metrics:
                    self.tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}mean",
                        np.mean(
                            valid_metrics[f"{evaluation_metric} of {target_col[i]}"]
                        ),
                        epoch,
                    )
                    self.tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}median",
                        np.median(
                            valid_metrics[f"{evaluation_metric} of {target_col[i]}"]
                        ),
                        epoch,
                    )
        else:
            val_log = "Epoch {} Valid Loss {:.4f} ".format(epoch, valid_loss)
            print(val_log)
            self.tb.add_scalar("ValidLoss", valid_loss, epoch)

    def save_model_and_params(self, model, epoch, params):
        final_epoch = params["training_cfgs"]["epochs"]
        save_epoch = params["training_cfgs"]["save_epoch"]
        if save_epoch is None or save_epoch == 0 and epoch != final_epoch:
            return
        if (save_epoch > 0 and epoch % save_epoch == 0) or epoch == final_epoch:
            # save for save_epoch
            model_file = os.path.join(
                self.training_save_dir, f"model_Ep{str(epoch)}.pth"
            )
            save_model(model, model_file)
        if epoch == final_epoch:
            self._save_final_epoch(params, model)

    def _save_final_epoch(self, params, model):
        # In final epoch, we save the model and params in test_path
        final_path = params["data_cfgs"]["test_path"]
        params["run"] = self.session_params
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_save_path = os.path.join(final_path, f"{time_stamp}_model.pth")
        save_model(model, model_save_path)
        save_model_params_log(params, final_path)
        # also save one for a training directory for one hyperparameter setting
        save_model_params_log(params, self.training_save_dir)

    def plot_hist_img(self, model, global_step):
        for tag, parm in model.named_parameters():
            self.tb.add_histogram(
                f"{tag}_hist", parm.detach().cpu().numpy(), global_step
            )
            if len(parm.shape) == 2:
                img_format = "HW"
                if parm.shape[0] > parm.shape[1]:
                    img_format = "WH"
                    self.tb.add_image(
                        f"{tag}_img",
                        parm.detach().cpu().numpy(),
                        global_step,
                        dataformats=img_format,
                    )

    def plot_model_structure(self, model):
        """plot model structure in tensorboard

        Parameters
        ----------
        model :
            torch model
        """
        # input4modelplot = torch.randn(
        #     self.data_cfgs["batch_size"],
        #     self.data_cfgs["forecast_history"],
        #     # self.model_cfgs["model_hyperparam"]["n_input_features"],
        #     self.model_cfgs["model_hyperparam"]["input_size"],
        # )
        if self.data_cfgs["model_mode"] == "single":
            input4modelplot = [
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_features"] - 1,
                ),
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["cnn_size"],
                ),
                torch.rand(
                    self.data_cfgs["batch_size"], 1, self.data_cfgs["output_features"]
                ),
            ]
        else:
            input4modelplot = [
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_features"],
                ),
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_size_encoder2"],
                ),
                torch.rand(
                    self.data_cfgs["batch_size"], 1, self.data_cfgs["output_features"]
                ),
            ]
        self.tb.add_graph(model, input4modelplot)
