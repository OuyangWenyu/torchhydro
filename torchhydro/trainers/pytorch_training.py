"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-09-24 15:25:57
LastEditors: Wenyu Ouyang
Description: Training function for DL models
FilePath: \torchhydro\torchhydro\trainers\pytorch_training.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from hydroutils.hydro_stat import stat_error
from hydroutils import hydro_file
from torchhydro.datasets.data_sets import KuaiSampler
from torchhydro.models.model_dict_function import (
    pytorch_opt_dict,
    pytorch_criterion_dict,
)
from torchhydro.models.crits import GaussianLoss
from torchhydro.trainers.time_model import PyTorchForecast
from torchhydro.trainers.train_utils import model_infer, denormalize4eval, EarlyStopper


def model_train(forecast_model: PyTorchForecast) -> None:
    """
    Function to train any PyTorchForecast model

    Parameters
    ----------
    forecast_model
        A properly wrapped PyTorchForecast model

    Returns
    -------
    None

    Raises
    -------
    ValueError
        if nan values exist, raise error
    """
    # A dictionary of the necessary parameters for training
    training_params = forecast_model.params["training_params"]
    # The file path to load model weights from; defaults to "model_save"
    model_filepath = forecast_model.params["data_params"]["test_path"]
    data_params = forecast_model.params["data_params"]
    es = None
    if "early_stopping" in forecast_model.params:
        es = EarlyStopper(forecast_model.params["early_stopping"]["patience"])
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
    criterion = pytorch_criterion_dict[training_params["criterion"]](
        **criterion_init_params
    )
    params_in_opt = forecast_model.model.parameters()
    opt = pytorch_opt_dict[training_params["optimizer"]](
        params_in_opt, **training_params["optim_params"]
    )
    lr_scheduler = training_params["lr_scheduler"]
    max_epochs = training_params["epochs"]
    save_epoch = training_params["save_epoch"]
    start_epoch = training_params["start_epoch"]
    # use PyTorch's DataLoader to load the data into batches in each epoch
    data_loader, validation_data_loader = get_train_valid_dataloader(
        forecast_model, training_params, data_params
    )
    session_params = []
    # use tensorboard to visualize the training process
    hyper_param_set = (
        "opt_"
        + training_params["optimizer"]
        + "_lr_"
        + str(opt.defaults["lr"])
        + "_bsize_"
        + str(training_params["batch_size"])
    )
    training_save_dir = os.path.join(model_filepath, hyper_param_set)
    tb = SummaryWriter(training_save_dir)
    param_save_dir = os.path.join(training_save_dir, "training_params")
    if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)
    for epoch in range(start_epoch, max_epochs + 1):
        if lr_scheduler is not None and epoch in lr_scheduler.keys():
            # now we only support manual setting lr scheduler
            for param_group in opt.param_groups:
                param_group["lr"] = lr_scheduler[epoch]
        t0 = time.time()
        total_loss, n_iter_ep = torch_single_train(
            forecast_model.model,
            opt,
            criterion,
            data_loader,
            device=forecast_model.device,
            writer=tb,
            i_epoch=epoch,
            which_first_tensor=training_params["which_first_tensor"],
        )
        log_str = "Epoch {} Loss {:.3f} time {:.2f}".format(
            epoch, total_loss, time.time() - t0
        )
        tb.add_scalar("Loss", total_loss, epoch)
        print(log_str)
        if data_params["t_range_valid"] is not None:
            valid_obss_np, valid_preds_np, valid_loss = compute_validation(
                forecast_model.model,
                criterion,
                validation_data_loader,
                device=forecast_model.device,
                which_first_tensor=training_params["which_first_tensor"],
            )
            evaluation_metrics = forecast_model.params["evaluate_params"]["metrics"]
            fill_nan = forecast_model.params["evaluate_params"]["fill_nan"]
            target_col = forecast_model.params["data_params"]["target_cols"]
            valid_metrics = evaluate_validation(
                validation_data_loader,
                valid_preds_np,
                valid_obss_np,
                evaluation_metrics,
                fill_nan,
                target_col,
            )
            val_log = "Epoch {} Valid Loss {:.3f} Valid Metric {}".format(
                epoch, valid_loss, valid_metrics
            )
            tb.add_scalar("ValidLoss", valid_loss, epoch)
            for i in range(len(target_col)):
                for evaluation_metric in evaluation_metrics:
                    tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}mean",
                        np.mean(
                            valid_metrics[
                                f"{evaluation_metric} of {target_col[i]}"
                            ]
                        ),
                        epoch,
                    )
                    tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}median",
                        np.median(
                            valid_metrics[
                                f"{evaluation_metric} of {target_col[i]}"
                            ]
                        ),
                        epoch,
                    )
            print(val_log)
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "validation_loss": str(valid_loss),
                "validation_metric": valid_metrics,
                "time": log_str,
                "iter_num": n_iter_ep,
            }
            session_params.append(epoch_params)
            if es and not es.check_loss(forecast_model.model, valid_loss):
                print("Stopping model now")
                forecast_model.model.load_state_dict(torch.load("checkpoint.pth"))
                break
        else:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "time": log_str,
                "iter_num": n_iter_ep,
            }
            session_params.append(epoch_params)
        if save_epoch > 0 and epoch % save_epoch == 0:
            # save model
            model_file = os.path.join(model_filepath, f"model_Ep{str(epoch)}.pth")
            save_model(forecast_model.model, model_file)
            # sometimes we train a model in a directory with different hyperparameters
            # we want save models for each of the hyperparameter settings
            model_for_one_training_file = os.path.join(
                param_save_dir, f"model_Ep{str(epoch)}.pth"
            )
            save_model(forecast_model.model, model_for_one_training_file)
    tb.close()
    forecast_model.params["run"] = session_params
    forecast_model.save_model(model_filepath, max_epochs)
    save_model_params_log(forecast_model.params, training_save_dir)


def get_train_valid_dataloader(forecast_model, training_params, data_params):
    worker_num = 0
    pin_memory = False
    if "num_workers" in training_params:
        worker_num = training_params["num_workers"]
        print(f"using {str(worker_num)} workers")
    if "pin_memory" in training_params:
        pin_memory = training_params["pin_memory"]
        print(f"Pin memory set to {str(pin_memory)}")
    train_dataset = forecast_model.training
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
        valid_dataset = forecast_model.validation
        validation_data_loader = DataLoader(
            valid_dataset,
            batch_size=training_params["batch_size"],
            shuffle=False,
            num_workers=worker_num,
            pin_memory=pin_memory,
            timeout=0,
        )

    return data_loader, validation_data_loader


def save_model_params_log(params, save_log_path):
    params_save_path = os.path.join(
        save_log_path, f"params_log_{int(time.time())}.json"
    )
    hydro_file.serialize_json(params, params_save_path)


def save_model(model, model_file):
    try:
        torch.save(model.state_dict(), model_file)
    except RuntimeError:
        torch.save(model.module.state_dict(), model_file)


def evaluate_validation(
    validation_data_loader, output, labels, evaluation_metrics, fill_nan, target_col
):
    """
    calculate metrics for validation

    Parameters
    ----------
    output
        model output
    labels
        model target
    evaluation_metrics
        metrics to evaluate
    fill_nan
        how to fill nan
    target_col
        target columns

    Returns
    -------
    tuple
        metrics
    """
    if type(fill_nan) is list and len(fill_nan) != len(target_col):
        raise Exception("length of fill_nan must be equal to target_col's")
    eval_log = {}
    # renormalization to get real metrics
    preds_xr, obss_xr = denormalize4eval(validation_data_loader, output, labels)
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
            ].tolist()
    return eval_log


def compute_loss(
    labels: torch.Tensor, output: torch.Tensor, criterion, **kwargs
) -> float:
    """
    Function for computing the loss

    Parameters
    ----------
    labels
        The real values for the target. Shape can be variable but should follow (batch_size, time)
    output
        The output of the model
    criterion
        loss function
    validation_dataset
        Only passed when unscaling of data is needed.
    m
        defaults to 1

    Returns
    -------
    float
        the computed loss
    """
    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        return g_loss(labels)
    if (
        isinstance(output, torch.Tensor)
        and len(labels.shape) != len(output.shape)
        and len(labels.shape) > 1
    ):
        if labels.shape[1] == output.shape[1]:
            labels = labels.unsqueeze(2)
        else:
            labels = labels.unsqueeze(0)
    assert labels.shape == output.shape
    return criterion(output, labels.float())


def torch_single_train(
    model,
    opt: optim.Optimizer,
    criterion,
    data_loader: DataLoader,
    device=None,
    **kwargs,
) -> float:
    """
    Training function for one epoch

    Parameters
    ----------
    model
        a PyTorch model inherit from nn.Module
    opt
        optimizer function from PyTorch optim.Optimizer
    criterion
        loss function
    data_loader
        object for loading data to the model
    multi_targets
        with multi targets, we will use different loss function
    device
        where we put the tensors and models

    Returns
    -------
    tuple(float, int)
        loss of this epoch and number of all iterations

    Raises
    --------
    ValueError
        if nan exits, raise a ValueError
    """
    # we will set model.eval() in the validation function so here we should set model.train()
    model.train()
    writer = kwargs["writer"]
    n_iter_ep = 0
    running_loss = 0.0
    which_first_tensor = kwargs["which_first_tensor"]
    seq_first = which_first_tensor != "batch"
    pbar = tqdm(data_loader)
    i_epoch = kwargs["i_epoch"]

    for _, (src, trg) in enumerate(pbar):
        # iEpoch starts from 1, iIter starts from 0, we hope both start from 1
        trg, output = model_infer(seq_first, device, model, src, trg)
        loss = compute_loss(trg, output, criterion, **kwargs)
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        model.zero_grad()
        if torch.isnan(loss) or loss == float("inf"):
            raise ValueError(
                "Error infinite or NaN loss detected. Try normalizing data or performing interpolation"
            )
        running_loss += loss.item()
        n_iter_ep += 1
    plot_hist_img(model, writer, i_epoch)
    total_loss = running_loss / float(n_iter_ep)
    return total_loss, n_iter_ep


def plot_hist_img(model, writer, global_step):
    # TODO: a bug for add_histogram and add_image, maybe version problem, need to fix
    pass
    # for tag, parm in model.named_parameters():
    #     writer.add_histogram(tag + "_hist", parm.detach().cpu().numpy(), global_step)
    #     if len(parm.shape) == 2:
    #         img_format = "HW"
    #         if parm.shape[0] > parm.shape[1]:
    #             img_format = "WH"
    #             writer.add_image(
    #               tag + "_img",
    #               parm.detach().cpu().numpy(),
    #               global_step,
    #               dataformats=img_format,
    #             )


def compute_validation(
    model,
    criterion,
    data_loader: DataLoader,
    device: torch.device = None,
    **kwargs,
) -> float:
    """
    Function to compute the validation loss metrics

    Parameters
    ----------
    model
        the trained model
    criterion
        torch.nn.modules.loss
    dataloader
        The data-loader of either validation or test-data
    device
        torch.device

    Returns
    -------
    tuple
        validation observations (numpy array), predictions (numpy array) and the loss of validation
    """
    model.eval()
    seq_first = kwargs["which_first_tensor"] != "batch"
    obs = []
    preds = []
    with torch.no_grad():
        for src, trg in data_loader:
            trg, output = model_infer(seq_first, device, model, src, trg)
            obs.append(trg)
            preds.append(output)
        # first dim is batch
        obs_final = torch.cat(obs, dim=0)
        pred_final = torch.cat(preds, dim=0)
        valid_loss = compute_loss(obs_final, pred_final, criterion)
    y_obs = obs_final.detach().cpu().numpy()
    y_pred = pred_final.detach().cpu().numpy()
    return y_obs, y_pred, valid_loss
