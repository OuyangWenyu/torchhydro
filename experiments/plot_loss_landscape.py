"""
Example: Visualizing Loss Landscape for TorchHydro Models

This script demonstrates how to compute and visualize the loss landscape
of a trained hydrological deep learning model.

Usage:
    python plot_loss_landscape.py

    For MPI parallel execution:
    mpirun -n 4 python plot_loss_landscape.py --mpi
"""

import argparse
import os
import copy

import torch
import torch.nn as nn

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import _get_deep_hydro
from torchhydro.explainers.loss_landscape import (
    compute_loss_surface,
    plot_1d_loss,
    plot_2d_contour,
    plot_3d_surface,
    plot_all_2d,
    setup_MPI,
    barrier,
    is_master,
)
from hydrodataset.hydro_dataset import StandardVariable


class RmseLoss(torch.nn.Module):
    def __init__(self):
        """
        RMSE loss which could ignore NaN values

        Now we only support 3-d tensor and 1-d tensor
        """
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        if target.dim() == 1:
            mask = target == target
            p = output[mask]
            t = target[mask]
            return torch.sqrt(((p - t) ** 2).mean())
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            p = torch.where(torch.isnan(p), torch.full_like(p, 0), p)
            t = t0[mask]
            t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
            temp = torch.sqrt(((p - t) ** 2).mean())
            loss = loss + temp
        return loss


def main():
    parser = argparse.ArgumentParser(description="Visualize Loss Landscape")
    parser.add_argument(
        "--mpi", "-m", action="store_true", help="Use MPI for parallel computation"
    )
    parser.add_argument("--cuda", "-c", action="store_true", help="Use CUDA")
    parser.add_argument("--x", default="-1:1:21", help="X range: xmin:xmax:xnum")
    parser.add_argument(
        "--y", default=None, help="Y range: ymin:ymax:ynum (for 2D plot)"
    )
    parser.add_argument(
        "--dir_type", default="weights", help="Direction type: weights | states"
    )
    parser.add_argument(
        "--xnorm", default="filter", help="X normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--ynorm", default="filter", help="Y normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--output_dir", default="./loss_landscape_output", help="Output directory"
    )
    parser.add_argument(
        "--vmin", default=0.1, type=float, help="Minimum value for plotting"
    )
    parser.add_argument(
        "--vmax", default=10, type=float, help="Maximum value for plotting"
    )
    parser.add_argument("--show", action="store_true", help="Show plots")

    args = parser.parse_args()

    torch.manual_seed(123)
    # Setup MPI if requested
    if args.mpi:
        comm = setup_MPI()
        rank = 0 if comm is None else comm.Get_rank()
    else:
        comm, rank = None, 0

    # Parse coordinate ranges
    xmin, xmax, xnum = [float(a) for a in args.x.split(":")]
    x_range = (xmin, xmax, int(xnum))

    y_range = None
    if args.y:
        ymin, ymax, ynum = [float(a) for a in args.y.split(":")]
        y_range = (ymin, ymax, int(ynum))

    # Device setup
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------------
    # Configure your model and data here
    # ------------------------------------------------------------------------
    source_path = SETTING["local_data_path"]["datasets-origin"]

    config_args = cmd(
        train_mode=False,
        stat_dict_file=r"/Users/cylenlc/work/torchhydro/results/results/SimpleLSTM/dapengscaler_stat.json",
        sub=os.path.join("results", "SimpleLSTM"),
        source_cfgs={"source_name": "camels_us", "source_path": source_path},
        ctx=[-1] if device == "cpu" else [0],
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 23,
            "output_size": 1,
            "hidden_size": 128,
            "dr": 0.4,
        },
        gage_id=[
            "01022500",
            "01030500",
            "01047000",
            "01054200",
            "01057000",
            "01073000",
        ],
        batch_size=256,
        train_epoch=20,
        save_epoch=1,
        hindcast_length=0,
        forecast_length=365,
        min_time_unit="D",
        min_time_interval="1",
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.DAYLIGHT_DURATION,
            StandardVariable.SOLAR_RADIATION,
            StandardVariable.TEMPERATURE_MAX,
            StandardVariable.TEMPERATURE_MIN,
            StandardVariable.VAPOR_PRESSURE,
        ],
        var_out=[StandardVariable.STREAMFLOW],
        dataset="StreamflowDataset",
        scaler="DapengScaler",
        model_loader={
            "load_way": "pth",
            "pth_path": f"/Users/cylenlc/work/torchhydro/results/results/SimpleLSTM/best_model.pth",
        },
        train_period=["2000-10-01", "2001-10-01"],
        valid_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0001},
        lr_scheduler={"lr_factor": 0.95},
        which_first_tensor="sequence",
        evaluator={"eval_way": "1pace", "pace_idx": -1},
        early_stopping=True,
        rolling=0,
        patience=2,
        model_type="Normal",
    )

    # Load config and get model
    config_data = default_config_file()
    update_cfg(config_data, config_args)

    deep_hydro = _get_deep_hydro(config_data)
    net = deep_hydro.load_model(mode="infer")
    net = net.to(device)

    # Get dataloader for training data (used to compute loss surface)
    from torch.utils.data import DataLoader

    train_dataset = deep_hydro.make_dataset("train")
    dataloader = DataLoader(
        train_dataset,
        batch_size=config_data["training_cfgs"]["batch_size"],
        shuffle=False,
    )

    # Define loss function
    criterion = nn.MSELoss()
    # criterion = RmseLoss

    # Model file path (for naming)
    model_file = config_data["evaluation_cfgs"]["model_loader"]["pth_path"]

    # ------------------------------------------------------------------------
    # Compute loss surface
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("Computing Loss Surface")
    print("=" * 70)

    surf_file = compute_loss_surface(
        net=net,
        dataloader=dataloader,
        output_dir=args.output_dir,
        model_file=model_file,
        x_range=x_range,
        y_range=y_range,
        dir_type=args.dir_type,
        xnorm=args.xnorm,
        ynorm=args.ynorm if y_range else "",
        criterion=criterion,
        device=device,
        which_first_tensor=config_data["training_cfgs"]["which_first_tensor"],
        use_mpi=args.mpi,
    )

    # Wait for all processes
    barrier(comm)

    # ------------------------------------------------------------------------
    # Generate plots (only on master)
    # ------------------------------------------------------------------------
    if is_master(comm):
        print("=" * 70)
        print("Generating Plots")
        print("=" * 70)

        if y_range is None:
            # 1D plot
            plot_1d_loss(
                surf_file,
                xmin=xmin,
                xmax=xmax,
                loss_max=args.vmax,
                show=args.show,
            )
        else:
            # 2D plots
            plot_all_2d(
                surf_file,
                vmin=args.vmin,
                vmax=args.vmax,
                show=args.show,
            )

        print("=" * 70)
        print("Done! Output files saved to:", args.output_dir)
        print("=" * 70)


if __name__ == "__main__":
    main()
