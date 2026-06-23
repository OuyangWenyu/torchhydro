"""
Loss surface computation module.

This module handles the main computation of loss values on a 1D or 2D grid
for loss landscape visualization.

Adapted from tomgoldstein/loss-landscape.
"""

import os
import copy
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from torchhydro.explainers.loss_landscape import net_plotter
from torchhydro.explainers.loss_landscape import mpi_utils as mpi
from torchhydro.explainers.loss_landscape import scheduler
from torchhydro.explainers.loss_landscape.evaluation import eval_loss


def name_surface_file(
    dir_file,
    xmin,
    xmax,
    xnum,
    ymin=None,
    ymax=None,
    ynum=None,
    raw_data=False,
    data_split=1,
    split_idx=0,
):
    """
    Generate a name for the surface file.

    Parameters
    ----------
    dir_file : str
        Direction file path (used as prefix)
    xmin, xmax : float
        X-axis range
    xnum : int
        Number of points on x-axis
    ymin, ymax : float, optional
        Y-axis range
    ynum : int, optional
        Number of points on y-axis
    raw_data : bool
        Whether using raw data without normalization
    data_split : int
        Number of data splits
    split_idx : int
        Index of data split

    Returns
    -------
    str
        Surface file path
    """
    # Remove .h5 extension from dir_file if present
    surf_file = dir_file.replace(".h5", "")

    # Resolution
    surf_file += "_[%s,%s,%d]" % (str(xmin), str(xmax), int(xnum))
    if ymin is not None:
        surf_file += "x[%s,%s,%d]" % (str(ymin), str(ymax), int(ynum))

    # Dataloader parameters
    if raw_data:
        surf_file += "_rawdata"
    if data_split > 1:
        surf_file += "_datasplit=" + str(data_split) + "_splitidx=" + str(split_idx)

    return surf_file + ".h5"


def setup_surface_file(surf_file, dir_file, xcoordinates, ycoordinates=None):
    """
    Setup the HDF5 file to store the loss surface.

    Parameters
    ----------
    surf_file : str
        Path to surface file
    dir_file : str
        Path to direction file
    xcoordinates : np.ndarray
        X-axis coordinates
    ycoordinates : np.ndarray, optional
        Y-axis coordinates

    Returns
    -------
    str
        Surface file path
    """
    # Skip if the surface file already exists with coordinates
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, "r")
        if (
            ycoordinates is not None and "ycoordinates" in f.keys()
        ) or "xcoordinates" in f.keys():
            f.close()
            print("%s is already set up" % surf_file)
            return surf_file

    f = h5py.File(surf_file, "a")
    f["dir_file"] = dir_file
    f["xcoordinates"] = xcoordinates

    if ycoordinates is not None:
        f["ycoordinates"] = ycoordinates

    f.close()
    return surf_file


def crunch(
    surf_file,
    net,
    w,
    s,
    d,
    dataloader,
    loss_key,
    comm,
    rank,
    dir_type="weights",
    device="cpu",
    criterion=None,
    which_first_tensor="sequence",
):
    """
    Calculate the loss values of modified models in parallel using MPI.

    Parameters
    ----------
    surf_file : str
        Path to surface file
    net : torch.nn.Module
        The neural network model
    w : list
        Original weights
    s : dict
        Original state_dict
    d : list
        Directions: [xdirection] or [xdirection, ydirection]
    dataloader : DataLoader
        Data loader
    loss_key : str
        Key for loss values in HDF5 file (e.g., 'train_loss')
    comm : MPI.Comm or None
        MPI communicator
    rank : int
        MPI rank
    dir_type : str
        'weights' or 'states'
    device : str or torch.device
        Device
    criterion : torch.nn.Module
        Loss function
    which_first_tensor : str
        Tensor layout
    """
    if criterion is None:
        criterion = nn.MSELoss()

    f = h5py.File(surf_file, "r+" if rank == 0 else "r")
    losses = []
    xcoordinates = f["xcoordinates"][:]
    ycoordinates = f["ycoordinates"][:] if "ycoordinates" in f.keys() else None

    if loss_key not in f.keys():
        shape = (
            xcoordinates.shape
            if ycoordinates is None
            else (len(xcoordinates), len(ycoordinates))
        )
        losses = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
    else:
        losses = f[loss_key][:]

    # Get indices and coordinates for this rank
    inds, coords, inds_nums = scheduler.get_job_indices(
        losses, xcoordinates, ycoordinates, comm
    )

    print("Computing %d values for rank %d" % (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if dir_type == "weights":
            net_plotter.set_weights(net, w, d, coord)
        elif dir_type == "states":
            net_plotter.set_states(net, s, d, coord)

        # Compute loss
        loss_start = time.time()
        loss, _ = eval_loss(net, criterion, dataloader, device, which_first_tensor)
        loss_compute_time = time.time() - loss_start

        # Record the result
        losses.ravel()[ind] = loss

        # Send updated plot data to the master node
        sync_start = time.time()
        losses = mpi.reduce_max(comm, losses)
        sync_time = time.time() - sync_start
        total_sync += sync_time

        # Only the master node writes to the file
        if rank == 0:
            f[loss_key][:] = losses
            f.flush()

        print(
            "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s=%.3f \ttime=%.2f \tsync=%.2f"
            % (
                rank,
                count,
                len(inds),
                100.0 * count / len(inds) if len(inds) > 0 else 0,
                str(coord),
                loss_key,
                loss,
                loss_compute_time,
                sync_time,
            )
        )

    # Keep calling reduce for processes with less work to avoid blocking
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)

    total_time = time.time() - start_time
    print("Rank %d done! Total time: %.2f Sync: %.2f" % (rank, total_time, total_sync))

    f.close()


def compute_loss_surface(
    net,
    dataloader,
    output_dir,
    model_file,
    x_range=(-1, 1, 51),
    y_range=None,
    dir_type="weights",
    xnorm="filter",
    ynorm="filter",
    xignore="biasbn",
    yignore="biasbn",
    same_dir=False,
    criterion=None,
    device="cpu",
    which_first_tensor="sequence",
    use_mpi=False,
    ngpu=1,
):
    """
    Main function to compute loss surface.

    Parameters
    ----------
    net : torch.nn.Module
        The trained neural network model
    dataloader : DataLoader
        Data loader for computing loss
    output_dir : str
        Directory to save output files
    model_file : str
        Path to model file (used for naming)
    x_range : tuple
        (xmin, xmax, xnum) for x-axis
    y_range : tuple, optional
        (ymin, ymax, ynum) for y-axis. If None, 1D plot.
    dir_type : str
        'weights' or 'states'
    xnorm : str
        Normalization for x direction
    ynorm : str
        Normalization for y direction
    xignore : str
        Ignore setting for x direction
    yignore : str
        Ignore setting for y direction
    same_dir : bool
        Use same direction for x and y
    criterion : torch.nn.Module
        Loss function
    device : str or torch.device
        Device
    which_first_tensor : str
        Tensor layout
    use_mpi : bool
        Whether to use MPI for parallel computation
    ngpu : int
        Number of GPUs

    Returns
    -------
    str
        Path to the generated surface file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Setup MPI
    if use_mpi:
        comm = mpi.setup_MPI()
        rank = 0 if comm is None else comm.Get_rank()
        nproc = 1 if comm is None else comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # Parse ranges
    xmin, xmax, xnum = x_range
    xnum = int(xnum)

    y = y_range is not None
    if y:
        ymin, ymax, ynum = y_range
        ynum = int(ynum)
    else:
        ymin, ymax, ynum = None, None, None

    # Extract initial parameters
    w = net_plotter.get_weights(net)
    s = copy.deepcopy(net.state_dict())

    # Handle multi-GPU
    if ngpu > 1 and torch.cuda.is_available():
        net = nn.DataParallel(
            net, device_ids=range(min(ngpu, torch.cuda.device_count()))
        )

    # Setup direction file
    dir_file = os.path.join(
        output_dir,
        net_plotter.name_direction_file(
            os.path.basename(model_file),
            dir_type,
            xnorm,
            ynorm,
            xignore,
            yignore,
            same_dir,
            y,
        ),
    )

    if rank == 0:
        net_plotter.setup_direction(
            dir_file,
            net.module if ngpu > 1 else net,
            dir_type,
            xnorm,
            ynorm,
            xignore,
            yignore,
            same_dir,
            y,
        )

    # Wait for master to setup
    mpi.barrier(comm)

    # Setup surface file
    xcoordinates = np.linspace(xmin, xmax, num=xnum)
    ycoordinates = np.linspace(ymin, ymax, num=ynum) if y else None

    surf_file = os.path.join(
        output_dir,
        os.path.basename(
            name_surface_file(dir_file, xmin, xmax, xnum, ymin, ymax, ynum)
        ),
    )

    if rank == 0:
        setup_surface_file(surf_file, dir_file, xcoordinates, ycoordinates)

    mpi.barrier(comm)

    # Load directions
    d = net_plotter.load_directions(dir_file)

    # Compute surface
    if criterion is None:
        criterion = nn.MSELoss()

    crunch(
        surf_file,
        net.module if ngpu > 1 else net,
        w,
        s,
        d,
        dataloader,
        "train_loss",
        comm,
        rank,
        dir_type,
        device,
        criterion,
        which_first_tensor,
    )

    return surf_file
