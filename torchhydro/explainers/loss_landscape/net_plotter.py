"""
Network parameter manipulation and direction generation for loss landscape visualization.

Adapted from tomgoldstein/loss-landscape.
"""

import os
import copy
from os.path import exists, commonprefix

import h5py
import torch

from torchhydro.explainers.loss_landscape import h5_util


def get_weights(net):
    """
    Extract parameters from net and return a list of tensors.

    NOTE: Returns CLONES of the parameters to avoid mutation issues.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model

    Returns
    -------
    list
        List of parameter tensors (cloned)
    """
    return [p.data.clone() for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
    """
    Overwrite the network's weights with a specified list of tensors
    or change weights along directions with a step size.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model
    weights : list
        List of weight tensors (original weights, should be cloned)
    directions : list, optional
        List of direction tensors [xdirection] or [xdirection, ydirection]
    step : float or tuple, optional
        Step size along direction(s)
    """
    if directions is None:
        # Simply copy weights
        for p, w in zip(net.parameters(), weights):
            p.data.copy_(w.type_as(p.data))
    else:
        assert (
            step is not None
        ), "If a direction is specified then step must be specified as well"

        if len(directions) == 2:
            # dx and dy are the direction tensors
            dx = directions[0]
            dy = directions[1]

            # Ensure step is a tuple and matches the number of directions
            assert (
                isinstance(step, tuple) and len(step) == 2
            ), "Step must be a tuple with two values for two directions"

            # Calculate the change in weights based on directions and step
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            # Only one direction is provided, apply step
            changes = [d * step for d in directions[0]]

        # Update the weights using the calculated changes
        for p, w, d in zip(net.parameters(), weights, changes):
            p.data.copy_(w + torch.Tensor(d).type_as(w))


def set_states(net, states, directions=None, step=None):
    """
    Overwrite the network's state_dict or change it along directions with a step size.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model
    states : dict
        Original state_dict
    directions : list, optional
        List of direction tensors
    step : float or tuple, optional
        Step size along direction(s)
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert (
            step is not None
        ), "If direction is provided then the step must be specified as well"

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d * step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert len(new_states) == len(changes)

        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


def get_random_weights(weights):
    """
    Produce a random direction that is a list of random Gaussian tensors
    with the same shape as the network's weights.

    Parameters
    ----------
    weights : list
        List of weight tensors

    Returns
    -------
    list
        List of random direction tensors (on same device as weights)
    """
    return [torch.randn(w.size(), device=w.device, dtype=w.dtype) for w in weights]


def get_random_states(states):
    """
    Produce a random direction that is a list of random Gaussian tensors
    with the same shape as the network's state_dict.

    Parameters
    ----------
    states : dict
        Network state_dict

    Returns
    -------
    list
        List of random direction tensors (on same device as state values)
    """
    return [
        torch.randn(w.size(), device=w.device, dtype=w.dtype) for k, w in states.items()
    ]


def get_diff_weights(weights, weights2):
    """Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


def normalize_direction(direction, weights, norm="filter"):
    """
    Rescale the direction so that it has similar norm as the corresponding
    model weights in different levels.

    NOTE: direction and weights are single layer tensors (e.g., [out_channels, in_channels, H, W])
    For 'filter' norm, we iterate over the first dimension (filters).

    Parameters
    ----------
    direction : torch.Tensor
        The random direction for one layer
    weights : torch.Tensor
        The original model weights for one layer
    norm : str
        Normalization method: 'filter' | 'layer' | 'weight' | 'dfilter' | 'dlayer'
    """
    if norm == "filter":
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == "layer":
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == "weight":
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == "dfilter":
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == "dlayer":
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(
    direction, weights, norm="filter", ignore="biasbn"
):
    """
    Normalize direction entries according to the entries of weights.

    Parameters
    ----------
    direction : list
        List of direction tensors
    weights : list
        List of weight tensors
    norm : str
        Normalization method
    ignore : str
        'biasbn' to ignore biases and BN parameters
    """
    assert len(direction) == len(weights)

    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm="filter", ignore="ignore"):
    """Normalize direction entries for state_dict."""
    assert len(direction) == len(states)

    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def ignore_biasbn(directions):
    """Set bias and bn parameters in directions to zero."""
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)


def create_target_direction(net, net2, dir_type="states"):
    """
    Create a target direction from one model to another.

    Parameters
    ----------
    net : torch.nn.Module
        The source model
    net2 : torch.nn.Module
        The target model with the same architecture
    dir_type : str
        'weights' or 'states'

    Returns
    -------
    list
        The target direction
    """
    assert net2 is not None

    if dir_type == "weights":
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == "states":
        s = net.state_dict()
        s2 = net2.state_dict()
        direction = get_diff_states(s, s2)

    return direction


def create_random_direction(net, dir_type="weights", ignore="biasbn", norm="filter"):
    """
    Create a random (normalized) direction with the same dimension as
    the weights or states.

    Parameters
    ----------
    net : torch.nn.Module
        The trained model
    dir_type : str
        'weights' or 'states'
    ignore : str
        'biasbn' to ignore biases and BN parameters
    norm : str
        Direction normalization method: 'filter' | 'layer' | 'weight' | 'dlayer' | 'dfilter'

    Returns
    -------
    list
        A random direction with the same dimension as weights or states
    """
    if dir_type == "weights":
        weights = get_weights(net)
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == "states":
        states = net.state_dict()
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def setup_direction(
    dir_file,
    net,
    dir_type="weights",
    xnorm="filter",
    ynorm="filter",
    xignore="biasbn",
    yignore="biasbn",
    same_dir=False,
    y=False,
    net2=None,
    net3=None,
):
    """
    Setup the h5 file to store the directions.

    Parameters
    ----------
    dir_file : str
        Path to direction file
    net : torch.nn.Module
        The trained model
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
    y : bool
        Whether to create y direction (2D plot)
    net2 : torch.nn.Module, optional
        Target model for x direction
    net3 : torch.nn.Module, optional
        Target model for y direction
    """
    print("-------------------------------------------------------------------")
    print("setup_direction")
    print("-------------------------------------------------------------------")

    # Setup env for preventing lock on h5py file
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, "r")
        if (y and "ydirection" in f.keys()) or "xdirection" in f.keys():
            f.close()
            print("%s is already set up" % dir_file)
            return
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file, "w")

    print("Setting up the plotting directions...")
    if net2 is not None:
        xdirection = create_target_direction(net, net2, dir_type)
    else:
        xdirection = create_random_direction(net, dir_type, xignore, xnorm)
    h5_util.write_list(f, "xdirection", xdirection)

    if y:
        if same_dir:
            ydirection = xdirection
        elif net3 is not None:
            ydirection = create_target_direction(net, net3, dir_type)
        else:
            ydirection = create_random_direction(net, dir_type, yignore, ynorm)
        h5_util.write_list(f, "ydirection", ydirection)

    f.close()
    print("direction file created: %s" % dir_file)


def name_direction_file(
    model_file,
    dir_type="weights",
    xnorm="",
    ynorm="",
    xignore="",
    yignore="",
    same_dir=False,
    y=False,
    idx=0,
):
    """
    Name the direction file that stores the random directions.

    Parameters
    ----------
    model_file : str
        Path to model file
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
    y : bool
        Whether using y direction
    idx : int
        Index for repeated experiments

    Returns
    -------
    str
        Direction file path
    """
    dir_file = model_file + "_" + dir_type

    if xignore:
        dir_file += "_xignore=" + xignore
    if xnorm:
        dir_file += "_xnorm=" + xnorm

    if y:
        if yignore:
            dir_file += "_yignore=" + yignore
        if ynorm:
            dir_file += "_ynorm=" + ynorm
        if same_dir:
            dir_file += "_same_dir"

    if idx > 0:
        dir_file += "_idx=" + str(idx)

    dir_file += ".h5"

    return dir_file


def load_directions(dir_file):
    """
    Load direction(s) from the direction file.

    Parameters
    ----------
    dir_file : str
        Path to direction file

    Returns
    -------
    list
        List of directions: [xdirection] or [xdirection, ydirection]
    """
    f = h5py.File(dir_file, "r")

    if "ydirection" in f.keys():
        xdirection = h5_util.read_list(f, "xdirection")
        ydirection = h5_util.read_list(f, "ydirection")
        directions = [xdirection, ydirection]
    else:
        directions = [h5_util.read_list(f, "xdirection")]

    f.close()
    return directions
