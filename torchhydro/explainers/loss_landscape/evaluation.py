"""
Loss evaluation for hydrological deep learning models.

Adapted from tomgoldstein/loss-landscape for regression tasks.
"""

import torch
import torch.nn as nn


def eval_loss(net, criterion, loader, device="cpu", which_first_tensor="sequence"):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    This function is adapted for hydrological regression models where we compute
    MSE or other regression losses instead of classification accuracy.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model
    criterion : torch.nn.Module
        Loss function (e.g., nn.MSELoss, nn.L1Loss)
    loader : torch.utils.data.DataLoader
        DataLoader providing the dataset
    device : str or torch.device
        Device to run evaluation on
    which_first_tensor : str
        'sequence' or 'batch', indicating tensor layout

    Returns
    -------
    tuple
        (average_loss, None) - second element is None since we don't compute accuracy for regression
    """
    total_loss = 0.0
    total_samples = 0
    num_batch = len(loader)

    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    # (x, c, y) format with static features
                    inputs, static, targets = batch
                    inputs = (inputs, static)
                else:
                    raise ValueError(
                        f"Unexpected batch format with {len(batch)} elements"
                    )
            else:
                raise ValueError("Batch should be a tuple or list")

            # Move to device
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
                batch_size = (
                    inputs[0].size(0)
                    if which_first_tensor == "batch"
                    else inputs[0].size(1)
                )
            else:
                inputs = inputs.to(device)
                batch_size = (
                    inputs.size(0) if which_first_tensor == "batch" else inputs.size(1)
                )

            targets = targets.to(device)

            # Forward pass
            if isinstance(inputs, tuple):
                outputs = net(*inputs)
            else:
                outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # Return (loss, None) - no accuracy for regression
    return avg_loss, None


def eval_loss_simple(net, criterion, loader, device="cpu"):
    """
    Simplified loss evaluation for standard (input, target) batches.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model
    criterion : torch.nn.Module
        Loss function
    loader : torch.utils.data.DataLoader
        DataLoader
    device : str or torch.device
        Device

    Returns
    -------
    float
        Average loss value
    """
    total_loss = 0.0
    total_samples = 0

    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0
