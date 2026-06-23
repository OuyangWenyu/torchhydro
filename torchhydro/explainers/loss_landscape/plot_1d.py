"""
1D plotting functions for loss landscape visualization.

Adapted from tomgoldstein/loss-landscape.
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt


def plot_1d_loss(
    surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False, save_path=None
):
    """
    Plot 1D loss curve.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    xmin : float
        Minimum x value to display
    xmax : float
        Maximum x value to display
    loss_max : float
        Maximum loss value to display
    log : bool
        Use log scale for loss values
    show : bool
        Show the plot
    save_path : str, optional
        Path to save the figure. If None, saves next to surf_file.

    Returns
    -------
    str
        Path to saved figure
    """
    print("------------------------------------------------------------------")
    print("plot_1d_loss")
    print("------------------------------------------------------------------")

    f = h5py.File(surf_file, "r")
    print(list(f.keys()))

    x = f["xcoordinates"][:]
    assert "train_loss" in f.keys(), "'train_loss' does not exist"
    train_loss = f["train_loss"][:]

    print("train_loss")
    print(train_loss)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    # Plot loss curve
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if log:
        ax1.semilogy(x, train_loss, "b-", label="Training loss", linewidth=2)
    else:
        ax1.plot(x, train_loss, "b-", label="Training loss", linewidth=2)

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(0, loss_max)
    ax1.set_xlabel("Direction", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.legend(loc="upper right", fontsize=12)
    ax1.grid(True, alpha=0.3)

    plt.title("1D Loss Landscape", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = surf_file.replace(
            ".h5", "_1d_loss" + ("_log" if log else "") + ".pdf"
        )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    f.close()
    plt.close(fig)

    return save_path


def plot_1d_loss_comparison(
    surf_files,
    labels=None,
    xmin=-1.0,
    xmax=1.0,
    loss_max=5,
    log=False,
    show=False,
    save_path=None,
):
    """
    Plot multiple 1D loss curves for comparison.

    Parameters
    ----------
    surf_files : list
        List of surface file paths
    labels : list, optional
        Labels for each curve
    xmin : float
        Minimum x value
    xmax : float
        Maximum x value
    loss_max : float
        Maximum loss value
    log : bool
        Use log scale
    show : bool
        Show the plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    str
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(surf_files)))

    for i, surf_file in enumerate(surf_files):
        f = h5py.File(surf_file, "r")
        x = f["xcoordinates"][:]
        train_loss = f["train_loss"][:]
        label = labels[i] if labels else f"Model {i+1}"

        if log:
            ax.semilogy(x, train_loss, color=colors[i], label=label, linewidth=2)
        else:
            ax.plot(x, train_loss, color=colors[i], label=label, linewidth=2)

        f.close()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, loss_max)
    ax.set_xlabel("Direction", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.title("1D Loss Landscape Comparison", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = surf_files[0].replace(".h5", "_1d_comparison.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    return save_path
