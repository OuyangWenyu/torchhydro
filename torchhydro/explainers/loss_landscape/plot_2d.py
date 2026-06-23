"""
2D plotting functions for loss landscape visualization.

Adapted from tomgoldstein/loss-landscape.
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_2d_contour(
    surf_file,
    surf_name="train_loss",
    vmin=0.1,
    vmax=10,
    vlevel=0.5,
    show=False,
    save_path=None,
):
    """
    Plot 2D contour map of the loss surface.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    surf_name : str
        Name of the surface data in HDF5 file
    vmin : float
        Minimum value to map
    vmax : float
        Maximum value to map
    vlevel : float
        Contour level interval
    show : bool
        Show the plot
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    str
        Path to saved figure
    """
    print("------------------------------------------------------------------")
    print("plot_2d_contour")
    print("------------------------------------------------------------------")

    f = h5py.File(surf_file, "r")
    x = np.array(f["xcoordinates"][:])
    y = np.array(f["ycoordinates"][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    else:
        print("%s is not found in %s" % (surf_name, surf_file))
        f.close()
        return None

    print("loading surface file: " + surf_file)
    print("len(xcoordinates): %d   len(ycoordinates): %d" % (len(x), len(y)))
    print("max(%s) = %f \t min(%s) = %f" % (surf_name, np.max(Z), surf_name, np.min(Z)))

    if len(x) <= 1 or len(y) <= 1:
        print("The length of coordinates is not enough for plotting contours")
        f.close()
        return None

    # Plot 2D contours
    fig = plt.figure(figsize=(10, 8))
    CS = plt.contour(X, Y, Z, cmap="summer", levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xlabel("Direction 1", fontsize=14)
    plt.ylabel("Direction 2", fontsize=14)
    plt.title("Loss Landscape Contour", fontsize=16)
    plt.colorbar(CS)

    if save_path is None:
        save_path = surf_file.replace(".h5", "_" + surf_name + "_2dcontour.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    f.close()
    plt.close(fig)

    return save_path


def plot_2d_contourf(
    surf_file,
    surf_name="train_loss",
    vmin=0.1,
    vmax=10,
    vlevel=0.5,
    show=False,
    save_path=None,
):
    """
    Plot filled 2D contour map.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    surf_name : str
        Name of the surface data
    vmin : float
        Minimum value
    vmax : float
        Maximum value
    vlevel : float
        Contour level interval
    show : bool
        Show the plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    str
        Path to saved figure
    """
    f = h5py.File(surf_file, "r")
    x = np.array(f["xcoordinates"][:])
    y = np.array(f["ycoordinates"][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    else:
        print("%s is not found in %s" % (surf_name, surf_file))
        f.close()
        return None

    fig = plt.figure(figsize=(10, 8))
    CS = plt.contourf(X, Y, Z, cmap="summer", levels=np.arange(vmin, vmax, vlevel))
    plt.xlabel("Direction 1", fontsize=14)
    plt.ylabel("Direction 2", fontsize=14)
    plt.title("Loss Landscape Filled Contour", fontsize=16)
    plt.colorbar(CS)

    if save_path is None:
        save_path = surf_file.replace(".h5", "_" + surf_name + "_2dcontourf.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    f.close()
    plt.close(fig)

    return save_path


def plot_2d_heatmap(
    surf_file, surf_name="train_loss", vmin=0.1, vmax=10, show=False, save_path=None
):
    """
    Plot 2D heatmap of the loss surface.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    surf_name : str
        Name of the surface data
    vmin : float
        Minimum value
    vmax : float
        Maximum value
    show : bool
        Show the plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    str
        Path to saved figure
    """
    if not HAS_SEABORN:
        print("Seaborn not available, using matplotlib instead")
        return plot_2d_contourf(
            surf_file, surf_name, vmin, vmax, show=show, save_path=save_path
        )

    f = h5py.File(surf_file, "r")

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    else:
        print("%s is not found in %s" % (surf_name, surf_file))
        f.close()
        return None

    fig = plt.figure(figsize=(10, 8))
    sns_plot = sns.heatmap(
        Z,
        cmap="viridis",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
    )
    sns_plot.invert_yaxis()
    plt.xlabel("Direction 1", fontsize=14)
    plt.ylabel("Direction 2", fontsize=14)
    plt.title("Loss Landscape Heatmap", fontsize=16)

    if save_path is None:
        save_path = surf_file.replace(".h5", "_" + surf_name + "_2dheat.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    f.close()
    plt.close(fig)

    return save_path


def plot_3d_surface(
    surf_file, surf_name="train_loss", show=False, save_path=None, elev=30, azim=45
):
    """
    Plot 3D surface of the loss landscape.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    surf_name : str
        Name of the surface data
    show : bool
        Show the plot
    save_path : str, optional
        Path to save figure
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view

    Returns
    -------
    str
        Path to saved figure
    """
    f = h5py.File(surf_file, "r")
    x = np.array(f["xcoordinates"][:])
    y = np.array(f["ycoordinates"][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    else:
        print("%s is not found in %s" % (surf_name, surf_file))
        f.close()
        return None

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel("Direction 1", fontsize=12)
    ax.set_ylabel("Direction 2", fontsize=12)
    ax.set_zlabel("Loss", fontsize=12)
    ax.set_title("3D Loss Landscape", fontsize=16)
    ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, shrink=0.5, aspect=10)

    if save_path is None:
        save_path = surf_file.replace(".h5", "_" + surf_name + "_3dsurface.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {save_path}")

    if show:
        plt.show()

    f.close()
    plt.close(fig)

    return save_path


def plot_all_2d(
    surf_file,
    surf_name="train_loss",
    vmin=0.1,
    vmax=10,
    vlevel=0.5,
    show=False,
    output_dir=None,
):
    """
    Generate all 2D plots: contour, filled contour, heatmap, and 3D surface.

    Parameters
    ----------
    surf_file : str
        Path to surface HDF5 file
    surf_name : str
        Name of the surface data
    vmin : float
        Minimum value
    vmax : float
        Maximum value
    vlevel : float
        Contour level interval
    show : bool
        Show plots
    output_dir : str, optional
        Output directory for figures

    Returns
    -------
    list
        List of saved figure paths
    """
    import os

    saved_files = []

    # Generate save paths
    base_name = os.path.basename(surf_file).replace(".h5", "")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, base_name)
    else:
        prefix = surf_file.replace(".h5", "")

    # Contour
    path = plot_2d_contour(
        surf_file,
        surf_name,
        vmin,
        vmax,
        vlevel,
        show,
        save_path=prefix + "_contour.pdf",
    )
    if path:
        saved_files.append(path)

    # Filled contour
    path = plot_2d_contourf(
        surf_file,
        surf_name,
        vmin,
        vmax,
        vlevel,
        show,
        save_path=prefix + "_contourf.pdf",
    )
    if path:
        saved_files.append(path)

    # Heatmap
    path = plot_2d_heatmap(
        surf_file, surf_name, vmin, vmax, show, save_path=prefix + "_heatmap.pdf"
    )
    if path:
        saved_files.append(path)

    # 3D surface
    path = plot_3d_surface(surf_file, surf_name, show, save_path=prefix + "_3d.pdf")
    if path:
        saved_files.append(path)

    print(f"Generated {len(saved_files)} figures")
    return saved_files
