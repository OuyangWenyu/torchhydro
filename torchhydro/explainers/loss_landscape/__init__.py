"""
Loss Landscape Visualization Module for TorchHydro

This module provides tools to visualize the loss surface of hydrological
deep learning models. Adapted from tomgoldstein/loss-landscape.

Main functions:
- compute_loss_surface: Compute loss values on a 1D or 2D grid
- plot_1d_loss: Plot 1D loss curve
- plot_2d_contour: Plot 2D contour map
- plot_3d_surface: Plot 3D surface visualization
"""

from torchhydro.explainers.loss_landscape.net_plotter import (
    get_weights,
    set_weights,
    set_states,
    create_random_direction,
    create_target_direction,
    setup_direction,
    load_directions,
    name_direction_file,
)
from torchhydro.explainers.loss_landscape.evaluation import eval_loss
from torchhydro.explainers.loss_landscape.surface import (
    compute_loss_surface,
    setup_surface_file,
    name_surface_file,
)
from torchhydro.explainers.loss_landscape.plot_1d import plot_1d_loss
from torchhydro.explainers.loss_landscape.plot_2d import (
    plot_2d_contour,
    plot_2d_contourf,
    plot_2d_heatmap,
    plot_3d_surface,
    plot_all_2d,
)
from torchhydro.explainers.loss_landscape.mpi_utils import (
    setup_MPI,
    barrier,
    reduce_max,
    is_master,
)

__all__ = [
    # Net plotter
    "get_weights",
    "set_weights",
    "set_states",
    "create_random_direction",
    "create_target_direction",
    "setup_direction",
    "load_directions",
    "name_direction_file",
    # Evaluation
    "eval_loss",
    # Surface
    "compute_loss_surface",
    "setup_surface_file",
    "name_surface_file",
    # Plotting
    "plot_1d_loss",
    "plot_2d_contour",
    "plot_2d_contourf",
    "plot_2d_heatmap",
    "plot_3d_surface",
    "plot_all_2d",
    # MPI
    "setup_MPI",
    "barrier",
    "reduce_max",
    "is_master",
]
