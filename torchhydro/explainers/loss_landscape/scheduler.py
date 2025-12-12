"""
Task scheduler for distributing jobs across MPI workers.

Adapted from tomgoldstein/loss-landscape.
"""

import numpy as np


def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    """
    Get indices of grid points that have not yet been calculated.

    Parameters
    ----------
    vals : np.ndarray
        Values at (x, y), with value -1 when the value is not yet calculated.
    xcoordinates : np.ndarray
        x locations, e.g., [-1, -0.5, 0, 0.5, 1]
    ycoordinates : np.ndarray, optional
        y locations, e.g., [-1, -0.5, 0, 0.5, 1]

    Returns
    -------
    inds : np.ndarray
        Indices into vals for points that have not yet been calculated.
    coords : np.ndarray
        Corresponding coordinates, with one x/y coordinate per row.
    """
    # Create a list of indices into the vectorized vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coordinates of the points to be plotted
    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1, s2]
    else:
        return inds, xcoordinates.ravel()[inds]


def split_inds(num_inds, nproc):
    """
    Evenly slice out a set of jobs that are handled by each MPI process.

    Parameters
    ----------
    num_inds : int
        Total number of indices/jobs
    nproc : int
        Number of MPI processes

    Returns
    -------
    list
        List of ranges, one per process
    """
    chunk = num_inds // nproc
    remainder = num_inds % nproc
    splitted_idx = []

    for rank in range(0, nproc):
        # Set the starting index for this slice
        start_idx = rank * chunk + min(rank, remainder)
        # The stopping index can't go beyond the end of the array
        stop_idx = start_idx + chunk + (rank < remainder)
        splitted_idx.append(range(start_idx, stop_idx))

    return splitted_idx


def get_job_indices(vals, xcoordinates, ycoordinates, comm):
    """
    Prepare the job indices for the current MPI rank.

    Parameters
    ----------
    vals : np.ndarray
        The value matrix
    xcoordinates : np.ndarray
        x locations
    ycoordinates : np.ndarray or None
        y locations
    comm : MPI.Comm or None
        MPI communicator

    Returns
    -------
    inds : np.ndarray
        Indices assigned to current rank
    coords : np.ndarray
        Coordinates for current rank
    inds_nums : list
        Number of indices for each rank
    """
    inds, coords = get_unplotted_indices(vals, xcoordinates, ycoordinates)

    rank = 0 if comm is None else comm.Get_rank()
    nproc = 1 if comm is None else comm.Get_size()
    splitted_idx = split_inds(len(inds), nproc)

    # Split the indices over the available MPI processes
    inds = inds[splitted_idx[rank]]
    coords = coords[splitted_idx[rank]]

    # Figure out the number of jobs that each MPI process needs to calculate
    inds_nums = [len(idx) for idx in splitted_idx]

    return inds, coords, inds_nums
