"""
Serialization and deserialization of directions in HDF5 files.

This module provides utilities for reading and writing lists of tensors
to HDF5 files, used for storing direction vectors.
"""

import torch
import numpy as np


def write_list(f, name, direction):
    """
    Save the direction to the hdf5 file with name as the key.

    Parameters
    ----------
    f : h5py.File
        HDF5 file object
    name : str
        Key name for the direction group
    direction : list
        A list of tensors representing the direction
    """
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            # Move to CPU before converting to numpy
            l = l.detach().cpu().numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """
    Read group with name as the key from the hdf5 file and return a list of numpy arrays.

    Parameters
    ----------
    f : h5py.File
        HDF5 file object
    name : str
        Key name for the direction group

    Returns
    -------
    list
        List of numpy arrays
    """
    grp = f[name]
    return [np.array(grp[str(i)]) for i in range(len(grp))]
