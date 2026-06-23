"""
MPI utilities for parallel loss surface computation.

This module provides convenience methods for using mpi4py with PyTorch.
Adapted from tomgoldstein/loss-landscape.

Note: MPI is optional. If mpi4py or MPI library is not available,
functions will work in single-process mode.
"""

import numpy as np

# Lazy MPI loading - don't import at module level to avoid errors
# when MPI library is not installed
_mpi4py = None
_MPI = None
_MPI_AVAILABLE = None


def _check_mpi():
    """Lazily check if MPI is available."""
    global _mpi4py, _MPI, _MPI_AVAILABLE

    if _MPI_AVAILABLE is not None:
        return _MPI_AVAILABLE

    try:
        import mpi4py
        from mpi4py import MPI

        _mpi4py = mpi4py
        _MPI = MPI
        _MPI_AVAILABLE = True
    except (ImportError, RuntimeError, OSError):
        # ImportError: mpi4py not installed
        # RuntimeError/OSError: MPI library not available
        _MPI_AVAILABLE = False

    return _MPI_AVAILABLE


def setup_MPI():
    """
    Setup MPI communicator.

    Returns
    -------
    comm : MPI.Comm or None
        MPI communicator object, or None if MPI is not available
    """
    if not _check_mpi():
        return None

    try:
        comm = _MPI.COMM_WORLD

        # Convert the Object to a Class so that it is possible to add attributes later
        class MPIComm(_mpi4py.MPI.Intracomm):
            pass

        comm = MPIComm(comm)
    except Exception:
        comm = None

    return comm


def print_once(comm, *message):
    """Print message only from rank 0."""
    if not comm or comm.Get_rank() == 0:
        print("".join(str(i) for i in message))


def is_master(comm):
    """Check if current process is master (rank 0)."""
    return not comm or comm.Get_rank() == 0


def get_rank(comm):
    """Get the rank of current process."""
    if not comm:
        return 0
    try:
        return comm.Get_rank()
    except Exception:
        return 0


def get_num_procs(comm):
    """Get the number of MPI processes."""
    if not comm:
        return 1
    try:
        return comm.Get_size()
    except Exception:
        return 1


def barrier(comm):
    """MPI barrier synchronization."""
    if comm:
        comm.barrier()


def allreduce_max(comm, array, display_info=False):
    """
    Allreduce operation with MAX.

    Parameters
    ----------
    comm : MPI.Comm or None
        MPI communicator
    array : np.ndarray
        Array to reduce
    display_info : bool
        Whether to display debug info

    Returns
    -------
    np.ndarray
        Reduced array
    """
    if not comm:
        return array

    array = np.asarray(array, dtype="d")
    total = np.zeros_like(array)
    float_min = np.finfo(np.float64).min
    total.fill(float_min)

    if display_info:
        print("(%d): sum=%f : size=%d" % (get_rank(comm), np.sum(array), array.nbytes))
        rows = str(comm.gather(array.shape[0]))
        cols = str(comm.gather(array.shape[1]) if array.ndim > 1 else "N/A")
        print_once(comm, "reduce: %s, %s" % (rows, cols))

    comm.Allreduce(array, total, op=_mpi4py.MPI.MAX)
    return total


def reduce_max(comm, array, display_info=False):
    """
    Reduce operation with MAX to root.

    Parameters
    ----------
    comm : MPI.Comm or None
        MPI communicator
    array : np.ndarray
        Array to reduce
    display_info : bool
        Whether to display debug info

    Returns
    -------
    np.ndarray
        Reduced array (valid only on root)
    """
    if not comm:
        return array

    array = np.asarray(array, dtype="d")
    total = np.zeros_like(array)
    float_min = np.finfo(np.float64).min
    total.fill(float_min)

    if display_info:
        print("(%d): sum=%f : size=%d" % (get_rank(comm), np.sum(array), array.nbytes))

    comm.Reduce(array, total, op=_mpi4py.MPI.MAX, root=0)
    return total


def reduce_min(comm, array, display_info=False):
    """Reduce operation with MIN to root."""
    if not comm:
        return array

    array = np.asarray(array, dtype="d")
    total = np.zeros_like(array)
    float_max = np.finfo(np.float64).max
    total.fill(float_max)

    comm.Reduce(array, total, op=_mpi4py.MPI.MIN, root=0)
    return total


def get_mpi_info():
    """Get MPI vendor information."""
    if _check_mpi():
        try:
            return _mpi4py.MPI.get_vendor()
        except Exception:
            return "mpi4py available, but vendor info unavailable"
    return "MPI not available"
