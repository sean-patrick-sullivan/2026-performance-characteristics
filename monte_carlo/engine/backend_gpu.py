"""
Monte Carlo computation of p-values using GPU architeture

This module implements a JIT-compiled and parallel computing of 
p-values for the JT and DD test in the hot loops of Monte Carlo studies.
"""

import math

import numpy as np
from numpy.typing import NDArray as Array
from numba import cuda, uint32, float64

from monte_carlo.stats import compute_stats_for_perm_gpu


# --- CPU compute kernel and launcher logic ---

@cuda.jit
def compute_kernel_gpu(k: int, n: int, data_array: Array[float], perm_array: Array[int], 
                       pval_array: Array[float]):
    """
    Compute p-values of JT and DD permutation tests when applied to num_simulation 
    supplied data sets.
    
    Parameters
    ----------
    k : uint16
        Number of groups
    n : uint16
        Number of observations per group
    data_array : float64[:,:]
        Simulation data of shape (num_simulations, k*n)
    perm_array : uint16[:,:]
        Permutation array of shape (num_perms, k*n)
    pval_array : float64[:,:]
        [Output] pre-allocated array of shape (num_simulations, 2) 
        where p-value results are written in-place
        
    Returns
    -------
    None
        This kernel modifies `pval_array` in-place
    
    Notes
    -----
    Running on GPU requires compatible hardware and drivers. This implementation assumes that 
    perm_array[0] reconstitutes the original order of the supplied sample data.
    """
    idx_sim = cuda.grid(1)
    num_simulations = data_array.shape[0]
    if idx_sim >= num_simulations:
        return

    num_perms = perm_array.shape[0]

    J_obs, D_obs = compute_stats_for_perm_gpu(
        k, n, data_array[idx_sim], perm_array[0]
    )

    count_J = uint32(0)
    count_D = uint32(0)

    for idx_perm in range(num_perms):
        J, D = compute_stats_for_perm_gpu(
            k, n, data_array[idx_sim], perm_array[idx_perm]
        )
        if J >= J_obs:
            count_J += uint32(1)
        if D >= D_obs:
            count_D += uint32(1)

    pval_array[idx_sim, 0] = float64(count_J) / float64(num_perms)
    pval_array[idx_sim, 1] = float64(count_D) / float64(num_perms)

def launch_compute_kernel_gpu(k: int, n: int, data_array: Array[float], perm_array: Array[int], 
                              threads_per_block: int = 256):
    """
    NVIDIA CUDA lancher for GPU compute kernel
    
    Parameters
    ----------
    k : uint16
        Number of groups
    n : uint16
        Number of observations per group
    data_array : float64[:,:]
        Simulation data of shape (num_simulations, k*n)
    perm_array : uint16[:,:]
        Permutation array of shape (num_perms, k*n)
    threads_per_block : int16
        Tuning parameter for optimizing GPU occupancy

    Returns
    -------
    d_pval : float64[:,:]
        p-value array of shape(num_simulation, 2)
    """
    num_simulations = data_array.shape[0]

    d_data  = cuda.to_device(np.ascontiguousarray(data_array, dtype=np.float64))
    d_perms = cuda.to_device(np.ascontiguousarray(perm_array, dtype=np.uint16))
    d_pval  = cuda.device_array((num_simulations, 2), dtype=np.float64)

    blocks = math.ceil(num_simulations / threads_per_block)
    compute_kernel_gpu[blocks, threads_per_block]( # type: ignore[reportIndexIssue]
        np.uint16(k), np.uint16(n), d_data, d_perms, d_pval
    )

    return d_pval.copy_to_host()