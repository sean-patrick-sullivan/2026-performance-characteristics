"""
Wrapper functions for performing one-off permutation tests on CPU

This module provides functions for performing one-off permutation tests
using CPU computations.
"""

import math

import numpy as np
from numpy.typing import NDArray as Array
from numba import cuda

from monte_carlo.stats.stats_gpu import (
    compute_J_for_perm_gpu, 
    compute_D_for_perm_gpu,
    compute_stats_for_perm_gpu,
)

# --- GPU permutation test core funcitons ---

"""
Common function signature for core fuctions

Parameters
----------
k : uint16
    Number of groups
n : uint16
    Number of observations per group
num_perms : uint32
    Number of permutations to evaluate
data : float64[:]
    Simulation data array of length (k*n)
perm_arry : uint16[:,:]
    Permutation array of length (num_perms, k*n)

Returns
-------
None
"""

@cuda.jit
def core_permutation_test_J_gpu(k: int, n: int, num_perms: int, data: Array[float], 
                                perm_array: Array[int], out_J: int):
    """
    Core permutation test computation for JT test
    
    Parameters
    ----------
    out_J : uint16[:]
        [Output] pre-allocated array of length (num_perms) 
        where J static values are written in-place
    """
    i = cuda.grid(1)
    if i >= num_perms:
        return
    stat = compute_J_for_perm_gpu(k, n, data, perm_array[i])
    out_J[i] = stat # type: ignore

@cuda.jit
def core_permutation_test_D_gpu(k, n, num_perms, data, perm_array, out_D):
    """
    Core permutation test computation for DD test
    
    Parameters
    ----------
    out_D : float64[:]
        [Output] pre-allocated array of length (num_perms) 
        where D static values are written in-place
    """
    i = cuda.grid(1)
    if i >= num_perms:
        return
    stat = compute_D_for_perm_gpu(k, n, data, perm_array[i])
    out_D[i] = stat

@cuda.jit
def core_permutation_test_stats_gpu(k, n, num_perms, data, perm_array, out_J, out_D):
    """
    Core permutation test computation for JT and DD tests
    
    Parameters
    ----------
    out_J, out_D : uint16[:], float64[:]
        [Output] pre-allocated arrays of length (num_perms) 
        where J and D static values are written in-place
    """
    i = cuda.grid(1)
    if i >= num_perms:
        return
    J, D = compute_stats_for_perm_gpu(k, n, data, perm_array[i])
    out_J[i] = J
    out_D[i] = D


# --- GPU permutation test wrapper functions ---

"""
Common function signature for permutation test functions

Parameters
----------
k : uint16
    Number of groups
n : uint16
    Number of observations per group
data : float64[:]
    Simulation data of array of length (k*n)
perm_arry : uint16[:,:]
    Permutation array of length (num_perms, k*n)
threads_per_block : uint16
    Tuning parameter for optimizing GPU occupancy
"""

def permutation_test_J_gpu(k: int, n: int, num_perms: int, data: Array[float], 
                           perm_array: Array[int], threads_per_block=256):
    """
    Wrapper for one-off performance of JT test.

    Returns
    -------
    pval, observed, count : float64, uint16, uint32
        p-value, observed value of J statistic, and count of as-or-more-
        extreme J values in the empirical null distribution
    """
    # Transfer data to device
    d_data = cuda.to_device(np.ascontiguousarray(data, dtype=np.float64))
    d_perms = cuda.to_device(np.ascontiguousarray(perm_array, dtype=np.uint16))

    # Output buffer: one uint16 J-value per permutation
    d_out = cuda.device_array(num_perms, dtype=np.uint16)

    # Launch: one thread per permutation
    blocks = math.ceil(num_perms / threads_per_block)
    core_permutation_test_J_gpu[blocks, threads_per_block]( # type: ignore
        np.uint16(k), np.uint16(n), np.uint32(num_perms), d_data, d_perms, d_out
    )

    # Copy results back and reduce on CPU
    out = d_out.copy_to_host()
    observed = int(out[0])
    count = int(np.sum(out >= observed))
    pval = count / num_perms

    return pval, observed, count

def permutation_test_D_gpu(k: int, n: int, num_perms: int, data: Array[float], 
                           perm_array: Array[int], threads_per_block=256):
    """
    Wrapper for one-off performance of JT test.

    Returns
    -------
    pval, observed, count : float64, float64, uint32
        p-value, observed value of D statistic, and count of as-or-more-
        extreme D values in the empirical null distribution
    """
    # Transfer data to device
    d_data = cuda.to_device(np.ascontiguousarray(data, dtype=np.float64))
    d_perms = cuda.to_device(np.ascontiguousarray(perm_array, dtype=np.uint16))

    # Output buffer: one float64 D-value per permutation
    d_out = cuda.device_array(num_perms, dtype=np.float64)

    # Launch: one thread per permutation
    blocks = math.ceil(num_perms / threads_per_block)
    core_permutation_test_D_gpu[blocks, threads_per_block]( # type: ignore
        np.uint16(k), np.uint16(n), np.uint32(num_perms), d_data, d_perms, d_out
    )

    # Copy results back and reduce on CPU
    out = d_out.copy_to_host()
    observed = float(out[0])
    count = int(np.sum(out >= observed))
    pval = count / num_perms

    return pval, observed, count

def permutation_test_stats_gpu(k: int, n: int, num_perms: int, data: Array[float], 
                               perm_array: Array[int], threads_per_block=256):
    """
    Wrapper for one-off performance of both JT and DD tests

    Returns
    -------
    (pval_J, pval_D), (observed_J, observed_D), (count_J, count_D)
        Tupples of pvalues, observed test statistics, and as-or-more-extreme counts
        for the JT test and DD test, repectively
    """
    # Transfer data to device
    d_data  = cuda.to_device(np.ascontiguousarray(data, dtype=np.float64))
    d_perms = cuda.to_device(np.ascontiguousarray(perm_array, dtype=np.uint16))

    # Separate output buffers for J and D
    d_out_J = cuda.device_array(num_perms, dtype=np.float64)
    d_out_D = cuda.device_array(num_perms, dtype=np.float64)

    # Launch: one thread per permutation
    blocks = math.ceil(num_perms / threads_per_block)
    core_permutation_test_stats_gpu[blocks, threads_per_block]( # type: ignore
        np.uint16(k), np.uint16(n), np.uint32(num_perms), d_data, d_perms, d_out_J, d_out_D
    )

    # Copy results back and reduce on CPU
    out_J = d_out_J.copy_to_host()
    out_D = d_out_D.copy_to_host()

    observed_J = float(out_J[0])
    observed_D = float(out_D[0])

    count_J  = int(np.sum(out_J >= observed_J))
    count_D  = int(np.sum(out_D >= observed_D))
    pval_J = count_J / num_perms
    pval_D = count_D / num_perms

    return (pval_J, pval_D), (observed_J, observed_D), (count_J, count_D)