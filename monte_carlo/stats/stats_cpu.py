"""
Compute test statistic values using CPU architeture

This module contains functions for computing test statistic values for a 
supplied sample and permutation index array. These functions are exposed for 
use in one-off testing via the wrappers_cpu module and are directly used by 
Monte Carlo orchestration modules through compute_kernel_cpu in the 
engine.backend_cpu module.
"""

from numpy.typing import NDArray as Array
from numba import njit, uint16, float64


# --- CPU computation of test statistic values ---

"""
Common function signature for compute_[...]_for_perm_cpu

Parameters
----------
k : uint16
    Number of groups
n : uint16
    Number of observations per group
data : float64[:]
    Simulation data of array of length (k*n)
perm : uint16[:]
    Permutation array of length (k*n)
"""

@njit(uint16(uint16, uint16, float64[:], uint16[:]))
def compute_J_for_perm_cpu(k: int, n: int, data: Array[float], perm: Array[int]):
    """
    Compute J statistic for supplied data and permutation

    Returns
    -------
    J : uint16
        J statistic value as defined in equation (2)
    """
    J = 0
    kn = k * n
    for idx1 in range(kn):
        g1 = idx1 // n
        v1 = data[perm[idx1]]
        for idx2 in range(kn):
            g2 = idx2 // n
            if g2 > g1:
                v2 = data[perm[idx2]]
                if v2 > v1:
                    J += 1
    return J

@njit(float64(uint16, uint16, float64[:], uint16[:]))
def compute_D_for_perm_cpu(k: int, n: int, data: Array[float], perm: Array[int]):
    """
    Compute D statistic for supplied data and permutation

    Returns
    -------
    D : float64
        D statistic value as defined in equation (5)
    """
    D = 0.0
    kn = k * n
    for idx1 in range(kn):
        g1 = idx1 // n
        v1 = data[perm[idx1]]
        for idx2 in range(kn):
            g2 = idx2 // n
            if g2 > g1:
                v2 = data[perm[idx2]]
                D += (v2 - v1)
    return D

@njit((uint16, uint16, float64[:], uint16[:]))
def compute_stats_for_perm_cpu(k: int, n: int, data: Array[float], perm: Array[int]):
    """
    Compute both J and D statistic for supplied data and permutation

    Returns
    -------
    J, D : uint16, float64
        J and D statistic value as defined in equations (2) and (5)
    """
    J = 0
    D = 0.0
    kn = k * n
    for idx1 in range(kn):
        g1 = idx1 // n
        v1 = data[perm[idx1]]
        for idx2 in range(kn):
            g2 = idx2 // n
            if g2 > g1:
                v2 = data[perm[idx2]]
                D += (v2 - v1)
                if v2 > v1:
                    J += 1
    return J, D