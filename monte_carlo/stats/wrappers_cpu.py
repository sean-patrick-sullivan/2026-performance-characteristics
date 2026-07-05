"""
Wrapper functions for performing one-off permutation tests on CPU

This module provides functions for performing one-off permutation tests
using CPU computations.
"""

from numpy.typing import NDArray as Array
from numba import njit, uint16, uint32, float64

from monte_carlo.stats.stats_cpu import (
    compute_J_for_perm_cpu, 
    compute_D_for_perm_cpu,
    compute_stats_for_perm_cpu,
)

# --- CPU permutation test core funcitons ---

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
"""

@njit(uint32(uint16, uint16, uint32, float64[:], uint16[:, :]))
def core_permutation_test_J_cpu(k: int, n: int, num_perms: int, 
                                data: Array[float], perm_array: Array[int]):
    """
    Core permutation test computation for JT test

    Returns
    -------
    count : uint32
        Number of as-or-more-extreme J statistic values in the num_perms
        possible alternative permutations of the sample data.
    """
    observed = compute_J_for_perm_cpu(k, n, data, perm_array[0])
    count = 0
    for i in range(num_perms):
        stat = compute_J_for_perm_cpu(k, n, data, perm_array[i])
        if stat >= observed:
            count += 1
    return count

@njit(uint32(uint16, uint16, uint32, float64[:], uint16[:, :]))
def core_permutation_test_D_cpu(k: int, n: int, num_perms: int, 
                                data: Array[float], perm_array: Array[int]):
    """
    Core permutation test computation for JT test

    Returns
    -------
    count : uint32
        Number of as-or-more-extreme D statistic values in the num_perms
        possible alternative permutations of the sample data.
    """
    observed = compute_D_for_perm_cpu(k, n, data, perm_array[0])
    count = 0
    for i in range(num_perms):
        stat = compute_D_for_perm_cpu(k, n, data, perm_array[i])
        if stat >= observed:
            count += 1
    return count

@njit((uint16, uint16, uint32, float64[:], uint16[:, :]))
def core_permutation_test_stats_cpu(k: int, n: int, num_perms: int, 
                                data: Array[float], perm_array: Array[int]):
    """
    Core permutation test computation for JT and DD tests

    Returns
    -------
    count, count : uint32, uint32
        Number of as-or-more-extreme J statistic values and D statistic values
        in the num_perms possible alternative permutations of the sample data.
    """
    obs_J = compute_J_for_perm_cpu(k, n, data, perm_array[0])
    obs_D = compute_D_for_perm_cpu(k, n, data, perm_array[0])
    count_J = 0
    count_D = 0
    for i in range(num_perms):
        stat_J = compute_J_for_perm_cpu(k, n, data, perm_array[i])
        stat_D = compute_D_for_perm_cpu(k, n, data, perm_array[i])
        if stat_J >= obs_J:
            count_J += 1
        if stat_D >= obs_D:
            count_D += 1
    return count_J, count_D


# --- CPU permutation test wrapper functions ---

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
"""

def permutation_test_J_cpu(k: int, n: int, data: Array[float], 
                           perm_array: Array[int]):
    """
    Wrapper for one-off performance of JT test.

    Returns
    -------
    pval, observed, count : float64, uint16, uint32
        p-value, observed value of J statistic, and count of as-or-more-
        extreme J values in the empirical null distribution
    """
    num_perms = perm_array.shape[0]
    
    # Compute observed value, count greater, and p_value
    observed = compute_J_for_perm_cpu(k, n, data, perm_array[0])
    count = core_permutation_test_J_cpu(k, n, num_perms, data, perm_array)
    pval = count / num_perms

    return pval, observed, count

def permutation_test_D_cpu(k: int, n: int, data: Array[float], 
                           perm_array: Array[int]):
    """
    Wrapper for one-off performance of DD test.

    Returns
    -------
    pval, observed, count : float64, float64, uint32
        p-value, observed value of D statistic, and count of as-or-more-
        extreme D values in the empirical null distribution
    """
    num_perms = perm_array.shape[0]
    
    # Compute observed value, count greater, and p_value
    observed = compute_D_for_perm_cpu(k, n, data, perm_array[0])
    count = core_permutation_test_D_cpu(k, n, num_perms, data, perm_array)
    pval = count / num_perms

    return pval, observed, count

def permutation_test_stats_cpu(k: int, n: int, data: Array[float], 
                           perm_array: Array[int]):
    """
    Wrapper for one-off performance of both JT and DD tests

    Returns
    -------
    (pval_J, pval_D), (observed_J, observed_D), (count_J, count_D)
        Tupples of pvalues, observed test statistics, and as-or-more-extreme counts
        for the JT test and DD test, repectively
    """
    num_perms = perm_array.shape[0]
    
    # Compute observed value, count greater, and p_value
    obs_J, obs_D = compute_stats_for_perm_cpu(k, n, data, perm_array[0])
    count_J, count_D = core_permutation_test_stats_cpu(k, n, num_perms, data, perm_array)
    pval_J, pval_D = count_J / num_perms, count_D / num_perms

    return (pval_J, pval_D), (obs_J, obs_D), (count_J, count_D)