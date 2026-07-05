"""
Monte Carlo computation of p-values using CPU architeture

This module implements a JIT-compiled and parallel computing of 
p-values for the JT and DD test in the hot loops of Monte Carlo studies.
"""

import numpy as np
from numpy.typing import NDArray as Array
from numba import njit, prange, uint16, float64

from monte_carlo.stats import compute_stats_for_perm_cpu

# --- CPU compute kernel ---

@njit(float64[:,:](uint16, uint16, float64[:,:], uint16[:,:]), parallel=True)
def compute_kernel_cpu(k: int, n: int, data_array: Array[float], perm_array: Array[int]):
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
        
    Returns
    -------
    pval_array : float64[:,:]
        p-value array of shape (num_simulations,2)
    
    Notes
    -----
    This implementation assumes that perm_array[0] reconstitutes the original order of the
    supplied sample data.
    """
    num_simulations = data_array.shape[0]
    num_perms = perm_array.shape[0]

    pval_array = np.empty((num_simulations,2), dtype=np.float64)
    for idx_sim in prange(num_simulations):
        J_obs, D_obs = compute_stats_for_perm_cpu(k, n, data_array[idx_sim], perm_array[0])
        count_J = 0
        count_D = 0
        for idx_perm in range(num_perms):
            J, D = compute_stats_for_perm_cpu(k, n, data_array[idx_sim], perm_array[idx_perm])
            if J >= J_obs:
                count_J += 1
            if D >= D_obs:
                count_D += 1
        pval_array[idx_sim, 0] = count_J / num_perms
        pval_array[idx_sim, 1] = count_D / num_perms
    
    return pval_array