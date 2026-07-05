"""
Orchestration logic for conducting secondary Monte Carlo studies

This module orchestes the steps in conduct the secondary Monte Carlo study 
described and reported in Part 5.3.
"""

# Standard libraries
from itertools import product
from typing import Callable

# Third-party libraries
import numpy as np
import pandas as pd # type: ignore

# Local modules
import monte_carlo.dists.variable_dists as dists
from monte_carlo.engine import (
    gen_data_var_array_sym, 
    gen_data_var_array_asy,
    compute_kernel_cpu, 
    launch_compute_kernel_gpu,
)
from monte_carlo.stats import gen_perm_array
from monte_carlo.utils import hash_secondary_profile


# --- Module-level registries ---

DIST_MAP: dict[str, Callable] = {
    "F1": dists.gen_data_var_F1,
    "F2": dists.gen_data_var_F2,
    "F3": dists.gen_data_var_F3,
    "F4": dists.gen_data_var_F4,
    "F5": dists.gen_data_var_F5,
    "F6": dists.gen_data_var_F6,
    "F7": dists.gen_data_var_F7,
    "F8": dists.gen_data_var_F8,
}

TREATMENT_MAP: dict[str, Callable] = {
    "symmetric": gen_data_var_array_sym,
    "asymmetric": gen_data_var_array_asy
}

TEST_NAMES = ("JT_test", "DD_test")

ALPHAS = ([0.05])


# --- Orchestration layer for secondary study ---

def run_secondary_study(
    k_array: list[int],
    n_array: list[int],
    treatment_shift_dict: dict[int, dict[int, dict[str, float]]],
    num_vars: int,
    var_bounds_dict: dict[int, dict[int, dict[str, tuple[float, float]]]],
    num_simulations: int,
    *,
    dist_map: dict[str, Callable] | None = None, # type: ignore
    treatment: str = "symmetric",
    base_seed: int = 138492,
    compute_arch: str = 'cpu',
    threads_per_block: int = 256
):
    """
    Orchestration of secondary Monte Carlo study design.
    
    Parameters
    ----------
    k_array : list[uint16]
        Number of groups
    n_array : list[uint16]
        Number of observations per group
    treatment_shift_map : dict[int, dict[int, dict[str, float]]]
        Treatment shift to apply to 
    num_parameters : uint16
        Number of parameter values to test between the lower and upper bounds
        for the given distribution and parameter
    var_bounds_dict: dict[int, dict[int, dict[str, tuple[float, float]]]]
        Specified parameter value bounds per combination of (k, n, dist_name),
        callable as `parameter_bounds_dict[k][n][dist_name]`
    num_simulations : uint64
        Number of simulations to run per combination of (k, n, dist_name, shift)
    dist_map : dict[str, Callable]
        Mapping of dist_name to gen_data_[F1-F8] functions
    treatment : str
        Specification of "symmetric" or "asymmetric" treatment effects
    base_seed : uint64
        Base seed to use for this Monte Carlo study; all subsequent random number
        generation is deterministic for the supplied base_seed
    compute_arch : str
        Specification of "cpu" or "gpu" archicture for performing permutation test
        computations in the hot loop of this study
    threads_per_block : int
        Tuning parameter for optimizing GPU occupancy

    Returns
    -------
    results_accumulator : pd.DataFrame
        Long-format simulation data with the following columns:
            k : int16,
            n : int16,
            dist_name : str,
            treatment_shift : float64, 
            parameter : float64,
            test_name : str,
            alpha : float64,
            rejection_rate : float64

    Notes
    -----
    Running on GPU requires compatible hardware and drivers. 
    """
    
    # Collect constants and defaults
    if dist_map is None:
        dist_map = DIST_MAP
    gen_data_var_array = TREATMENT_MAP[treatment]
    
    # Pre-generate permutations once per (k, n)
    perm_array_cache = {
        (k, n): gen_perm_array(k, n) 
        for k, n in product(k_array, n_array)
    }

    results_accumulator = []

    # Iterate through the simulation profile grid
    for k, n, (dist_idx, dist_name) in product(k_array, n_array, enumerate(dist_map.keys())):
        
        # Step 0. Collect and generate profile parameters
        print(f"Running profile (k={k}, n={n}, dist={dist_name})")
        gen_data_var_fun = dist_map[dist_name]
        treatment_shift = treatment_shift_dict[k][n][dist_name]
        min_parameter, max_parameter = var_bounds_dict[k][n][dist_name]
        parameters = np.linspace(min_parameter, max_parameter,
                                       num=num_vars)
        
        # Step 1. Batch data generation
        # Stacked 2D array of shape (num_simulations * num_treatments, k*n)
        num_stacked_rows = num_vars * num_simulations
        stacked_data_array = np.empty((num_stacked_rows, k * n), dtype=np.float64)
        
        for parameter_idx, parameter in enumerate(parameters):
            start_row = parameter_idx * num_simulations
            end_row = start_row + num_simulations
            
            # Fill a slice of stacked_data_array
            profile_seed = hash_secondary_profile(base_seed, k, n, dist_idx, parameter_idx)
            stacked_data_array[start_row:end_row] = gen_data_var_array(
                k, n, num_simulations, treatment_shift, gen_data_var_fun, parameter, profile_seed
            )

        # Step 2. Launch compute kernel
        if compute_arch == 'gpu':
            stacked_pvals = launch_compute_kernel_gpu(k, n, stacked_data_array, perm_array_cache[(k, n)],
                                                      threads_per_block)
        else:
            stacked_pvals = compute_kernel_cpu(k, n, stacked_data_array, perm_array_cache[(k, n)])

        # Step 3. Disaggregate stacked results and compute averages
        # Add num_treatments * 2 (tests) * 3 (levels) rows to results_accumulator
        for parameter_idx, parameter in enumerate(parameters):
            
            # Extract the (num_simulations, 2) p-values specific to this treatment parameter
            start_row = parameter_idx * num_simulations
            end_row = start_row + num_simulations
            pvals_slice = stacked_pvals[start_row:end_row]
            
            for alpha in ALPHAS:
                # Count how often p < alpha across all simulations in this slice
                rejection_rates = (pvals_slice < alpha).mean(axis=0)
                
                for test_idx, test_name in enumerate(TEST_NAMES):
                    results_accumulator.append({
                        'k': k,
                        'n': n,
                        'dist': dist_name,
                        'treatment_shift': treatment_shift,
                        'parameter' : parameter,
                        'test_name': test_name,
                        'alpha': alpha,
                        'rejection_rate': rejection_rates[test_idx]
                    })

    # Return results
    return pd.DataFrame(results_accumulator)