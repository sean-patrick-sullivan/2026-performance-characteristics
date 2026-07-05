"""
Generate simulated sample data

This module contains JIT-compiled functions for generating simulated sample
data drawn from supplied distributions with the application of a location-shift
treatment. The data generating function is an application of expression (6) 
with the appropriate stype of shift from expression (7).
"""

from typing import Callable

import numpy as np
from numba import njit, prange, uint16, uint64, float64, types

from monte_carlo.utils import hash_seed


# --- Treatment application functions ---

@njit(float64[:](uint16, uint16, float64[:], float64))
def apply_treatment_sym(k: int, n: int, data: np.ndarray, shift: float):
    """
    Apply symmetric location shift
    
    Parameters
    ----------
    k : int
        Number of groups
    n : int
        Number of observations per group
    data : array(float)
        Flat array of n*k data values to treat
    shift : float
        Size of shift to be applied

    Returns
    -------
    data : array(float)
        Input data after symmetric shift by `shift` has been applied

    Notes
    -----
    As described in Part 3, the symmetric treatment effect is (1) not applied to 
    the first group $j=1$, and (2) applied as $x + (j - 1) * shift$ for all groups
    $j>1$.
    """
    kn = k * n
    for idx in range(kn):
        group = idx // n
        data[idx] += shift * group
    return data

@njit(float64[:](uint16, uint16, float64[:], float64))
def apply_treatment_asy(k: int, n: int, data: np.ndarray, shift: float):
    """
    Apply asymmetric location shift

    Parameters
    ----------
    k : int
        Number of groups
    n : int
        Number of observations per group
    data : array(float)
        Flat array of n*k data values to treat
    shift : float
        Size of shift to be applied

    Returns
    -------
    data : array(float)
        Input data after asymmetric shift by `shift` has been applied

    Notes
    -----
    As described in Part 4.3, the asymmetric treatment effect is (1) not applied to 
    the first group $j=1$, and (2) applied as $x + shift$ for all groups $j>1$.
    """
    kn = k * n
    for idx in range(kn):
        group = idx // n
        if group > 0:
            data[idx] += shift
    return data


# --- Producing treated data from fixed distributions ---

# Numba type signature for `gen_data_` functions
gen_data_fun_type = types.FunctionType(float64[:](uint16, uint16, uint64))

def gen_data_array_factory(apply_treatment_fun):
    """
    Function factory for producing data generation functions for fixed
    distributions

    Parameters
    ----------
    apply_treatment_fun : function
        Treatment type to apply to simulated data

    Returns
    -------
    gen_data_array : function
        @njit optimized data generation function
    """
    @njit(float64[:,:](uint16, uint16, uint64, float64,
                    gen_data_fun_type, uint64), parallel=True)
    def gen_data_array(k: int, n: int, num_simulations: int, treatment_shift: float, 
                       generate_data_fun: Callable, pass_seed: int):
        """
        Produce random data arrays of shape (num_simulations, k * n) with base data drawn
        from the supplied distribution and then treated with named treatment function and
        supplied shift value.
        
        Parameters
        ----------
        k : int
            Number of groups
        n : int
            Number of observations per group
        num_simulations : int
            Number of simulated data sets to create
        treatment_shift : float
            Size of treatment shift to apply
        generate_data_fun : function
            Specified data generation function, `gen_data_F[1-8]`
        pass_seed : int
            Randomization base to use in per-simulation seed hashing
        
        Returns
        -------
        data : array(float)
            Simulated dataset array of shape (num_simulations, k * n)
        """
        
        result = np.empty((num_simulations, k * n), dtype=np.float64)
        for idx in prange(num_simulations):
            sim_seed = hash_seed(pass_seed, uint64(idx))
            untreated_data = generate_data_fun(k, n, sim_seed)
            result[idx] = apply_treatment_fun(k, n, untreated_data, treatment_shift)
        
        return result
    return gen_data_array

gen_data_array_sym = gen_data_array_factory(apply_treatment_sym)
gen_data_array_asy = gen_data_array_factory(apply_treatment_asy)


# --- Producing treated data from variable distributions ---

# Numba type signature for `gen_data_var` functions
gen_data_var_fun_type = types.FunctionType(
    float64[:](uint16, uint16, float64, uint64)
    )

def gen_data_var_array_factory(apply_treatment_fun):
    """
    Function factory for producing data generation functions for variable
    distributions

    Parameters
    ----------  
    apply_treatment_fun : function
        Treatment type to apply to simulated data

    Returns
    -------
    gen_data_array : function
        @njit optimized data generation function
    """
    @njit(float64[:,:](uint16, uint16, uint64, float64,
                    gen_data_var_fun_type, float64, uint64), parallel=True)
    def gen_data_var_array(k: int, n: int, num_simulations: int, 
                           treatment_shift: float, 
                           generate_data_var_fun: Callable, 
                           parameter: float, pass_seed: int):
        """
        Produce random data arrays of shape (num_simulations, k * n) with base 
        data drawn from the supplied variable distribution with supplied 
        parameter value, treated with named treatment function and
        supplied shift value.
        
        Parameters
        ----------
        k : int16
            Number of groups
        n : int16
            Number of observations per group
        num_simulations : int64
            Number of simulated data sets to create
        treatment_shift : float64
            Size of treatment shift to apply
        generate_data_var_fun : function
            Specified data generation function, `gen_data_var_F[1, 3, 5, 7]`
        parameter : float64
            Parameter value to be passed to data generation function
        pass_seed : int
            Randomization base to use in per-simulation seed hashing
        
        Returns
        -------
        data : array(float)
            Simulated dataset array of shape (num_simulations, k * n)
        """
        
        result = np.empty((num_simulations, k * n), dtype=np.float64)
        for idx in prange(num_simulations):
            sim_seed = hash_seed(pass_seed, uint64(idx))
            untreated_data = generate_data_var_fun(k, n, parameter, sim_seed)
            result[idx] = apply_treatment_fun(k, n, untreated_data, treatment_shift)
        
        return result
    return gen_data_var_array

gen_data_var_array_sym = gen_data_var_array_factory(apply_treatment_sym)
gen_data_var_array_asy = gen_data_var_array_factory(apply_treatment_asy)