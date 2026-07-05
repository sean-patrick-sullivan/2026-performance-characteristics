"""
Random number generation from probability distributions var_F1 - var_F8

This module contains JIT-compiled implementations of the variable-parameter
distributions used in the secondary Monte Carlo study. These align with 
the definitions described in Part 5.3 of the manuscript.
"""

from math import sqrt, pi
from typing import Callable

import numpy as np
from numba import njit, uint16, uint64, float64

# --- Data generation for distributions F1 through F8 ---

"""
Common function signature for var_F1 through var_F8

Parameters
----------
size : uint64
    Number of random draws to simulate
parameter : float64
    Parameter value passed to underlying distribution.
seed : uint64
    Seed passed to np.random

Returns
-------
result : float64[:]
    Array of `size` random draws from the named distribution.
"""

@njit(float64[:](uint64, float64, uint64))
def var_F1(size: int, var: float, seed: int):
    r"""
    Generate random draws from normal distribution with mean zero and supplied
    variance.

    Notes
    -----
    Normally distributed random variable, $N(0, var)$.
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(var))
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F2(size: int, var: float, seed: int):
    r"""
    Generate random draws from uniform distribution with mean zero and 
    supplied variance.

    Notes
    -----
    Uniform random variable, symmetric around zero, with support width 
    calculated to produce supplied variance.)
    """
    width = sqrt(12*var)
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.uniform(-width/2, width/2)
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F3(size: int, var: float, seed: int):
    r"""
    Generate random draws from a mixture of normal distributions such that the
    random variable has variance `var`.

    Notes
    -----
    A mixture model composed of an 80% chance of a draw from $N(0,5)$ and a 20% 
    chance of a draw from $N(0,outliner_var). The value of `outlier_var` is 
    computed to ensure that the overall mixture variable has specified 
    variance. Permissible variance argument values are constrained by 
    $var \ge 4$.
    """
    np.random.seed(seed)
    primaryVar = np.empty(size, dtype=np.float64)
    outlierVar = np.empty(size, dtype=np.float64)
    binomialChoice = np.empty(size, dtype=np.int8)
    outlier_var = 5*(var-4)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        primaryVar[i] = np.random.normal(0, sqrt(5))
        outlierVar[i] = np.random.normal(0, sqrt(outlier_var))
        binomialChoice[i] = np.random.binomial(1, 0.8)
        result[i] = (primaryVar[i] * binomialChoice[i]) + (outlierVar[i] * (1 -  binomialChoice[i]))
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F4(size: int, var: float, seed: int):
    r"""
    Generate random draws from a convolution of a normal distribution and a 
    uniform distribution such that the random variable has variance `var`.
    
    Notes
    -----
    Convolution model random variable constructed by adding $N(0,x_var)$ to an 
    independent uniform random variable with support $[-y_width/2,y_width/2]$. 
    The values of `x_var` and `y_width` are computed to ensure that the 
    resulting random variable has variance `var` while preserving a 1:8 ratio 
    between the variances of the normal and uniform parts.
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    x_var = var/9
    y_width = sqrt(96./9*var)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(x_var)) + np.random.uniform(-y_width/2, y_width/2) 
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F5(size: int, var: float, seed: int):
    r"""
    Generate random draws from a logistic distribution with scale parameter
    calculated to yield a variance of `var`.
    """
    np.random.seed(seed)
    scale = sqrt(3*var)/pi
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.logistic(0, scale)
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F6(size: int, mad: float64, seed: int):
    r"""
    Generate random draws from Cauchy distribution with scale paraemter set
    to yeild a mediam absolute deviation of `mad`.

    Notes
    -----
    The value of `mad` is constrainted by $mad > 0$
    """
    np.random.seed(seed)
    scale = mad
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.standard_cauchy() * scale
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F7(size: int, var: float, seed: int):
    r"""
    Generate Gumbel distributed random variable with location and scale 
    parameters computed to set expectation to zero and variance to `var`.
    """
    np.random.seed(seed)
    beta = sqrt(var*6)/pi
    mu = -beta * np.euler_gamma # euler_gamma \approx 0.5772
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.gumbel(mu,beta)
    return result

@njit(float64[:](uint64, float64, uint64))
def var_F8(size: int, var: float, seed: int):
    r"""
    Generate exponential distributed random variable with scale and locational 
    shift calculated to result in a random variable with expectation zero and 
    specified variance, `var`.

    Notes
    -----
    Exponential distributed random variable shifted back by expectation to 
    achieve mean zero location.
    """
    np.random.seed(seed)
    scale = sqrt(var)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.exponential(scale) - scale
    return result

# --- Data generation functions for var_F1 through var_F8 ---

def gen_data_var_factory(dist_function: Callable):
    """
    Function factory for producing data generation functions

    Parameters
    ----------
    dist_function : function
        Distribution F1 through F8 to use when drawing simulated data.

    Returns
    -------
    generate_data : function
        @njit optimized data generation function
    """
    @njit(float64[:](uint16, uint16, float64, uint64))
    def generate_data_var(k: int, n: int, parameter: float, seed: int):
        """
        Generate flat array of random data from named distribution.

        Parameters
        ----------
        k : uint16
            Number of treatment groups (data_array rows)
        n : uint16
            Number of observations per treatment group (data_array columns)
        parameter : float64
            Parameter to pass to named distribution
        seed : uint64
            Seed passed to np.random before data generation

        Returns
        -------
        result : float64[:]
            contiguous numpy array of length (k*n)
        """
        result = np.ascontiguousarray(dist_function(k * n, parameter, seed))
        return result
    return generate_data_var

gen_data_var_F1 = gen_data_var_factory(var_F1)
gen_data_var_F2 = gen_data_var_factory(var_F2)
gen_data_var_F3 = gen_data_var_factory(var_F3)
gen_data_var_F4 = gen_data_var_factory(var_F4)
gen_data_var_F5 = gen_data_var_factory(var_F5)
gen_data_var_F6 = gen_data_var_factory(var_F6)
gen_data_var_F7 = gen_data_var_factory(var_F7)
gen_data_var_F8 = gen_data_var_factory(var_F8)