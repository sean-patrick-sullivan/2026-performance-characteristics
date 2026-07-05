"""
Random number generation from probability distributions F1 - F8

This module contains JIT-compiled implementations of the fixed-parameter
distributions used in the principal Monte Carlo study. These align with 
the definitions in Part 4 of the manuscript.
"""

from math import sqrt, pi
from typing import Callable

import numpy as np
from numba import njit, uint16, uint64, float64

# --- Data generation for distributions F1 through F8 ---

"""
Common function signature for F1 through F8

Parameters
----------
size : uint64
    Number of random draws to simulate
seed : uint64
    Seed passed to np.random

Returns
-------
result : float64[:]
    Array of `size` random draws from the named distribution.
"""

@njit(float64[:](uint64, uint64))
def F1(size: int, seed: int):
    r"""
    Generate random draws from F1.

    Notes
    -----
    Normally distributed random variable, $N(0, 50)$. 
    (Expectation 0 and variance 50.)
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(50))
    return result

@njit(float64[:](uint64, uint64))
def F2(size: int, seed: int):
    r"""
    Generate random draws from F2.

    Notes
    -----
    Uniformly distributed random variable on support 
    $[-10\sqrt(6),10\sqrt(6)]$. (Expectation 0 and variance 200.)
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.uniform(-10*sqrt(6), 10*sqrt(6))
    return result

@njit(float64[:](uint64, uint64))
def F3(size: int, seed: int):
    r"""
    Generate random draws from F3.

    Notes
    -----
    A mixture model composed of an 80% chance of a draw from $N(0,5)$ and a 20% 
    chance of a draw from $N(0,25). This can be thought of as a $N(0,5)$ 
    variable with occasional outliers. (Expectation 0 and variance $(0.80 * 5) 
    + (.20 * 25) = 9$.)
    """
    np.random.seed(seed)
    primaryVar = np.empty(size, dtype=np.float64)
    outlierVar = np.empty(size, dtype=np.float64)
    binomialChoice = np.empty(size, dtype=np.int8)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        primaryVar[i] = np.random.normal(0, sqrt(5))
        outlierVar[i] = np.random.normal(0, sqrt(25))
        binomialChoice[i] = np.random.binomial(1, 0.8)
        result[i] = (primaryVar[i] * binomialChoice[i]) + (outlierVar[i] * (1 -  binomialChoice[i]))
    return result

@njit(float64[:](uint64, uint64))
def F4(size: int, seed: int):
    r"""
    Generate random draws from F4.

    Notes
    -----
    Convolution model random variable constructed by adding $N(0,25)$ to an independent uniform 
    random variable with support $[-10\sqrt(6),10\sqrt(6)]$. (Expectation 0 and variance 25 + 200 = 225).
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(25)) + np.random.uniform(-10*sqrt(6), 10*sqrt(6)) 
    return result

@njit(float64[:](uint64, uint64))
def F5(size: int, seed: int):
    r"""
    Generate random draws from F5.

    Notes
    -----
    Logistic random variable with location 0 and scale 5\sqrt(6)/pi. (Expectation 0 and variance 50.)
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.logistic(0, 5*sqrt(6)/pi)
    return result

@njit(float64[:](uint64, uint64))
def F6(size: int, seed: int):
    r"""
    Generate random draws from F6.

    Notes
    -----
    Cauchy distributed random variable with location parameter 0 and scale parameter 1. (Expectation 
    undefined and variance undefined; median 0 and median absolute deviation 1.)
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.standard_cauchy() # multiply by x to set scale x
    return result

@njit(float64[:](uint64, uint64))
def F7(size: int, seed: int):
    r"""
    Generate random draws from F7.

    Notes
    -----
    Gumbel distributed random variable with location parameter $\approx$ -23 and 
    scale parameter $10\sqrt(3)/\pi$. This choice of scale parameter shifts the 
    distribution back to an expectation of zero. (Expectation 0 and variance 50.)
    """
    np.random.seed(seed)
    beta = 10*sqrt(3)/pi
    mu = -beta * np.euler_gamma # euler_gamma \approx 0.5772
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.gumbel(mu,beta)
    return result

@njit(float64[:](uint64, uint64))
def F8(size: int, seed: int):
    r"""
    Generate random draws from F8.

    Notes
    -----
    Exponential distributed random variable with scale 10/sqrt(2), shifted back 
    by $10/\sqrt(2)$ to achieve mean zero location. (Expectation 0 and variance 50.)
    """
    np.random.seed(seed)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.exponential(10.0/sqrt(2)) - 10.0/sqrt(2)
    return result


# --- Data generation functions for F1 through F8---

def gen_data_factory(dist_function: Callable):
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
    @njit(float64[:](uint16, uint16, uint64))
    def generate_data(k: int, n: int, seed: int):
        """
        Generate flat array of random data from named distribution.

        Parameters
        ----------
        k : uint16
            Number of treatment groups (data_array rows)
        n : uint16
            Number of observations per treatment group (data_array columns)
        seed : uint64
            Seed passed to np.random before data generation

        Returns
        -------
        result : float64[:]
            contiguous numpy array of length (k*n)
        """
        result = np.ascontiguousarray(dist_function(k * n, seed))
        return result
    return generate_data

gen_data_F1 = gen_data_factory(F1)
gen_data_F2 = gen_data_factory(F2)
gen_data_F3 = gen_data_factory(F3)
gen_data_F4 = gen_data_factory(F4)
gen_data_F5 = gen_data_factory(F5)
gen_data_F6 = gen_data_factory(F6)
gen_data_F7 = gen_data_factory(F7)
gen_data_F8 = gen_data_factory(F8)