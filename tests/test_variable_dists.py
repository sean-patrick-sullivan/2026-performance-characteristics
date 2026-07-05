"""
Test random number generation from var_F1 through var_F8

Unit tests to check that random number generation functions produce results
that conform to predicted moments or analogous measures.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, float64
import pytest

from monte_carlo.dists.variable_dists import *

# --- Median and MAD ---

@njit(float64(float64[:]))
def median(x: npt.NDArray[np.float64]) -> float:
    """
    Compute the empirical median of a 1D array
    
    Parameters
    ----------
    x : float64[:]
        sample of observations
        
    Returns
    -------
    float64
        Empirical median
    """
    n = len(x)
    if n == 0:
        return np.nan
    
    sorted = np.sort(x)
    mid_idx = n // 2
    
    if n % 2 == 0: # Even number of elements
        return (sorted[mid_idx - 1] + sorted[mid_idx]) / 2.0
    else:
        return sorted[mid_idx]

@njit(float64(float64[:]))
def median_abs_deviation(x: npt.NDArray[np.float64]) -> float:
    """
    Compute the empirical median absolute deviation
    
    Parameters
    ----------
    x : float64[:]
        sample of observations
        
    Returns
    -------
    float64
        Empirical median absolute deviation
    """
    n = len(x)
    if n == 0:
        return np.nan
    
    m = median(x)
    abs_deviations = np.abs(x - m)
    
    return median(abs_deviations)


# --- Test Suite ---

SAMPLE_SIZE = 500_000
SEED = 12345

WELL_BEHAVED_MOMENTS = [
    (var_F1, (25.00, 50.0, 75.0)),
    (var_F2, (100., 200., 300.)),
    (var_F3, (4., 9., 14.)),
    (var_F4, (125., 225., 325.)),
    (var_F5, (25., 50., 75.)),
    (var_F7, (25., 50., 75.)),
    (var_F8, (25., 50., 75.)),
]

POORLY_BEHAVED_MEASURES = [
    (var_F6, (0.5, 1.0, 1.5)),
]

@pytest.mark.parametrize("var_F_fun, vars", 
                         WELL_BEHAVED_MOMENTS)
def test_well_behaved(var_F_fun, vars):
    """
    Test mean and variance of well-behaved distributions
    """
    for var in vars:

        # Sample generation
        sample = var_F_fun(SAMPLE_SIZE, var, SEED)
        
        # Predicted moments
        expected_mean = 0.0
        expected_var = var
        
        # Empirical moments
        sample_mean = np.mean(sample)
        sample_var = np.var(sample)
        
        # Test conformity to expectation
        assert np.isclose(sample_mean, expected_mean, atol=0.1), \
            f"{var_F_fun.__name__} mean {sample_mean} far from {expected_mean}"
            
        assert np.isclose(sample_var, expected_var, rtol=0.1), \
            f"{var_F_fun.__name__} variance {sample_var} far from {expected_var}"

@pytest.mark.parametrize("var_F_fun, mads", 
                         POORLY_BEHAVED_MEASURES)
def test_poorly_behaved(var_F_fun, mads):
    """
    Test mean and variance of well-behaved distributions
    """
    for mad in mads:

        # Sample generation
        sample = var_F_fun(SAMPLE_SIZE, mad, SEED)
        
        # Predicted moments
        expected_median = 0.0
        expected_mad = mad
        
        # Empirical moments
        sample_median = median(sample)
        sample_mad = median_abs_deviation(sample)
        
        # Test conformity to expectation
        assert np.isclose(sample_median, expected_median, atol=0.1), \
            f"{var_F_fun.__name__} median {sample_median} far from {expected_median}"
            
        assert np.isclose(sample_mad, expected_mad, rtol=0.1), \
            f"{var_F_fun.__name__} mad {sample_mad} far from {expected_mad}"