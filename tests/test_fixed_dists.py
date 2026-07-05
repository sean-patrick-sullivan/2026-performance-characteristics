"""
Test random number generation from F1 through F8

Unit tests to check that random number generation functions produce results
that conform to predicted moments or analogous measures.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, float64
import pytest

from monte_carlo.dists.fixed_dists import *

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
    
    if n % 2 == 0:
        return (sorted[mid_idx - 1] + sorted[mid_idx]) / 2.0
    else:
        return sorted[mid_idx]

@njit(float64(float64[:]))
def mad(x: npt.NDArray[np.float64]) -> float:
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
    (F1, 0.0, 50.0),
    (F2, 0.0, 200.0),
    (F3, 0.0, 9.0),
    (F4, 0.0, 225.0),
    (F5, 0.0, 50.0),
    (F7, 0.0, 50.0),
    (F8, 0.0, 50.0),
]

POORLY_BEHAVED_MEASURES = [
    (F6, 0.0, 1.0),
]

@pytest.mark.parametrize("F_fun, expected_mean, expected_var", WELL_BEHAVED_MOMENTS)
def test_well_behaved(F_fun, expected_mean, expected_var):
    """
    Test mean and variance of well-behaved distributions
    """
    # Sample generation
    sample = F_fun(SAMPLE_SIZE, SEED)
    
    # Empirical moments
    sample_mean = np.mean(sample)
    sample_var = np.var(sample)
    
    # Test conformity to expectation
    assert np.isclose(sample_mean, expected_mean, atol=0.1), \
        f"{F_fun.__name__} mean {sample_mean} far from {expected_mean}"
        
    assert np.isclose(sample_var, expected_var, rtol=0.1), \
        f"{F_fun.__name__} variance {sample_var} far from {expected_var}"

@pytest.mark.parametrize("F_fun, expected_median, expected_mad", POORLY_BEHAVED_MEASURES)
def test_poorly_behaved(F_fun, expected_median, expected_mad):
    """
    Test median and median absolute deviation for poorly-behaved distributions
    """
    # Sample generation
    sample = F_fun(SAMPLE_SIZE, SEED)
    
    # Empirical measures
    sample_median = median(sample)
    sample_mad = mad(sample)
    
    # Test conformity to expectation
    assert np.isclose(sample_median, expected_median, atol=0.1), \
        f"{F_fun.__name__} median {sample_median} far from {expected_median}"
        
    assert np.isclose(sample_mad, expected_mad, rtol=0.1), \
        f"{F_fun.__name__} MAD {sample_mad} far from {expected_mad}"