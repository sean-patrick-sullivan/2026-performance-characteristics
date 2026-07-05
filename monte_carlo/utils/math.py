"""
Math and counting tools

This module provides functions useful for computing permutation counts.
"""

from numba import njit, uint64

# --- Count tools ---

@njit(uint64(uint64))
def factorial(n: int):
    """ Compute factorial in numba-compatible method. """
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit(uint64(uint64,uint64))
def permutation_count(k: int, n: int):
    """ Compute permutation count in numba-compatible method. """
    return factorial(k*n) / (factorial(n)**k )