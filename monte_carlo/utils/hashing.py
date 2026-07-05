"""
Hashing tools for seed management

This module provides functions for hashing Monte Carlo profiles
and simulation seeds.
"""

from numba import njit, uint16, uint64


# --- Hash utilities ---

@njit(uint64(uint64,uint64))
def hash_seed(hash_base: int, hash_value: int):
    """ 
    Hash two values

    Parameters
    ----------
    hash_base : uint64
        Base to use in hash algorithm
    hash_value : uint64
        Value to use in hash algorithm
    
    Returns
    -------
    uint64
        Return hash
    
    Notes
    -----
    Simple implementation of the algorithm suggested by Sebastiano
    Vigna (2015): https://prng.di.unimi.it/splitmix64.c
    """
    z = uint64(hash_base) ^ uint64(hash_value)
    z = (z ^ (z >> 30)) * uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> 27)) * uint64(0x94d049bb133111eb)
    return z ^ (z >> 31)

@njit(uint64(uint64, uint16, uint16, uint16, uint16))
def hash_principal_profile(base_seed: int, k: int, n: int, 
                           dist_idx: int, treatment_idx: int):
    """ 
    Hash for principal simulation profile

    Parameters
    ----------
    base_seed : uint64
        Base to use in hash algorithm
    k : uint16
        Number of groups
    n : uint16
        Number of observations per group
    dist_idx : uint16
        Number id of distribution
    treatment_idx : uint16
        Number id of treatment shift
    
    Returns
    -------
    z : uint64
        Return hash
    """
    z = hash_seed(base_seed, uint64(k))
    z = hash_seed(z, uint64(n))
    z = hash_seed(z, uint64(dist_idx))
    z = hash_seed(z, uint64(treatment_idx))
    return z

@njit(uint64(uint64, uint16, uint16, uint16, uint16))
def hash_secondary_profile(base_seed: int, k: int, n: int, 
                           dist_idx: int, parameter_idx: int):
    """ 
    Hash for secondary simulation profile

    Parameters
    ----------
    base_seed : uint64
        Base to use in hash algorithm
    k : uint16
        Number of groups
    n : uint16
        Number of observations per group
    dist_idx : uint16
        Number id of distribution
    parameter_idx : uint16
        Number id of parameter value
    
    Returns
    -------
    z : uint64
        Return hash
    """
    z = hash_seed(base_seed, uint64(k))
    z = hash_seed(z, uint64(n))
    z = hash_seed(z, uint64(dist_idx))
    z = hash_seed(z, uint64(parameter_idx))
    return z