"""
Generate permutation index arrays

This module provides a recursive function for generating all $(n*k)!/(n!)^k$ 
possible permutation index arrays for a given combination of k and n values.
"""

from itertools import combinations
import numpy.typing as npt

import numpy as np
from numpy import uint16

from monte_carlo.utils import permutation_count

# --- Generation of permutation index array ---

def gen_perm_array(k: uint16, n: uint16) -> npt.NDArray[np.uint16]:
    """
    Recursive generation of all possible permutations of index values.

    Parameters
    ----------
    k : uint16
        Number of groups
    n : uint16
        Number of observations per group

    Returns
    -------
    np.array : uint16[:,:]
        Array of all $(n*k)!/(n!)^k$ sets of possible permutation indexes, 
        returned as an array of shape (num_perms, n*k)
    
    Notes
    -----
    Recursive construction of the array is slower than more complicated 
    stack-based construction, but Monte Carlo orchestration layers minimize 
    this performance penalty by precomputing permutation index arrays once 
    outside the primary Monte Carlo loops. Because the effective compute cost
    is low, this implementation prioritizes intuition and legibility over
    raw performance.
    """
    nk = k * n

    def partitions(available, groups_left):
        """Yield all ordered partitions of `available` into `groups_left` groups of size n."""
        if groups_left == 0:
            yield ()
            return
        for combination in combinations(available, n):
            remaining = tuple(x for x in available if x not in set(combination))
            for rest in partitions(remaining, groups_left - 1):
                yield combination + rest

    return np.array(list(partitions(tuple(range(nk)), k)), dtype=np.uint16)