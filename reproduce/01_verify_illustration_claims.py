"""
Verify claims in Part 3 of manuscript

The follow scripts verify the claims made in Part 3 regarding the application
of the JT and DD tests to the illustrative example of selected data from an 
attacker/defender experiment reported by Holt, Sahu and Smith (2022). 
"""

import numpy as np

from monte_carlo.stats import gen_perm_array
from monte_carlo.stats.wrappers_cpu import (
    permutation_test_J_cpu,
    permutation_test_D_cpu
)
from monte_carlo.utils import print_section


# --- Ingest Table 1 ---

print_section("Table 1, selected data from Holt, Sahu, and Smith (2022)")

# Ingest 2D table for display, 1D table for processing
tbl_1 = np.array([
	    [0.10, 0.23, 0.23],
	    [0.33, 0.20, 0.47],
	    [0.57, 0.43, 0.40],
    ], dtype=np.float64)
tbl_1_flat = tbl_1.ravel()

# Print to verify accuracy
np.set_printoptions(
 precision=2,
 suppress=True,
 linewidth=80
)
print(tbl_1)


# --- Precompute permutation array ---

# Precompute array of all 1_680 possible permutations of sample data 
# where within-group order does not matter.
perm_array_33 = gen_perm_array(3, 3)


# --- Run JT test ---

print_section("JT test, applied to data in Table 1")

results_JT = permutation_test_J_cpu(3, 3, tbl_1_flat, perm_array_33)

print(f"J_obs:              {results_JT[1]}")
print(f"Null J gte J_obs:   {results_JT[2]}")
print(f"JT test p-value:    {results_JT[0]:.4f}")


# --- Run DD test ---

print_section("DD test, applied to data in Table 1")

results_DD = permutation_test_D_cpu(3, 3, tbl_1_flat, perm_array_33)

print(f"D_obs:              {results_DD[1]:.2f}")
print(f"Null D gte D_obs:   {results_DD[2]}")
print(f"DD test p-value:    {results_DD[0]:.4f}")