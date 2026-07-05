"""
Run secondary Monte Carlo study (with symmetric shifts)

The following script performs the secondary Monte Carlo study described in 
Part 5.3 of the manuscript, considering distribution parameterizations beyond 
those explored in the principal study. The secondary study uses fixed, 
symmetric shifts, and distribution families parameterized by their variance
(or median absolute deviation, in the case of the Cauchy distribution), 
constructed to reproduce principal-study distributions as a special case.
"""

from pathlib import Path

from monte_carlo.orchestration import run_secondary_study


# --- Set output location for data file ---

script_location = Path(__file__).resolve().parent
output_file = script_location.parent / "data" / "secondary_symmetric.csv"


# --- Run Monte Carlo study ---

treatment_shift_dict = {
    3: { 
        3: {
            'F1': 7.5,
            'F2': 15.,
            'F3': 3.0,
            'F4': 15.,
            'F5': 7.5,
            'F6': 7.5,
            'F7': 7.5,
            'F8': 7.5,
        },
    }
}

var_bounds_dict = {
    3: { 
        3: {
            'F1': (15.,85.),
            'F2': (50.,350.),
            'F3': (4.,14.),
            'F4': (100.,350.),
            'F5': (15.,85.),
            'F6': (0.25,1.75),
            'F7': (15.,85.),
            'F8': (15.,85.),
        },
    }
}

"""
The following parameters reproduce the data described in the manuscript.
Executing this version of the study requires a compatible NVIDIA GPU and,
depending on hardware, may take several hours to complete.

results = run_secondary_study(
    k_array = [3],
    n_array = [3],
    treatment_shift_dict = treatment_shift_dict,
    num_vars = 150,
    var_bounds_dict = var_bounds_dict,
    num_simulations = 250_000,
    treatment = "symmetric",
    base_seed = 138492,
    compute_arch="gpu"
)

For convenience in quickly verifying the design and result patterns of the
reported study, the following reduced-size simulation will complete within 
a few minutes on any reasonably powerful modern CPU. Compute time scales 
linearly with the number of simulations being performed.
"""

results = run_secondary_study(
    k_array = [3],
    n_array = [3],
    treatment_shift_dict = treatment_shift_dict,
    num_vars = 150,
    var_bounds_dict = var_bounds_dict,
    num_simulations = 2_500,
    treatment = "symmetric",
    base_seed = 138492,
    compute_arch="cpu"
)

# --- Save results to file ---

results.to_csv(output_file, index=False)