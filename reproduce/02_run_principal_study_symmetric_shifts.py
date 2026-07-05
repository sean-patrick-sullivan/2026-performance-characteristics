"""
Run principal Monte Carlo study with symmetric treatment shifts

The following script performs the symmetric portion of the principal Monte 
Carlo study described in Parts 5.1 and 5.2 of the manuscript. This part of the
study simulates data sets with symmetric treatment shifts as described in 
expression (7).
"""

from pathlib import Path

from monte_carlo.orchestration import run_principal_study


# --- Set output location for data file ---

script_location = Path(__file__).resolve().parent
output_file = script_location.parent / "data" / "principal_symmetric.csv"


# --- Run Monte Carlo study ---

max_shift_dict = {
    3: { 
        2: { 
            'F1': 25.,
            'F2': 50.,
            'F3': 12.5,
            'F4': 45.,
            'F5': 25.,
            'F6': 20.,
            'F7': 25.,
            'F8': 30.,
        },
        3: {
            'F1': 15.,
            'F2': 30.,
            'F3': 7.,
            'F4': 35.,
            'F5': 15.,
            'F6': 15.,
            'F7': 15.,
            'F8': 20.,
        },
        4: {
            'F1': 15.,
            'F2': 25.,
            'F3': 7,
            'F4': 25.,
            'F5': 15.,
            'F6': 15.,
            'F7': 12.5,
            'F8': 15.,
        }
    }
}

"""
The following parameters reproduce the data described in the manuscript.
Executing this version of the study requires a compatible NVIDIA GPU and,
depending on hardware, may take several hours to complete.

results = run_principal_study(
    k_array = [3],
    n_array = [2,3,4],
    num_treatments = 90,
    max_shift_dict = max_shift_dict,
    num_simulations = 250_000,
    treatment = "symmetric",
    base_seed = 863458,
    compute_arch="gpu"
)

For convenience in quickly verifying the design and result patterns of the
reported study, the following reduced-size simulation will complete within 
a few minutes on any reasonably powerful modern CPU. Compute time scales 
linearly with the number of simulations being performed.
"""

results = run_principal_study(
    k_array = [3],
    n_array = [2,3,4],
    num_treatments = 90,
    max_shift_dict = max_shift_dict,
    num_simulations = 250,
    treatment = "symmetric",
    base_seed = 863458,
    compute_arch="cpu"
)


# --- Save results to file ---

results.to_csv(output_file, index=False)