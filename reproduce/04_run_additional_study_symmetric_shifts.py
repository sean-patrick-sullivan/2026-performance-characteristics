"""
Run additional Monte Carlo study with symmetric treatment shifts

The following script performs the symmetric portion of the additional Monte 
Carlo study described in Part 5.3 of the manuscript, considering sample shapes
beyond those explored in the principal study. This part of the additional
study simulates data sets with symmetric treatment shifts as described in 
expression (7).
"""

from pathlib import Path

from monte_carlo.orchestration import run_principal_study


# --- Set output location for data file ---

script_location = Path(__file__).resolve().parent
output_file = script_location.parent / "data" / "additional_symmetric.csv"


# --- Run Monte Carlo study ---

max_shift_dict = {
    4: { 
        2: { 
            'F1': 10.,
            'F2': 22.5,
            'F3': 5.,
            'F4': 22.5,
            'F5': 12.5,
            'F6': 15.,
            'F7': 12.5,
            'F8': 15.,
        },
        3: {
            'F1': 9.,
            'F2': 17.5,
            'F3': 4.,
            'F4': 17.5,
            'F5': 9.,
            'F6': 15.,
            'F7': 9.,
            'F8': 10.,
        },
    }
}

"""
The following parameters reproduce the data described in the manuscript.
Executing this version of the study requires a compatible NVIDIA GPU and,
depending on hardware, may take several hours to complete.

results = run_principal_study(
    k_array = [4],
    n_array = [2,3],
    num_treatments = 90,
    max_shift_dict = max_shift_dict,
    num_simulations = 50_000,
    treatment = "symmetric",
    base_seed = 263453,
    compute_arch="gpu"
)

For convenience in quickly verifying the design and result patterns of the
reported study, the following reduced-size simulation will complete within 
a few minutes on any reasonably powerful modern CPU. Compute time scales 
linearly with the number of simulations being performed.
"""

results = run_principal_study(
    k_array = [4],
    n_array = [2,3],
    num_treatments = 90,
    max_shift_dict = max_shift_dict,
    num_simulations = 100,
    treatment = "symmetric",
    base_seed = 263453,
    compute_arch="cpu"
)

# --- Save results to file ---

results.to_csv(output_file, index=False)