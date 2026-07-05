"""
Run additional Monte Carlo study with asymmetric treatment shifts

The following script performs the asymmetric portion of the additional Monte 
Carlo study described in Part 5.3 of the manuscript, considering sample shapes
beyond those explored in the principal study. This part of the additional
study simulates data sets with asymmetric treatment shifts as described in 
expression (7).
"""

from pathlib import Path

from monte_carlo.orchestration import run_principal_study


# --- Set output location for data file ---

script_location = Path(__file__).resolve().parent
output_file = script_location.parent / "data" / "additional_asymmetric.csv"


# --- Run Monte Carlo study ---

max_shift_dict = {
    4: { 
        2: { 
            'F1': 70.,
            'F2': 150.,
            'F3': 30.,
            'F4': 150.,
            'F5': 75.,
            'F6': 60.,
            'F7': 75.,
            'F8': 75.,
        },
        3: {
            'F1': 35.,
            'F2': 70.,
            'F3': 17.5,
            'F4': 80.,
            'F5': 40.,
            'F6': 50.,
            'F7': 40.,
            'F8': 45.,
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
    treatment = "asymmetric",
    base_seed = 926583,
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
    treatment = "asymmetric",
    base_seed = 926583,
    compute_arch="cpu"
)

# --- Save results to file ---

results.to_csv(output_file, index=False)