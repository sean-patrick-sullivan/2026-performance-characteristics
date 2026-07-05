"""
Run principal Monte Carlo study with asymmetric treatment shifts

The following script performs the asymmetric portion of the principal Monte 
Carlo study described in Parts 5.1 and 5.2 of the manuscript. This part of the
study simulates data sets with asymmetric treatment shifts as described in 
expression (7).
"""

from pathlib import Path

from monte_carlo.orchestration import run_principal_study


# --- Set output location for data file ---

script_location = Path(__file__).resolve().parent
output_file = script_location.parent / "data" / "principal_asymmetric.csv"


# --- Run Monte Carlo study ---

max_shift_dict = {
    3: { 
        2: { 
            'F1': 50.,
            'F2': 100.,
            'F3': 25.,
            'F4': 90.,
            'F5': 50.,
            'F6': 45.,
            'F7': 50.,
            'F8': 50.,
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
        4: {
            'F1': 30.,
            'F2': 60.,
            'F3': 15.,
            'F4': 60.,
            'F5': 30.,
            'F6': 40.,
            'F7': 30.,
            'F8': 35.,
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
    treatment = "asymmetric",
    base_seed = 649125,
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
    treatment = "asymmetric",
    base_seed = 649125,
    compute_arch="cpu"
)

# --- Save results to file ---

results.to_csv(output_file, index=False)