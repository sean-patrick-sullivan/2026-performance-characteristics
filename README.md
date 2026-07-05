# Data and Replication Materials for Holt, Kwiatkowski & Sullivan, Performance Characteristics of a Directional Difference Test (2026)

This repository collects reported data and reproduction code for Charles A. Holt, Daniel Kwiatkowski, & Sean P. Sullivan, Performance Characteristics of a Directional Difference Test (2026). The structure of data, artifacts, simulation code, and analysis scripts is as follows:

```pre
├── analysis (R code for producing manuscript exhibits)
├── data (output directory for data collected from Monte Carlo studies)
│   └── manuscript (archive version of data reported in manuscript)
├── monte_carlo (Python package for running Monte Carlo studies)
│   ├── dists
│   ├── engine
│   ├── orchestration
│   ├── stats
│   └── utils
├── outputs (exhibits illustrating data collected from Monte Carlo studies)
│   ├── plots
│   │   ├── densities
│   │   └── power
│   │       ├── principal_study
│   │       ├── additional_study
│   │       └── secondary_study
│   └── tables
├── reproduce (Python scripts for reproducing Monte Carlo study results)
├── tests (unit tests for random number generators)
├── ...
└── README.md
```


## Accessing simulation data and exhibits

### Raw data

Raw `.csv` files for all data described in the manuscript are available in `data/manuscript`. For example, `data/manuscript/principal_symmetric.csv` and `data/manuscript/principal_asymmetric.csv` provide study results for the principal Monte Carlo study described in manuscript Parts 5.1 and 5.2. 

- All simulation data in `.csv` format: [all raw data](data/manuscript/)

### Power plots

Power plots for every combination of Monte Carlo study parameters are available in `outputs/plots/power`. For example, the plot reproduced as mansucript Figure 1 is available at `outputs/plots/power/principal_study/symmetric_alpha05_n3.pdf`. Power plots for other combinations of study parameters are indexed via the pattern `outputs/plots/power/[study_version]/[shift-type]_[nominal_size]_[group_n]`.

- Principal study power plots for $k=3$, $n=2,3,4$, $F=F_1,...F_8$, $\alpha=0.1,0.05,0.01$, symmetric and asymmetric shifts: [principal study power plots](/outputs/plots/power/principal_study/)
- Additional study power plots for $k=4$, $n=2,3$, $F=F_1,...F_8$, $\alpha=0.1,0.05,0.01$, symmetric and asymmetric shifts: [additional study power plots](/outputs/plots/power/additional_study/)
- Secondary study power-variance plots for $k=3$, $n=3$, $F(\sigma^2)=F_1(\sigma^2),...F_8(\sigma^2)$, $\alpha=0.05$, symmetric shift: [secondary study power-variance plots](/outputs/plots/power/secondary_study/)

### Other exhibits

Other exhibits referenced in the manuscript or online appendix are also available in `outputs`.

- Size measurements from principal study in `.csv` format: [principal study size table](/outputs/tables/size/principal_study/test_size.csv)
- Graphical illustration of distributions $F_1,\ldots,F_8$: [density plots](/outputs/plots/densities/dist_densities_F1-F8.pdf)


## Running and reproducing Monte Carlo studies

The Monte Carlo package is written in Python and relies on packages listed as dependencies in `pyproject.toml`. The package relies heavily on `numba>=0.65.0` to achive reasonable compute times when running on CPU. When running on GPU, the package requires appropriate `cuda` libraries, including the following:
```
cuda-bindings>=13.2.0
cuda-core>=0.7.0
cuda-pathfinder>=1.5.2
numba-cuda>=0.30.0
```

Running simulations involves first installing the Monte Carlo package:

```sh
# from repository root ...

# install local package
$ pip install -e .

# in some cases, breaking changes may need to be permitted
$ pip install --break-system-packages -e .
```

Individual studies and manuscript assertions can be reproduced via scripts located in `reproduce/`:

```sh
# reproduce results asserted in manuscript Part 3
$ python3 reproduce/01_verify_illustration_claims.py

# reproduce results of principal Monte Carlo study describe in Parts 5.1 and 5.2
$ python3 reproduce/02_run_principal_study_symmetric_shifts.py
$ python3 reproduce/03_run_principal_study_asymmetric_shifts.py

# reproduce results of additional and secondary Monte Carlo studies described in Part 5.3
$ python3 reproduce/04_run_additional_study_symmetric_shifts.py
$ python3 reproduce/05_run_additional_study_asymmetric_shifts.py
$ python3 reproduce/06_run_secondary_study.py
```

By default, all run-study scripts are coded to execute reduced-size example studies. Parameter values for the full studies reported in the manuscript are provided in comments and can be executed by commenting out the example-study commands and uncommenting the full-study commands. Alternative Monte Carlo studies can be performed by manipulating arguments and parameter choices according to function documentation in the `monte_carlo` package.


## Quick start guide for a local environment

The simulations reported in the manuscript relied on dedicated GPU accelleration hardware. Running comparable simulations on CPU with desktop hardware would entail many hours of compute time, but reduced-size simulations will complete reasonably quickly on modern destkop hardware. Compute time scales linearly in the number of simulations performed, enabling accurate prediction of full-study compute time based on small pilot studies.

For fast and ephemeral reproduction of study results, the following `Dockerfile` produces a working environment:

```Dockerfile
# Use official Python runtime as parent image
FROM python:3.14

# Set the working directory to /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy>=2.4.6" \
        "scipy>=1.18.0" \
        "numba>=0.65.0" \
        "pandas>=3.0.3" \
        "typer>=0.26.7" \
        "pytest>=9.1.1"

# Override the default entrypoint to use bash
ENTRYPOINT ["/bin/bash"]
```

The resulting container can be used by bind-mounting the repository root to `/app`:

```sh
$ docker run -it --rm -v /path/to/repostiory/root:/app container_name
```


## References

Holt, Charles A., Daniel Kwiatkowski & Sean P. Sullivan, Performance Characteristics of a Directional Difference Test (2026).