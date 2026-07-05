from monte_carlo.stats.permutations import gen_perm_array
from monte_carlo.stats.stats_cpu import compute_stats_for_perm_cpu
from monte_carlo.stats.stats_gpu import compute_stats_for_perm_gpu

__all__ = [
    "gen_perm_array",
    "compute_stats_for_perm_cpu",
    "compute_stats_for_perm_gpu",
]