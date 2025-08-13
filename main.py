import numpy as np
import pandas as pd


from analysis.loaders import load_all_triax
from analysis.metrics import (
    compute_nmv, compute_nvd, compute_av,
    compute_vdv, compute_vdv_ratios, compute_nva,
    get_mean_and_std
)
from analysis.categories import categorize
from analysis.plotting import (
    plot_comfort_thresholds_nvm,
    plot_comfort_thresholds_nvd_nva,
    plot_vdv_over_time,
    plot_ratio_comparison,
    plot_compare_all_metrics,
    plot_comfort_timeseries,
    plot_iso_timeseries,
    plot_distributions,
    plot_cumulative_distribution
)



def process_data():
    pass

if __name__ == "__main__":
    process_data()