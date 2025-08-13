import argparse
from copy import deepcopy
import json
import os
import sys

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

from data_class.ride_class import Ride, triaxial_metrics


def process_data(args):
    # Category definitions Nmv, Nvd, Nva
    categories = [
        ((0, 1.5), 'Very Comfortable'),
        ((1.5, 2.5), 'Comfortable'),
        ((2.5, 3.5), 'Medium'),
        ((3.5, 4.5), 'Uncomfortable'),
        ((4.5, 6.0), 'Very Uncomfortable')
    ]

    # Category definitions Ccy, Ccz - might need to add to categories.py
    cats_cont = [
        ((0,0.2), 'Very Comfortable'),
        ((0.2,0.3), 'Comfortable'),
        ((0.3,0.4), 'Medium'),
        ((0.4,1), 'Less Comfortable')
    ]

    # Pair floor & seat triaxials (floor,seat)
    pairs = [
        ((1,6), 'Motor Truck'),
        ((3,2), 'Between Trucks'),
        ((5,4), 'Center Truck')
    ]

    # Floor triaxes (1,3,5) get a N_V_D, but seat triaxes get a N_V_A.

    floor_triaxes = [1, 3, 5]

    # instantiate Ride object
    ride = Ride(args.run, args.date, json_path=args.json_path)

    # Load data - will need to load paired triaxes (floor = p, seat = a&d)
    non_rave, rave = load_all_triax(args.data_path)

    metrics_dict = deepcopy(triaxial_metrics)

    for triax in range(1, 7):
        rave_curr = rave[str(triax)]
        
        if rave_curr is None:
            continue

        # Calculate metrics for each triaxial
        # Every 5th running average
        if triax == 5:
            x5 = rave_curr["4"].iloc[4::5]
            y5 = rave_curr["5"].iloc[4::5]
            z5 = rave_curr["6"].iloc[4::5]
        else:
            x5 = rave_curr["1"].iloc[4::5]
            y5 = rave_curr["2"].iloc[4::5]
            z5 = rave_curr["3"].iloc[4::5]

        Cx = x5['ISO-WD']
        Cy = y5['ISO-WD']
        Cz = z5['EN-WB']
        metrics_dict[str(triax)]["Cx"] = Cx
        metrics_dict[str(triax)]["Cy"] = Cy
        metrics_dict[str(triax)]["Cz"] = Cz
        metrics_dict[str(triax)]["Cx_mean"], metrics_dict[str(triax)]["Cx_std"] = get_mean_and_std(Cx)
        metrics_dict[str(triax)]["Cy_mean"], metrics_dict[str(triax)]["Cy_std"] = get_mean_and_std(Cy)
        metrics_dict[str(triax)]["Cz_mean"], metrics_dict[str(triax)]["Cz_std"] = get_mean_and_std(Cz)
        metrics_dict[str(triax)]["max_Cx"] = Cx.max()
        metrics_dict[str(triax)]["max_Cy"] = Cy.max()
        metrics_dict[str(triax)]["max_Cz"] = Cz.max()


        # Percentile inputs
        Ax95 = np.percentile(Cx, 95)
        Ay95 = np.percentile(Cy, 95)
        Az95 = np.percentile(Cz, 95)
        Ax50 = np.percentile(Cx, 50)
        Ay50 = np.percentile(Cy, 50)
        Az50 = np.percentile(Cz, 50)

        # Calculate comfort indices
        N_MV = compute_nmv(Ax95, Ay95, Az95)
        if triax in floor_triaxes:
            N_VD = compute_nvd(Ax50, Ay50, Az50, Ay95)
        else:
            N_VD = None
        metrics_dict[str(triax)]["N_MV"] = N_MV
        metrics_dict[str(triax)]["N_VD"] = N_VD


        print(f"triax:{triax} N_M_V = {N_MV:.2f} → {categorize(N_MV, categories)}")

        if N_VD is not None:
            print(f"triax:{triax} N_V_D = {N_VD:.2f} → {categorize(N_VD, categories)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ride comfort data.")
    parser.add_argument(
        "--data_path", type=str, default="data/",
        help="Path to the directory containing ride data files."
    )
    parser.add_argument(
        "--json_path", type=str, default="data_schema.json",
        help="Path to the JSON schema file."
    )
    parser.add_argument("--output_path", type=str, default="output/",
        help="Path to save the processed data and plots."
    )
    parser.add_argument( 
        "-date", type=str,
        help="Date of the ride data to be processed. Must be one of the valid dates."
    )
    parser.add_argument(
        "-run", type=str,
        help="Run number to be processed."
    )
    
    args = parser.parse_args()

    process_data(*args)