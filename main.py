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
    ride_obj = Ride(args.run, args.date, json_path=args.json_path)

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
    
    for triax in range(1, 7):
        if metrics_dict[str(triax)]["N_MV"] is None:
            continue
        plot_distributions(metrics_dict[str(triax)]["Cx"],
                        metrics_dict[str(triax)]["Cy"],
                        metrics_dict[str(triax)]["Cz"], triax=triax, ride_obj=ride_obj, save=True, save_dir=args.data_path)
    
    for triax in range(1, 7):
        if metrics_dict[str(triax)]["N_MV"] is None:
            continue
        plot_cumulative_distribution(metrics_dict[str(triax)]["Cx"],
                        metrics_dict[str(triax)]["Cy"],
                        metrics_dict[str(triax)]["Cz"], triax=triax, ride_obj=ride_obj, save=True, save_dir=args.data_path)
    
    for (floor, seat), name in pairs:
        floor_curr = rave[str(floor)]
        seat_curr = rave[str(seat)]

        if floor_curr is None or seat_curr is None:
            continue

        
        seat_Cx = metrics_dict[str(seat)]["Cx"]
        seat_Cy = metrics_dict[str(seat)]["Cy"]
        seat_Cz = metrics_dict[str(seat)]["Cz"]

        # Percentile inputs
        seat_Ax95 = np.percentile(seat_Cx, 95)
        seat_Ay95 = np.percentile(seat_Cy, 95)
        seat_Az95 = np.percentile(seat_Cz, 95)

        floor_Cz = metrics_dict[str(floor)]["Cz"]
        floor_Azp_95 = np.percentile(floor_Cz, 95)

        NVA = compute_nva(
            floor_Azp_95, seat_Ay95, seat_Az95, seat_Ax95
        )
        metrics_dict[str(seat)]["N_VA"] = NVA

        print(f"{name} N_V_A = {NVA:.2f} → {categorize(NVA, categories)}")
    

    N_MVs = []
    for triax in range(1, 7):
        
        N_MV = metrics_dict[str(triax)]["N_MV"]

        if N_MV is None:
            continue

        N_MVs.append((triax, N_MV))

    plot_comfort_thresholds_nvm(N_MVs, categories, floor_triaxes, ride_obj, safe=True, save_dir=args.data_path)

    N_VAs = []
    N_VDs = []
    for triax in range(1, 7):
        N_VD = metrics_dict[str(triax)]["N_VD"]
        N_VA = metrics_dict[str(triax)]["N_VA"]
        
        N_VDs.append((triax, N_VD))
        N_VAs.append((triax, N_VA))

    plot_comfort_thresholds_nvd_nva(N_VDs, N_VAs, categories, pairs, ride_obj, save=True, save_dir=args.data_path)

    for triax in range(1, 7):
        raw = non_rave[str(triax)]
        
        if raw is None:
            continue

        # Full time vector 
        if triax == 5:
            t_all = pd.to_timedelta(raw["4"]['Number'], unit='s')
        else:
            t_all = pd.to_timedelta(raw["1"]['Number'], unit='s')
        # Downsampled time vector to match 5s interval RAVE slices
        t_5s = t_all[4::5]
        t_5s_minutes = t_5s.dt.total_seconds() / 60
        # 5s signals plot
        Cx, Cy, Cz = metrics_dict[str(triax)]["Cx"], metrics_dict[str(triax)]["Cy"], metrics_dict[str(triax)]["Cz"]
        plot_comfort_timeseries(t_5s_minutes, Cx, Cy, Cz, ride_obj, save=True, save_dir=args.data_path)
    
    #### For av calculation, note that I'm not implementing the kb control flow ###

    for triax in range(1, 7):
        raw = non_rave[str(triax)]
        
        if raw is None:
            continue
        
        # Compute AV
        if triax == 5:
            X_ISO = raw["4"]['ISO-WD']
            Y_ISO = raw["5"]['ISO-WD']
            Z_ISO = raw["6"]['ISO-WB']
        else:
            X_ISO = raw["1"]['ISO-WD']
            Y_ISO = raw["2"]['ISO-WD']
            Z_ISO = raw["3"]['ISO-WB']
        
        a_v = compute_av(X_ISO, Y_ISO, Z_ISO)
        metrics_dict[str(triax)]["av"] = a_v
        # Compute a_v 
        if triax == 5:
            a_v_5s = compute_av(rave[str(triax)]["4"]['ISO-WD'], rave[str(triax)]["5"]['ISO-WD'], rave[str(triax)]["6"]['ISO-WB'])
        else:
            a_v_5s = compute_av(rave[str(triax)]["1"]['ISO-WD'], rave[str(triax)]["2"]['ISO-WD'], rave[str(triax)]["3"]['ISO-WB'])
        
        metrics_dict[str(triax)]["av_5s"] = a_v_5s
        metrics_dict[str(triax)]["ax"] = X_ISO
        metrics_dict[str(triax)]["ay"] = Y_ISO
        metrics_dict[str(triax)]["az"] = Z_ISO
        metrics_dict[str(triax)]["max_ax"] = X_ISO.max()
        metrics_dict[str(triax)]["max_ay"] = Y_ISO.max()
        metrics_dict[str(triax)]["max_az"] = Z_ISO.max()
        metrics_dict[str(triax)]["max_av"] = a_v.max()
        metrics_dict[str(triax)]["max_av_5s"] = a_v_5s.max()

        print(f"triax:{triax} a_v (1s ISO composite) = {np.mean(a_v):.3f} m/s²")
        print(f"triax:{triax}  a_v_5s (from RAVE)      = {np.mean(a_v_5s):.3f} m/s²")
    
    for triax in range(1, 7):
        rave_curr = rave[str(triax)]
        raw = non_rave[str(triax)]

        if raw is None or rave_curr is None:
            continue

        if triax == 5:
            t_5s_minutes = pd.to_timedelta(raw["4"]['Number'], unit='s')[4::5].dt.total_seconds() / 60

            x5 = rave_curr["4"]['ISO-WD'].iloc[4::5]
            y5 = rave_curr["5"]['ISO-WD'].iloc[4::5]
            z5 = rave_curr["6"]['ISO-WK'].iloc[4::5]
        else:
            t_5s_minutes = pd.to_timedelta(raw["1"]['Number'], unit='s')[4::5].dt.total_seconds() / 60

            x5 = rave_curr["1"]['ISO-WD'].iloc[4::5]
            y5 = rave_curr["2"]['ISO-WD'].iloc[4::5]
            z5 = rave_curr["3"]['ISO-WK'].iloc[4::5]
        
        plot_compare_all_metrics(
            t_5s_minutes,
            x5,
            y5,
            z5,
            metrics_dict[str(triax)]["Cx"], metrics_dict[str(triax)]["Cy"], metrics_dict[str(triax)]["Cz"], ride_obj, save=True, save_dir=args.data_path
        )
    
    for triax in range(1, 7):
        raw = non_rave[str(triax)]

        if raw is None:
            continue
        
        if triax == 5:
            t_all = pd.to_timedelta(raw["4"]['Number'], unit='s')
            t_5s_minutes = pd.to_timedelta(raw["4"]['Number'], unit='s')[4::5].dt.total_seconds() / 60
            X_ISO = raw["4"]['ISO-WD']
            Y_ISO = raw["5"]['ISO-WD']
            Z_ISO = raw["6"]['ISO-WB']

        else:
            t_all = pd.to_timedelta(raw["1"]['Number'], unit='s')
            t_5s_minutes = pd.to_timedelta(raw["1"]['Number'], unit='s')[4::5].dt.total_seconds() / 60
            X_ISO = raw["1"]['ISO-WD']
            Y_ISO = raw["2"]['ISO-WD']
            Z_ISO = raw["3"]['ISO-WB']
        a_v = metrics_dict[str(triax)]["av"]
        a_v_5s = metrics_dict[str(triax)]["av_5s"]

        plot_iso_timeseries(t_all, X_ISO, Y_ISO, Z_ISO, a_v, ride_obj, save=True, save_dir=args.data_path)
    

    for triax in range(1, 7):
        raw = non_rave[str(triax)]
        
        if raw is None:
            continue
        
        if triax == 5:
            t_all = pd.to_timedelta(raw["4"]['Number'], unit='s')
            X_VDV = raw["4"]['WD4th_VDV']
            Y_VDV = raw["5"]['WD4th_VDV']
            Z_VDV = raw["6"]['WK4th_VDV']

        else:
            t_all = pd.to_timedelta(raw["1"]['Number'], unit='s')
            # VDV data
            X_VDV = raw["1"]['WD4th_VDV']
            Y_VDV = raw["2"]['WD4th_VDV']
            Z_VDV = raw["3"]['WK4th_VDV']
            
        plot_vdv_over_time(t_all, X_VDV, Y_VDV, Z_VDV, ride_obj, save=True, save_dir=args.data_path)
    
    for triax in range(1, 7):
        raw = non_rave[str(triax)]

        if raw is None:
            continue
        
        if triax == 5:
            t_all = pd.to_timedelta(raw["4"]['Number'], unit='s')
            X_VDV = raw["4"]['WD4th_VDV']
            Y_VDV = raw["5"]['WD4th_VDV']
            Z_VDV = raw["6"]['WK4th_VDV']
            X_ISO = raw["4"]['ISO-WD']
            Y_ISO = raw["5"]['ISO-WD']
            Z_ISO = raw["6"]['ISO-WB']

        else:
            t_all = pd.to_timedelta(raw["1"]['Number'], unit='s')
            # VDV data
            X_VDV = raw["1"]['WD4th_VDV']
            Y_VDV = raw["2"]['WD4th_VDV']
            Z_VDV = raw["3"]['WK4th_VDV']
            X_ISO = raw["1"]['ISO-WD']
            Y_ISO = raw["2"]['ISO-WD']
            Z_ISO = raw["3"]['ISO-WB']

        ratios = [
            compute_vdv_ratios(X_VDV, compute_vdv(X_ISO)),
            compute_vdv_ratios(Y_VDV, compute_vdv(Y_ISO)),
            compute_vdv_ratios(Z_VDV, compute_vdv(Z_ISO))
        ]
        plot_ratio_comparison(t_all, ratios, labels=['X', 'Y', 'Z'], ride_obj=ride_obj, save=True, save_dir=args.data_path)
    
    ride_obj.metrics_dict = metrics_dict
    ride_obj.save_to_json(f"data/{ride_obj.ride_id}_ride_metrics.json")


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