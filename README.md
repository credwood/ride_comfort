# Ride Comfort Analysis

This project provides a modular Python implementation for computing and visualizing ride comfort metrics from vibration data. It should mirror the logic and structure of the MATLAB script.

---

## Setup

Clone repository

```
git clone https://github.com/credwood/ride_comfort.git
```

Create conda environment

```
conda create -n comfort python=3.11.13
```

After the conda environment is created (you will be prompted to type Y to continue installations), run:

```
conda activate comfort
```

Navigate to root folder of project, where `requirements.txt` lives and run:

```
pip install -r requirements.txt
```

## Overview

---

## What `calculations.ipynb` Does

1. **Loads Data**
   - Loads three-axis (X, Y, Z) datasets in two forms for all 6 sensors:
     - Raw filtered time series
     - Rolling RMS versions (RAVE)

2. **Processes Signals**
   - Extracts relevant acceleration columns
   - Downsamples RAVE data to every 5th row

3. **Computes Metrics**
   - `N_MV`: based on 95th percentiles of RMS signals
   - `N_VD`: based on 50th and 95th percentiles
   - `N_VA`: uses floor and seat sensors to calculate comfort level
   - Composite 3D signal (`a_v`) from 1s data
   - Composite 3D signal (`a_v_5s`) from 5s data
   - `VDV`: Vibration dose value; measures exposure to shocks, jolts, and intermittent vibration.
   - Ratios between VDV values

4. **Categorizes Comfort**
   - Maps metric values to comfort categories via threshold bands

5. **Visualizes**
   - 9 different plots corresponding to the calculated metrics and statistics, their ratios and overall comparisons

---

## Module Overview

### `analysis/loaders.py`

- `load_all_triax(path_to_data_dir, flip_x_y=False)`:  
  Loads both RAVE and raw time series data for each triax. If the flip variable is set to `True`, the `X` and `Y` components are swapped.

---

### `analysis/metrics.py`

- `compute_nmv(ax95, ay95, az95)`:  
  Calculates a scalar comfort metric from 95th percentile inputs.

- `compute_nvd(ax50, ay50, az50, ay95)`:  
  Computes an additional metric based on 50th/95th percentile values.

- `compute_nva(Azp_95, Aya_95, Aza_95, Axd_95)`
  Computes a composite metric assessing comfort using both floor and sensor data.

- `compute_av(x, y, z)`:  
  Combines three axes into a 3D root-sum-square acceleration signal.

- `compute_vdv(signal)`:  
  Computes VDV over time from a signal.

- `compute_vdv_ratios(vdv, rms)`:  
  Computes the ratio between VDV values.

---

### `analysis/plotting.py`

- `plot_comfort_thresholds_nvm(nmvs, categories, floor_triaxials)`:  
  Visualizes continuous comfort (standard method) metric against comfort bands.

- `plot_comfort_thresholds_nvd_nva(nvds, nvaz, categories, pair_dict)`
  Visualizes continuous comfort (complete method) metrics against comfort bands.

- `plot_vdv_over_time(t, x, y, z)`:  
  Plots VDV time series for each axis.

- `plot_ratio_comparison(t, ratios, labels)`:  
  Plots the VDV ratios against the standard threshold.

- `plot_comfort_timeseries(t_5s, Cx, Cy, Cz)`:  
  Plots rolling RMS signals over time with shaded threshold bands.

- `plot_iso_timeseries(t_all, X_ISO, Y_ISO, Z_ISO, a_v, a_v_5s, t_5s)`:  
  Plots 1s acceleration signals, composite acceleration, and overlays the 5s composite for comparison.

- `plot_compare_all_metrics(t_5s_minutes, X_ISO_5s, Y_ISO_5s, Z_ISO_5s, C_cx, C_cy, C_cz)`
  Plots calculations made according to both ISO and EN standards.

- `plot_distributions(C_cx, C_cy, C_cz, triax)`
  Plots histograms (20 bins) of the continuous comfort time series for each triax's dimension.

- `plot_cumulative_distribution(C_cx, C_cy, C_cz, triax)`
  Plots cumulative distributions of the continuous comfort time series for each triax's dimension.
---

## Notes

- CSV data files must be in the `data/` directory.
- Sampling assumptions:
  - Raw files: 1s interval
  - RAVE files: 5s rolling average, sampled every 1s
- The RAVE data is optionally downsampled every 5 samples to avoid overlap in metric calculations.

---
