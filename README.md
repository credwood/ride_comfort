# Ride Comfort Analysis

This project provides a modular Python implementation for computing and visualizing ride comfort metrics from vibration data. It should mirror the logic and structure of the MATLAB script. Note that this code is untested and is not meant to be executed. If you put data RAVE and non-RAVE data, it can be tested.

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

## What `main.py` Does

1. **Loads Data**
   - Loads three-axis (X, Y, Z) datasets in two forms:
     - Raw filtered time series
     - Rolling RMS versions (RAVE)

2. **Processes Signals**
   - Extracts relevant acceleration columns
   - Downsamples RAVE data to every 5th row

3. **Computes Metrics**
   - `N_MV`: based on 95th percentiles of RMS signals
   - `N_VD`: based on 50th and 95th percentiles
   - Composite 3D signal (`a_v`) from 1s data
   - VDV
   - Ratios between VDV values

4. **Categorizes Comfort**
   - Maps metric values to comfort categories via threshold bands

5. **Visualizes**
   - 5 different plots corresponding to the calculated metrics, their ratios and overall comparisons

---

## Module Overview

### `analysis/loaders.py`

- `load_rave_and_nonrave(x_path, y_path, z_path)`:  
  Loads both RAVE and raw time series data for each axis.

---

### `analysis/metrics.py`

- `compute_nmv(ax95, ay95, az95)`:  
  Calculates a scalar comfort metric from 95th percentile inputs.

- `compute_nvd(ax50, ay50, az50, ay95)`:  
  Computes an additional metric based on 50th/95th percentile values.

- `compute_av(x, y, z)`:  
  Combines three axes into a 3D root-sum-square acceleration signal.

- `compute_vdv(signal)`:  
  Computes VDV over time from a signal.

- `compute_vdv_ratios(vdv, rms)`:  
  Computes the ratio between VDV values.

---

### `analysis/plotting.py`

- `plot_comfort_thresholds(nmv, nvd, categories)`:  
  Visualizes scalar metrics against comfort bands.

- `plot_vdv_over_time(t, x, y, z)`:  
  Plots VDV time series for each axis.

- `plot_ratio_comparison(t, ratios, labels)`:  
  Plots the VDV ratios against the standard threshold.

- `plot_comfort_timeseries(t_5s, Cx, Cy, Cz)`:  
  Plots rolling RMS signals over time with shaded threshold bands.

- `plot_iso_timeseries(t_all, X_ISO, Y_ISO, Z_ISO, a_v, a_v_5s, t_5s)`:  
  Plots 1s acceleration signals, composite acceleration, and overlays the 5s composite for comparison.

---

## Notes

- CSV data files must be in the `data/` directory.
- Sampling assumptions:
  - Raw files: 1s interval
  - RAVE files: 5s rolling average, sampled every 1s
- The RAVE data is optionally downsampled every 5 samples to avoid overlap in metric calculations.

---

## Outputs

The script (in theory) produces:
- Console prints of calculated metrics
- A series of Matplotlib figures showing:
  - Comfort metric thresholds
  - Time series of acceleration and composite signals
  - VDV and ratio plots
---
