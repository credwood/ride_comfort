# analysis/loaders.py

import pandas as pd

def load_csv(filepath, skiprows=16):
    return pd.read_csv(filepath, skiprows=skiprows)

def load_rave_and_nonrave(x_file, y_file, z_file):
    # Load data
    x_rave = load_csv(x_file)
    y_rave = load_csv(y_file)
    z_rave = load_csv(z_file)

    # Load 5s running averages
    x_raw = load_csv(x_file.replace('-RAVE', ''))
    y_raw = load_csv(y_file.replace('-RAVE', ''))
    z_raw = load_csv(z_file.replace('-RAVE', ''))

    return (x_rave, y_rave, z_rave), (x_raw, y_raw, z_raw)
