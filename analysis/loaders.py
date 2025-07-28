import os

import pandas as pd

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing leading/trailing whitespace and converting to lowercase."""
    df.columns = df.columns.str.strip()
    return df

def load_csv(filepath, skiprows=16):
    df = pd.read_csv(filepath, skiprows=skiprows, index_col=False)
    return clean_names(df)

def load_rave_and_nonrave(folder_path: str):
    all_files = sorted(os.listdir(folder_path))

    rave_files = [f for f in all_files if "-RAVE" in f.upper()]
    raw_files = [f for f in all_files if "-RAVE" not in f.upper() and f.lower().endswith(".csv")]
    print(f"RAVE files: {rave_files}")
    print(f"Raw files: {raw_files}")

    if len(rave_files) < 3 or len(raw_files) < 3:
        raise ValueError("Expected at least 3 RAVE and 3 raw CSV files.")

    x_rave = load_csv(os.path.join(folder_path, rave_files[0]))
    y_rave = load_csv(os.path.join(folder_path, rave_files[1]))
    z_rave = load_csv(os.path.join(folder_path, rave_files[2]))

    x_raw = load_csv(os.path.join(folder_path, raw_files[0]))
    y_raw = load_csv(os.path.join(folder_path, raw_files[1]))
    z_raw = load_csv(os.path.join(folder_path, raw_files[2]))

    return (x_rave, y_rave, z_rave), (x_raw, y_raw, z_raw)
