import os

from typing import Tuple

import pandas as pd

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing leading/trailing whitespace and converting to lowercase."""
    df.columns = df.columns.str.strip()
    return df

def load_csv(filepath: str, skiprows=16) -> pd.DataFrame:
    df = pd.read_csv(filepath, skiprows=skiprows, index_col=False)
    return clean_names(df)

def load_rave_and_nonrave(folder_path: str) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]  :
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

def load_all_triax(dir_path: str, flip_x_y: bool=False) -> Tuple[dict, dict]:
    """
    1. map sensors to triax
    2. identiify direction of each of the three files mapped to a triax
    3. identify RAVE and raw versions of each file
    4. Load each file into a dict (RAVE or raw) with triax 
    """
    triaxes_files = {str(num): {"1": [], "2": [], "3": []} for num in range(1, 7)}
    triaxes_files["5"] = {"4": [], "5": [], "6": []}  # TEAC has 5th triax
    triax_sensor_map = {"rion_1": "1", "rion_2": "2", "rion_3": "4", "rion_4": "6", "teac": "5"}
    all_files = os.listdir(dir_path)

    for file in all_files:
        for sensor, triax in triax_sensor_map.items():
            if sensor in file.lower():
                if sensor == 'teac':
                    if int(file[-5]) > 6:
                        continue
                    if int(file[-5]) > 3:
                        triaxes_files["5"][file[-5]].append(file)
                    else:
                        triaxes_files["3"][file[-5]].append(file)
                else:
                    if int(file[-5]) > 3:
                        continue
                    triaxes_files[triax][file[-5]].append(file)

    triax_data = {str(num): {"1": [], "2": [], "3": []} for num in range(1, 7)}
    RAVE_triax_data = {str(num): {"1": [], "2": [], "3": []} for num in range(1, 7)}
    triax_data["5"] = {"4": [], "5": [], "6": []} 
    RAVE_triax_data["5"] = {"4": [], "5": [], "6": []}

    for triax, dim_files in triaxes_files.items():
        for dim, files in dim_files.items():
            for file in files:
                data_path = os.path.join(dir_path, file)
                if "RAVE" in file.upper():
                    RAVE_triax_data[triax][dim] = load_csv(data_path)
                else:
                    triax_data[triax][dim] = load_csv(data_path)
    
    for triax in triax_data:
        if triax == "5":
            if len(triax_data[triax]["4"]) == 0:
                triax_data[triax]["4"] = None
                RAVE_triax_data[triax]["4"] = None
        else:
            if len(triax_data[triax]["1"]) == 0:
                triax_data[triax] = None
                RAVE_triax_data[triax] = None
    
    if flip_x_y:
        for triax in triax_data:
            if triax == "5":
                if triax_data[triax] is not None:
                    triax_data[triax]["4"], triax_data[triax]["5"] = triax_data[triax]["5"], triax_data[triax]["4"]
                    RAVE_triax_data[triax]["4"], RAVE_triax_data[triax]["5"] = RAVE_triax_data[triax]["5"], RAVE_triax_data[triax]["4"]
            else:
                if triax_data[triax] is not None:
                    triax_data[triax]["1"], triax_data[triax]["2"] = triax_data[triax]["2"], triax_data[triax]["1"]
                    RAVE_triax_data[triax]["1"], RAVE_triax_data[triax]["2"] = RAVE_triax_data[triax]["2"], RAVE_triax_data[triax]["1"]

   
    return triax_data, RAVE_triax_data
