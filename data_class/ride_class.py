import json
import numpy as np
import pandas as pd
import weakref

VALID_DATES = ["5_29_2020", "5_22_2020", "5_30_2020", "5_24_2020"]


def to_json_ready(obj):
    if isinstance(obj, pd.Series):
        return {
            "__type__": "pd.Series",
            "data": obj.tolist(),
            "index": obj.index.tolist(),
            "dtype": str(obj.dtype),
            "name": obj.name,
        }
    if isinstance(obj, pd.DataFrame):
        return {"__type__": "pd.DataFrame", "split": obj.to_dict(orient="split")}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.float64, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_ready(v) for v in obj]
    if isinstance(obj, weakref.ReferenceType):
        return None  

    if hasattr(obj, "__dict__"):
        return to_json_ready(vars(obj))

    return obj

def from_json_ready(obj):
    if isinstance(obj, dict) and obj.get("__type__") == "pd.Series":
        return pd.Series(
            obj["data"],
            index=obj["index"],
            dtype=obj.get("dtype") or None,
            name=obj.get("name"),
        )
    if isinstance(obj, dict) and obj.get("__type__") == "pd.DataFrame":
        return pd.DataFrame(**obj["split"])
    if isinstance(obj, dict):
        return {k: from_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_json_ready(v) for v in obj]
    return obj


triaxial_metrics = {str(num): {"N_MV": None, 
                              "N_VD": None,
                              "N_VA": None,
                              "Cx": None, 
                              "Cy": None, 
                              "Cz": None,
                              "Cx_mean": None,
                              "Cx_std": None,
                              "Cy_mean": None,
                              "Cy_std": None,
                              "Cz_mean": None,
                              "Cz_std": None,
                              "max_Cx": None,
                              "max_Cy": None,
                              "max_Cz": None,
                              "ax": None,
                              "ay": None,
                              "az": None,
                              "av": None, 
                              "av_5s": None,
                              "max_ax": None,
                              "max_ay": None,
                              "max_az": None,
                              "max_av": None,
                              "max_av_5s": None,  
                              }
                              for num in range(1, 7)
                  }

class RideMetaData:
    def __init__(self, ride_id: str, date: str, json_path: str = "data_schema.json"):
        assert date in VALID_DATES, f"Invalid date: {date}. Valid dates are: {VALID_DATES}"

        self.ride_id = ride_id
        self.date = date
        self.load_meta_json(json_path)

    
    def load_meta_json(self, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = data[self.date]
            ride_data = data["runs"].get(self.ride_id, {})
            if not ride_data:
                raise ValueError(f"No data found for ride_id: {self.ride_id} on date: {self.date}")
            
        self.__dict__.update(data["metadata"])
        self.__dict__.update(ride_data)
    
    def __repr__(self):
        return f"RideMetaData(ride_id={self.ride_id}, date={self.date})"
    
    def to_dict(self):
        return dict(self.__dict__)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def save_to_json(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


class Ride(RideMetaData):
    def __init__(self, ride_id: str, date: str, json_path: str = "data_schema.json" ):
        super().__init__(ride_id, date, json_path)
        self.metrics_dict = None
    
    def to_json_dict(self):
        return to_json_ready(self)

    def save_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            raw = json.load(f)
        data = from_json_ready(raw)
        inst = cls.__new__(cls)
        inst.__dict__.update(data)
        return inst
