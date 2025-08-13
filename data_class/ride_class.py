import json

VALID_DATES = ["5_29_2020", "5_22_2020", "5_30_2020", "5_24_2020"]

class RideMetaData:
    def __init__(self, ride_id: str, date: str, json_path: str = "data_schema.json"):
        assert date in VALID_DATES, f"Invalid date: {date}. Valid dates are: {VALID_DATES}"

        self.ride_id = ride_id
        self.date = date
        self.load_json(json_path)

    
    def load_json(self, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = data[self.date]
            ride_data = data.get(self.ride_id, {})
            if not ride_data:
                raise ValueError(f"No data found for ride_id: {self.ride_id} on date: {self.date}")
            
        self.__dict__.update(ride_data)
    
    def __repr__(self):
        return f"RideMetaData(ride_id={self.ride_id}, date={self.date})"
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def save_to_json(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self.to_json(), file, indent=4)


class Ride(RideMetaData):
    def __init__(self, ride_id: str, date: str):
        super().__init__(ride_id, date)
        self.metrics_dict = None
