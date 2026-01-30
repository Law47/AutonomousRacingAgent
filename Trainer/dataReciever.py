import os
import pandas as pd
import threading

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(script_dir, "TelemetryData")

xDataColumns = ["Gear", "RPM", "SpeedMPH", "VelocityX", "VelocityY", "VelocityZ", "AccelerationX", "AccelerationY", "AccelerationZ", "FL_Slip", "FR_Slip", "RL_Slip", "RR_Slip", "FL_Load", "FR_Load", "RL_Load", "RR_Load", "WorldPositionX", "WorldPositionY", "WorldPositionZ"]

yDataColumns = ["Gas", "Brake", "Steer"]

# Global storage for latest telemetry
_latest_telemetry = None
_telemetry_lock = threading.Lock()

def get_latest_telemetry():
    """Get the latest received telemetry data as tuple (X features, y targets)"""
    global _latest_telemetry
    with _telemetry_lock:
        if _latest_telemetry is None:
            return None, None
        return _latest_telemetry

def set_latest_telemetry(X, y):
    """Store latest telemetry data"""
    global _latest_telemetry
    with _telemetry_lock:
        _latest_telemetry = (X, y)

def retrieveTrainingData():
    df = pd.DataFrame()

    for item in os.listdir(training_data_path):
        if item[-4:] != ".csv":
            continue

        newDF = pd.read_csv(os.path.join(training_data_path, item))

        df = pd.concat([df, newDF], ignore_index=False)


    yDf = df[yDataColumns]
    xDf = df[xDataColumns]

    yDf["Steer"] /= 320 # Account for telemetry gotten from AC Plugin

    return xDf.values, yDf.values