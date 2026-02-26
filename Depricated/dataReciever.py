import os
import pandas as pd
import threading

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(script_dir, "TelemetryData")

xDataColumns = ["Gear", "RPM", "SpeedMPH", "VelocityX", "VelocityY", "VelocityZ", "AccelerationX", "AccelerationY", "AccelerationZ", "FL_Slip", "FR_Slip", "RL_Slip", "RR_Slip", "FL_Load", "FR_Load", "RL_Load", "RR_Load", "WorldPositionX", "WorldPositionY", "WorldPositionZ"]

yDataColumns = ["Gas", "Brake", "Steer"]

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