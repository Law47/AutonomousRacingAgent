import os
import pandas as pd

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(script_dir, "TelemetryData")

xDataColumnsToDrop = ["PacketID", "SessionID", "LapID", "PacketDatetime", "Gas", "Brake", "Steer", "Clutch", "Handbrake", "Gear", "Fuel", "Weight", "ResetCount", "CollidedWith", "HeadlightsActive", "Ping", "SteerTorque", "FL_Camber", "FR_Camber", "RL_Camber", "RR_Camber", "FL_ToeIn", "FR_ToeIn", "RL_ToeIn", "RR_ToeIn", "FL_TyreRadius", "FR_TyreRadius", "RL_TyreRadius", "RR_TyreRadius", "FL_TyreWidth", "FR_TyreWidth", "RL_TyreWidth", "RR_TyreWidth", "FL_RimRadius", "FR_RimRadius", "RL_RimRadius", "RR_RimRadius"]

yDataColumns = ["Gas", "Brake", "Steer"]

def retrieveTrainingData():
    df = pd.DataFrame()

    for item in os.listdir(training_data_path):
        if item[-4:] != ".csv":
            continue

        newDF = pd.read_csv(os.path.join(training_data_path, item))

        df = pd.concat([df, newDF], ignore_index=False)


    yDf = df[yDataColumns]
    xDf = df.drop(columns=xDataColumnsToDrop)

    return xDf.values, yDf.values