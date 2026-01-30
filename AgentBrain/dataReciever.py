import os
import pandas as pd
from io import StringIO
import socket
import threading

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(script_dir, "TelemetryData")

xDataColumnsToDrop = ["PacketID", "SessionID", "LapID", "PacketDatetime", "Gas", "Brake", "Steer", "Clutch", "Handbrake", "Gear", "Fuel", "Weight", "ResetCount", "CollidedWith", "HeadlightsActive", "Ping", "SteerTorque", "FL_Camber", "FR_Camber", "RL_Camber", "RR_Camber", "FL_ToeIn", "FR_ToeIn", "RL_ToeIn", "RR_ToeIn", "FL_TyreRadius", "FR_TyreRadius", "RL_TyreRadius", "RR_TyreRadius", "FL_TyreWidth", "FR_TyreWidth", "RL_TyreWidth", "RR_TyreWidth", "FL_RimRadius", "FR_RimRadius", "RL_RimRadius", "RR_RimRadius"]

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
    xDf = df.drop(columns=xDataColumnsToDrop)

    return xDf.values, yDf.values

def handle_client(conn, address):
    print(f"Client connected: {address}")
    try:
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                try:
                    text = data.decode(errors='replace')
                except Exception:
                    text = str(data)
                
                df = pd.read_csv(StringIO(text))
                yDf = df[yDataColumns]
                xDf = df.drop(columns=xDataColumnsToDrop)
                
                # Store latest telemetry
                set_latest_telemetry(xDf.values, yDf.values)
                print(f"Got Packet from {address}")

    except Exception as e:
        print(f"Connection handler error for {address}: {e}")
    finally:
        print(f"Client disconnected: {address}")

def server_program(host='0.0.0.0', port=8000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"Server listening on {host}:{port}")
    try:
        while True:
            conn, address = server_socket.accept()
            t = threading.Thread(target=handle_client, args=(conn, address), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("Server shutting down")
    finally:
        server_socket.close()


if __name__ == '__main__':
    server_program()