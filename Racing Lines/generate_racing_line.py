"""
Script to convert Assetto Corsa .ai (racing line) files to CSV format.
Based on the parsing logic from assetto_corsa_gym.

Usage:
    python generate_racing_line.py <path_to_ai_file> [output_csv_path]

Example:
    python generate_racing_line.py "C:/AC/content/tracks/monza/ai/fast_lane.ai" monza_racing_line.csv
"""

import struct
import pandas as pd
import numpy as np
import argparse
import os
from operator import itemgetter
import math


def parse_ai_file(ai_file_path):
    """
    Parse binary .ai file from Assetto Corsa and extract racing line data.
    
    Args:
        ai_file_path: Path to the .ai file (e.g., content/tracks/monza/ai/fast_lane.ai)
        
    Returns:
        dict containing:
            - fast_lane: list of (x, y) coordinates for the racing line
            - left_lane: left track boundary coordinates
            - right_lane: right track boundary coordinates
            - throttle_arr: throttle values at each point
            - brake_arr: brake values at each point
            - speed_arr: recommended speed at each point
            - angle_arr: track angle/heading at each point
    """
    
    if not os.path.isfile(ai_file_path):
        raise FileNotFoundError(f"AI file not found: {ai_file_path}")
    
    track_data = {}
    left_array = []
    right_array = []
    brake_array = []
    throttle_array = []
    speed_array = []
    angle_array = []
    
    print(f"Parsing AI file: {ai_file_path}")
    
    with open(ai_file_path, "rb") as buffer:
        buffer.seek(0)
        
        # Read header
        header, detailCount, u1, u2 = struct.unpack("4i", buffer.read(4 * 4))
        print(f"Header: {header}, Detail Count: {detailCount}")
        
        # Read ideal racing line data (4 floats + 1 int per point)
        # Format: x, y, z, dist, id
        data_ideal = []
        for i in range(detailCount):
            data_ideal.append(struct.unpack("4f i", buffer.read(4 * 5)))
        
        # Read detailed track info (18 floats per point)
        # Contains: throttle, brake, speed, direction, left border, right border, etc.
        data_detail = []
        for i in range(detailCount):
            data_detail.append(struct.unpack("18f", buffer.read(4 * 18)))
        
        # Process the data
        dir_real = 0
        for i in range(detailCount):
            x, y, z, dist, id = data_ideal[i]
            throttle, brake = itemgetter(2, 3)(data_detail[i])
            speed = itemgetter(1)(data_detail[i])
            dir_angle, right, left = itemgetter(4, 6, 7)(data_detail[i])
            dir_real = dir_real + dir_angle
            
            # Calculate angle based on consecutive points
            index_n = i - 1 if i > 0 else detailCount - 1
            angle = math.degrees(math.atan2(data_ideal[index_n][2] - z, x - data_ideal[index_n][0])) * -1
            
            # Calculate left and right border points based on angle and distances
            lx = x + math.cos((-angle - 90) * math.pi / 180) * left
            lz = z - math.sin((-angle - 90) * math.pi / 180) * left
            
            rx = x + math.cos((-angle + 90) * math.pi / 180) * right
            rz = z - math.sin((-angle + 90) * math.pi / 180) * right
            
            # Store data (swap coordinates for consistency)
            left_array.append((rz, rx))
            right_array.append((lz, lx))
            throttle_array.append(throttle)
            brake_array.append(brake)
            speed_array.append(max(15, speed))  # Minimum speed of 15 m/s
            angle_array.append(angle)
        
        # Racing line is extracted from ideal line data
        track_data['fast_lane'] = [(el[2], el[0]) for el in data_ideal]
        track_data['left_lane'] = left_array
        track_data['right_lane'] = right_array
        track_data['throttle_arr'] = throttle_array
        track_data['brake_arr'] = brake_array
        track_data['speed_arr'] = speed_array
        track_data['angle_array'] = angle_array
    
    print(f"Successfully parsed {detailCount} waypoints from AI file")
    return track_data


def save_racing_line_to_csv(track_data, output_path):
    """
    Save racing line data to CSV file.
    
    Args:
        track_data: dict containing parsed track data
        output_path: path where to save the CSV file
    """
    
    fast_lane = track_data['fast_lane']
    
    # Extract x and y coordinates from fast_lane
    pos_x = [point[0] for point in fast_lane]
    pos_y = [point[1] for point in fast_lane]
    
    # Create DataFrame
    df = pd.DataFrame({
        'pos_x': pos_x,
        'pos_y': pos_y,
        'left_border_x': [point[0] for point in track_data['left_lane']],
        'left_border_y': [point[1] for point in track_data['left_lane']],
        'right_border_x': [point[0] for point in track_data['right_lane']],
        'right_border_y': [point[1] for point in track_data['right_lane']],
        'throttle': track_data['throttle_arr'],
        'brake': track_data['brake_arr'],
        'target_speed': track_data['speed_arr'],
        'yaw': track_data['angle_array']
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Racing line saved to: {output_path}")
    print(f"CSV contains {len(df)} waypoints with columns: {list(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert Assetto Corsa .ai racing line file to CSV format"
    )
    parser.add_argument(
        "ai_file",
        help="Path to the .ai file (e.g., content/tracks/monza/ai/fast_lane.ai)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: same name as input with .csv extension)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        output_path = os.path.splitext(args.ai_file)[0] + "_racing_line.csv"
    else:
        output_path = args.output
    
    try:
        # Parse the AI file
        track_data = parse_ai_file(args.ai_file)
        
        # Save to CSV
        df = save_racing_line_to_csv(track_data, output_path)
        
        print("\n✓ Conversion completed successfully!")
        print(f"Output file: {os.path.abspath(output_path)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
