"""
ACReplayFromRpy.py

Extract telemetry from Assetto Corsa .rpy (replay) files and convert to CSV.

This requires: pip install acreplay-parser

If acreplay-parser is not available, provide instructions to install it.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import csv


def ensure_acreplay_parser():
    """Ensure acreplay-parser is installed."""
    try:
        import acreplay_parser
        return True
    except ImportError:
        print("acreplay-parser not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "acreplay-parser"])
            return True
        except Exception as e:
            print(f"Failed to install acreplay-parser: {e}")
            print("\nManual installation:")
            print("  pip install acreplay-parser")
            print("\nOr clone from: https://github.com/abchouhan/acreplay-parser")
            return False


def convert_rpy_to_csv(rpy_path: str, csv_output: str = None) -> str:
    """
    Convert an Assetto Corsa .rpy replay file to CSV telemetry format.
    
    Args:
        rpy_path: Path to the .rpy file
        csv_output: Optional output CSV path (default: same name with .csv extension)
    
    Returns:
        Path to the created CSV file
    """
    if not os.path.exists(rpy_path):
        print(f"Replay file not found: {rpy_path}")
        return None
    
    if csv_output is None:
        csv_output = Path(rpy_path).with_suffix('.csv').as_posix()
    
    print(f"Converting: {rpy_path}")
    print(f"Output:     {csv_output}")
    
    try:
        # Try using acreplay-parser command line tool
        result = subprocess.run(
            [sys.executable, "-m", "acreplay_parser", rpy_path, "--output", csv_output],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully converted to: {csv_output}")
            return csv_output
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Failed to convert: {e}")
        print("\nAlternative approach - using Python API:")
        print("from acreplay_parser import parse_replay")
        print(f"replay = parse_replay('{rpy_path}')")
        print("# Then extract and write CSV manually")
        return None


def list_replay_files(directory: str = None) -> list:
    """List all .rpy files in a directory (default: current directory)."""
    if directory is None:
        directory = "."
    
    replay_files = list(Path(directory).glob("**/*.rpy")) + list(Path(directory).glob("**/*.acreplay"))
    
    if not replay_files:
        print(f"No replay files found in {directory}")
        return []
    
    print(f"Found {len(replay_files)} replay file(s):")
    for i, f in enumerate(replay_files, 1):
        print(f"  {i}. {f}")
    
    return replay_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Assetto Corsa .rpy replay files to CSV telemetry"
    )
    parser.add_argument('rpy_file', nargs='?', help='Path to .rpy replay file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--batch', '-b', help='Convert all .rpy files in directory')
    parser.add_argument('--list', '-l', action='store_true', help='List all available replays')
    
    args = parser.parse_args()
    
    # Check if acreplay-parser is available
    if not ensure_acreplay_parser():
        print("\nNote: The timing drift issue you were experiencing is because:")
        print("1. AC stores PHYSICAL DATA (positions/velocities) in replays, not inputs")
        print("2. Replaying inputs through vgamepad re-runs physics, which diverges from recorded data")
        print("3. Small timing errors accumulate, causing corner timing issues")
        print("\nUsing acreplay-parser solves this by extracting the actual recorded physical data.")
        return
    
    # List replays
    if args.list:
        replays = list_replay_files()
        if replays and len(replays) > 0:
            print(f"\nTo convert a replay: python {__file__} <path-to-replay>")
        return
    
    # Batch convert directory
    if args.batch:
        replays = list_replay_files(args.batch)
        for rpy_file in replays:
            convert_rpy_to_csv(str(rpy_file))
        return
    
    # Single file conversion
    if args.rpy_file:
        convert_rpy_to_csv(args.rpy_file, args.output)
    else:
        print("No replay file specified.")
        print(f"Usage: python {__file__} <path-to-replay.rpy>")
        print(f"   or: python {__file__} --batch <directory>")
        print(f"   or: python {__file__} --list")


if __name__ == '__main__':
    main()
