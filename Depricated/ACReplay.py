"""
ACReplay.py

Replay gas, brake and steering from a shared-memory CSV through a virtual Xbox controller.

Reads `steerAngle_deg` (preferred) or `steerAngle_rad`, `gas`, and `brake` from a CSV (default: Trainer/TelemetryData/ACShared_mem_test.csv),
maps steering using a symmetric steering lock (default 50° for GT3), and sends inputs via vgamepad.
Press SPACE to start replay, ` to cancel, Ctrl+C to stop during replay.
"""

import time
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import vgamepad as vg
import keyboard


DEFAULT_FILE = os.path.join("Trainer", "TelemetryData", "ACShared_mem_test.csv")


def create_gamepad():
    gp = vg.VX360Gamepad()
    try:
        gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gp.right_trigger_float(value_float=0.0)
        gp.left_trigger_float(value_float=0.0)
        gp.update()
    except Exception:
        try:
            gp.reset()
            gp.update()
        except Exception:
            pass
    return gp


def wait_for_start():
    print("\n" + "="*40)
    print("Press SPACE to start replay")
    print("Press ` to cancel")
    print("="*40 + "\n")
    while True:
        try:
            if keyboard.is_pressed('space'):
                print("Starting replay...\n")
                time.sleep(0.2)
                return True
            if keyboard.is_pressed('`'):
                print("Replay cancelled by user")
                return False
            time.sleep(0.05)
        except Exception:
            # keyboard may require elevated privileges on some platforms; fall back to immediate start
            print("Keyboard polling unavailable, starting immediately")
            return True


def map_steering_rad_to_xbox(steer_rad: float) -> float:
    """Map steering in radians to Xbox -1..1
    AC shared memory steerAngle_rad has a 1:1 mapping with the Xbox axis input.
    """
    return float(np.clip(steer_rad, -1.0, 1.0))


def detect_recording_rate(df):
    """
    Detect the actual recording rate from CSV timestamps.
    Handles dropped frames and timing variance.
    """
    if 'timestamp' not in df.columns:
        return None
    
    try:
        timestamps = pd.to_datetime(df['timestamp']).values
        time_diffs = np.diff(timestamps).astype('float64') / 1e9  # convert to seconds
        
        # Use median to ignore outliers from dropped frames
        median_frame_time = np.median(time_diffs)
        detected_hz = 1.0 / median_frame_time if median_frame_time > 0 else None
        
        return detected_hz
    except Exception as e:
        print(f"Could not detect recording rate: {e}")
        return None


def run_replay(csv_path: str, speed: float = 1.0, start: int = 0, end: int = None, rate_fallback: float = 333.0, throttle_test: bool = False, use_raw_input: bool = False):
    if not os.path.exists(csv_path):
        print(f"Telemetry file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded telemetry file: {csv_path}")
    print(f"Total frames: {len(df)}")
    
    # Detect actual recording rate from timestamps
    detected_rate = detect_recording_rate(df)
    if detected_rate is not None:
        print(f"Detected recording rate: {detected_rate:.2f} Hz (using this instead of {rate_fallback})")
        rate_fallback = detected_rate
    else:
        print(f"Using fallback rate: {rate_fallback} Hz")

    # Detect columns - use physics telemetry values OR raw controller input
    if use_raw_input:
        print("Using raw controller input from CSV...")
        steer_col = next((c for c in df.columns if c.lower() == 'controller_steering'), None)
        if steer_col is None:
            print("Error: 'controller_steering' column not found in CSV")
            return

        gas_col = next((c for c in df.columns if c.lower() == 'controller_gas'), None)
        if gas_col is None:
            print("Error: 'controller_gas' column not found in CSV")
            return

        brake_col = next((c for c in df.columns if c.lower() == 'controller_brake'), None)
        if brake_col is None:
            print("Error: 'controller_brake' column not found in CSV")
            return
    else:
        print("Using physics telemetry values from CSV...")
        steer_col = next((c for c in df.columns if c.lower() == 'steerangle_rad'), None)
        if steer_col is None:
            steer_col = next((c for c in df.columns if c.lower() == 'steerangle_deg'), None)
        if steer_col is None:
            print("Error: 'steerAngle_rad' or 'steerAngle_deg' column not found in CSV")
            return

        gas_col = next((c for c in df.columns if c.lower() == 'gas'), None)
        if gas_col is None:
            print("Error: 'gas' column not found in CSV")
            return

        brake_col = next((c for c in df.columns if c.lower() == 'brake'), None)
        if brake_col is None:
            print("Error: 'brake' column not found in CSV")
            return

    # Prepare gamepad
    gp = create_gamepad()

    # Prepare limits
    # No longer needed: AC steerAngle_rad maps 1:1 to Xbox axis input

    # Indexing
    if end is None:
        end = len(df)

    # Wait for user to start
    if not wait_for_start():
        return

    try:
        # Controls available during replay
        print("Controls: press 'p' to pause/resume, 'r' to restart, ` to cancel")
        print(f"Replay rate: {rate_fallback} Hz (frame time: {1000.0/rate_fallback:.3f}ms)")
        paused = False
        idx = start
        throttle_maxed = False  # Track if throttle has hit 1.0
        frame_time = 1.0 / rate_fallback  # Time per frame at target rate (e.g., 1/333 = 0.003003s)
        
        # Initialize timing with adaptive drift compensation
        next_time = time.perf_counter() + frame_time
        drift_samples = []  # Track drift to apply compensation
        
        while idx < end:
            row = df.iloc[idx]
            # Read telemetry values from CSV
            steer_val = float(row[steer_col])
            gas_val = float(row[gas_col])
            brake_val = float(row[brake_col])
            
            # Throttle test mode: once gas hits 1.0, keep it at 1.0
            if throttle_test:
                if gas_val >= 0.99:  # Use 0.99 to catch near-max values
                    throttle_maxed = True
                if throttle_maxed:
                    gas_val = 1.0

            # Check for pause/restart/cancel keys (keyboard may require privileges)
            try:
                if keyboard.is_pressed('`'):
                    print("Replay cancelled by user (`)")
                    break
                if keyboard.is_pressed('r'):
                    print("Restarting replay from start frame")
                    idx = start
                    next_time = time.perf_counter() + frame_time
                    time.sleep(0.25)  # debounce
                    continue
                if keyboard.is_pressed('p'):
                    paused = not paused
                    if paused:
                        print("Replay paused. Press 'p' to resume or 'r' to restart.")
                        try:
                            gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                            gp.right_trigger_float(value_float=0.0)
                            gp.left_trigger_float(value_float=0.0)
                            gp.update()
                        except Exception:
                            pass
                    else:
                        print("Resuming replay...")
                    time.sleep(0.25)  # debounce
            except Exception:
                # If keyboard polling unavailable, continue without pause features
                pass

            # If paused, wait in a small loop until unpaused/restarted/cancelled
            while paused:
                try:
                    if keyboard.is_pressed('`'):
                        print("Replay cancelled by user (`)")
                        paused = False
                        idx = end
                        break
                    if keyboard.is_pressed('r'):
                        print("Restarting replay from start frame")
                        paused = False
                        idx = start
                        next_time = time.perf_counter() + frame_time
                        time.sleep(0.25)
                        break
                    if keyboard.is_pressed('p'):
                        paused = False
                        print("Resuming replay...")
                        time.sleep(0.25)
                        break
                except Exception:
                    pass
                time.sleep(0.05)

            if idx >= end:
                break

            # Map steering (steerAngle_rad maps 1:1 to Xbox axis)
            xbox_steer = map_steering_rad_to_xbox(steer_val)
            xbox_gas = float(np.clip(gas_val, 0.0, 1.0))
            xbox_brake = float(np.clip(brake_val, 0.0, 1.0))

            # Wait until the scheduled frame time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Send to virtual controller immediately after waking
            gp.left_joystick_float(x_value_float=xbox_steer, y_value_float=0.0)
            gp.right_trigger_float(value_float=xbox_gas)
            gp.left_trigger_float(value_float=xbox_brake)
            gp.update()

            # Schedule next frame after sending
            next_time += frame_time / max(1.0, speed)

            if (idx - start + 1) % 200 == 0:
                actual_time = time.perf_counter()
                drift = (actual_time - next_time) * 1000  # milliseconds
                drift_samples.append(drift)
                
                # Apply adaptive compensation every 1000 frames
                if len(drift_samples) >= 5:  # 1000 frames / 200 per print = 5 samples
                    avg_drift = np.mean(drift_samples)
                    # If consistently drifting ahead (negative), slow down next_time slightly
                    if avg_drift < -1.0:
                        # Add compensation: we're running ~3ms fast
                        next_time += abs(avg_drift) * 0.0005  # Gradually apply 50% of avg drift
                    drift_samples = []
                
                print(f"Frame {idx+1}/{end} - steer: {steer_val:.4f}, gas: {xbox_gas:.4f}, brake: {xbox_brake:.4f} | drift: {drift:+.1f}ms")

            idx += 1

    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    finally:
        try:
            gp.reset()
            gp.update()
        except Exception:
            pass
        print("Controller reset to neutral state")


def main():
    parser = argparse.ArgumentParser(description="Replay AC shared-memory CSV to virtual Xbox controller")
    parser.add_argument('--file', '-f', default=DEFAULT_FILE, help='Telemetry CSV path')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed multiplier')
    parser.add_argument('--rate', type=float, default=333.0, help='Playback rate in Hz (default 333, matches ACRead.py recording rate)')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index')
    parser.add_argument('--throttle-test', action='store_true', help='Test mode: once throttle hits 1.0, keep it at 1.0 for entire replay')
    parser.add_argument('--use-raw-input', action='store_true', help='Use raw controller input from CSV instead of physics telemetry')
    args = parser.parse_args()

    run_replay(args.file, speed=args.speed, start=args.start, end=args.end, rate_fallback=args.rate, throttle_test=args.throttle_test, use_raw_input=args.use_raw_input)


if __name__ == '__main__':
    main()
