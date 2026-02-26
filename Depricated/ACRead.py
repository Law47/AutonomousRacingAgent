"""
ACRead.py

Read Assetto Corsa shared memory and record telemetry to CSV at a target sample rate (default 333 Hz).

Based on ACReadTest.py (reads shared memory tag `acpmf_physics`).
"""

import ctypes
import mmap
import time
import csv
import os
import argparse
import datetime
import pygame


# -----------------------------
# Assetto Corsa Physics Struct
# -----------------------------
class ACPhysics(ctypes.Structure):
    _fields_ = [
        ("packetId", ctypes.c_int),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int),
        ("rpms", ctypes.c_float),
        ("steerAngle", ctypes.c_float),
        ("speedKmh", ctypes.c_float),
        ("velocity", ctypes.c_float * 3),
        ("accG", ctypes.c_float * 3),
        ("wheelSlip", ctypes.c_float * 4),
        ("wheelLoad", ctypes.c_float * 4),
    ]


SHM_NAME = "acpmf_physics"


def default_output_path(output_dir: str = "Trainer/TelemetryData") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"ACSharedMemory_{ts}.csv"
    return os.path.join(output_dir, fname)


def write_header(writer):
    headers = [
        "packetId",
        "timestamp",
        "gas",
        "brake",
        "fuel",
        "gear",
        "rpms",
        "steerAngle_rad",
        "steerAngle_deg",
        "speedKmh",
        "velX",
        "velY",
        "velZ",
        "accGX",
        "accGY",
        "accGZ",
        "wheelSlip0",
        "wheelSlip1",
        "wheelSlip2",
        "wheelSlip3",
        "wheelLoad0",
        "wheelLoad1",
        "wheelLoad2",
        "wheelLoad3",
        "controller_gas",
        "controller_brake",
        "controller_steering",
    ]
    writer.writerow(headers)


def read_and_record(shm_name: str, out_path: str, rate_hz: float):
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Calculate sampling period
    period = 1.0 / float(rate_hz)

    # Initialize pygame for controller input
    pygame.init()
    pygame.joystick.init()
    
    joystick = None
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Connected to controller: {joystick.get_name()}")
    else:
        print("WARNING: No controller detected. Controller input will be recorded as 0.0")

    # Open shared memory
    try:
        shm_size = ctypes.sizeof(ACPhysics)
        mm = mmap.mmap(-1, shm_size, tagname=shm_name, access=mmap.ACCESS_READ)
    except FileNotFoundError:
        print(f"ERROR: Shared memory '{shm_name}' not available. Is Assetto Corsa running?")
        pygame.quit()
        return

    print(f"Connected to shared memory '{shm_name}' (reading {rate_hz:.1f} Hz). Output: {out_path}")

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        write_header(writer)
        csvfile.flush()

        next_time = time.perf_counter()
        frame_count = 0
        try:
            while True:
                # Read controller inputs FIRST (synchronize with physics read)
                if joystick:
                    controller_gas = joystick.get_axis(5)  # Right trigger (axis 5)
                    controller_brake = joystick.get_axis(4)  # Left trigger (axis 4)
                    controller_steering = joystick.get_axis(0)  # Left stick X (axis 0)
                else:
                    controller_gas = 0.0
                    controller_brake = 0.0
                    controller_steering = 0.0
                
                # Read raw shared memory into structure (synchronized with controller)
                phys = ACPhysics.from_buffer_copy(mm)

                # Timestamp AFTER both reads to represent this frame's time
                ts = datetime.datetime.now().isoformat()
                steer_rad = float(phys.steerAngle)
                steer_deg = steer_rad * (180.0 / 3.141592653589793)

                row = [
                    int(phys.packetId),
                    ts,
                    float(phys.gas),
                    float(phys.brake),
                    float(phys.fuel),
                    int(phys.gear),
                    float(phys.rpms),
                    steer_rad,
                    steer_deg,
                    float(phys.speedKmh),
                    float(phys.velocity[0]),
                    float(phys.velocity[1]),
                    float(phys.velocity[2]),
                    float(phys.accG[0]),
                    float(phys.accG[1]),
                    float(phys.accG[2]),
                    float(phys.wheelSlip[0]),
                    float(phys.wheelSlip[1]),
                    float(phys.wheelSlip[2]),
                    float(phys.wheelSlip[3]),
                    float(phys.wheelLoad[0]),
                    float(phys.wheelLoad[1]),
                    float(phys.wheelLoad[2]),
                    float(phys.wheelLoad[3]),
                    controller_gas,
                    controller_brake,
                    controller_steering,
                ]

                writer.writerow(row)
                csvfile.flush()
                frame_count += 1

                # Sleep until next sample time (333Hz = 1/333 = 0.003003 seconds)
                next_time += period
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # If we're late, DON'T reset next_time - just skip this frame and stay on schedule
                # This maintains consistent frame-time grid even with occasional missed samples

        except KeyboardInterrupt:
            print(f"\nRecording stopped by user. Recorded {frame_count} frames")
        finally:
            try:
                mm.close()
            except Exception:
                pass
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Record Assetto Corsa shared memory telemetry to CSV")
    parser.add_argument("--out", "-o", help="Output CSV path (default: Trainer/TelemetryData/ACSharedMemory_<ts>.csv)")
    parser.add_argument("--rate", "-r", type=float, default=333.0, help="Sample rate in Hz (default: 333)")
    parser.add_argument("--shm", default=SHM_NAME, help="Shared memory tag name (default: acpmf_physics)")

    args = parser.parse_args()

    out_path = args.out if args.out else default_output_path()

    read_and_record(args.shm, out_path, args.rate)


if __name__ == "__main__":
    main()
