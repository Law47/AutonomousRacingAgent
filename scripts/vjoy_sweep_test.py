import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Common.controller_vjoy import VJoyController


def hold_step(controller: VJoyController, steer: float, accel: float, brake: float, seconds: float, label: str) -> None:
    print(f"{label}: steer={steer:+.2f} accel={accel:.2f} brake={brake:.2f} for {seconds:.1f}s")
    controller.apply(steer=steer, accel=accel, brake=brake)
    time.sleep(seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual vJoy sweep test for Assetto Corsa input mapping.")
    parser.add_argument("--device-id", type=int, default=1, help="vJoy device id")
    parser.add_argument("--dll-path", type=str, default=None, help="Path to vJoyInterface.dll")
    parser.add_argument("--hold-seconds", type=float, default=1.5, help="Seconds to hold each test input")
    parser.add_argument("--skip-countdown", action="store_true", help="Start immediately")
    args = parser.parse_args()

    try:
        controller = VJoyController(device_id=args.device_id, dll_path=args.dll_path)
    except Exception as exc:
        print(f"Failed to initialize vJoy: {exc}")
        return 1

    try:
        print("Manual vJoy sweep test")
        print("Make sure Assetto Corsa is focused and mapped to the same vJoy device.")
        print("This will send neutral, steering, throttle, brake, and combined inputs.")

        if not args.skip_countdown:
            for countdown in range(3, 0, -1):
                print(f"Starting in {countdown}...")
                time.sleep(1.0)

        hold_step(controller, 0.0, 0.0, 0.0, args.hold_seconds, "Neutral")
        hold_step(controller, -1.0, 0.0, 0.0, args.hold_seconds, "Full left steer")
        hold_step(controller, 1.0, 0.0, 0.0, args.hold_seconds, "Full right steer")
        hold_step(controller, 0.0, 0.0, 0.0, args.hold_seconds, "Center steer")
        hold_step(controller, 0.0, 1.0, 0.0, args.hold_seconds, "Full throttle")
        hold_step(controller, 0.0, 0.0, 0.0, args.hold_seconds, "Throttle release")
        hold_step(controller, 0.0, 0.0, 1.0, args.hold_seconds, "Full brake")
        hold_step(controller, 0.0, 0.0, 0.0, args.hold_seconds, "Brake release")
        hold_step(controller, -0.5, 0.5, 0.0, args.hold_seconds, "Half throttle + left steer")
        hold_step(controller, 0.5, 0.0, 0.5, args.hold_seconds, "Half brake + right steer")
        hold_step(controller, 0.0, 0.0, 0.0, args.hold_seconds, "Neutral")
        print("Sweep complete.")
        return 0
    finally:
        try:
            controller.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
