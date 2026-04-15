import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETTO_GYM_DIR = ROOT_DIR / "assetto_corsa_gym"

sys.path.insert(0, str(ASSETTO_GYM_DIR))

from AssettoCorsaPlugin.plugins.sensors_par.car_control import Controls


def parse_args():
    parser = argparse.ArgumentParser(description="Repeatedly press shift-up and shift-down controller buttons.")
    parser.add_argument("--backend", default="vigem", choices=["vigem", "vjoy"], help="Controller backend to test.")
    parser.add_argument("--cycles", type=int, default=20, help="Number of up/down cycles to send.")
    parser.add_argument("--press-seconds", type=float, default=0.25, help="How long to hold each button press.")
    parser.add_argument("--gap-seconds", type=float, default=0.35, help="Delay between button presses.")
    parser.add_argument("--rate-hz", type=float, default=25.0, help="How often to refresh the virtual controller.")
    return parser.parse_args()


def neutral(controls):
    controls.set_controls(steer=0.0, acc=-1.0, brake=-1.0, enable_gear_shift=False)


def press_shift(controls, *, shift_up, duration_s, rate_hz):
    label = "A / shift up" if shift_up else "X / shift down"
    print(f"Pressing {label}")

    interval_s = 1.0 / rate_hz
    end_time = time.perf_counter() + duration_s
    while time.perf_counter() < end_time:
        controls.set_controls(
            steer=0.0,
            acc=-1.0,
            brake=-1.0,
            enable_gear_shift=True,
            shift_up=shift_up,
            shift_down=not shift_up,
        )
        time.sleep(interval_s)

    neutral(controls)


def main():
    args = parse_args()
    controls = Controls(backend=args.backend)

    try:
        neutral(controls)
        print(
            f"Testing {args.backend} shift buttons. "
            f"Watch an input monitor or Assetto Corsa bindings for A then X."
        )
        for cycle in range(1, args.cycles + 1):
            print(f"Cycle {cycle}/{args.cycles}")
            press_shift(controls, shift_up=True, duration_s=args.press_seconds, rate_hz=args.rate_hz)
            time.sleep(args.gap_seconds)
            press_shift(controls, shift_up=False, duration_s=args.press_seconds, rate_hz=args.rate_hz)
            time.sleep(args.gap_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted; releasing controls.")
    finally:
        neutral(controls)
        controls.close()


if __name__ == "__main__":
    main()
