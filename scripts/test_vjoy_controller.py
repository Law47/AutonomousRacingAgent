import argparse
import os
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETTO_GYM_DIR = ROOT_DIR / "assetto_corsa_gym"

sys.path.insert(0, str(ASSETTO_GYM_DIR))

from AssettoCorsaEnv.vjoy import resolve_vjoy_dll_path
from AssettoCorsaPlugin.plugins.sensors_par.car_control import Controls


def parse_args():
    parser = argparse.ArgumentParser(description="Interactively test the vJoy controller backend.")
    parser.add_argument(
        "--dll-path",
        default=None,
        help="Optional full path to vJoyInterface.dll. Overrides automatic discovery.",
    )
    parser.add_argument("--rate-hz", type=float, default=25.0, help="Controller update rate.")
    return parser.parse_args()


def print_keymap():
    print("Keyboard controls:")
    print("  a -> steer left")
    print("  s -> steer center")
    print("  d -> steer right")
    print("  1 -> full throttle")
    print("  2 -> half throttle")
    print("  3 -> no throttle")
    print("  q -> full brake")
    print("  w -> half brake")
    print("  e -> no brake")
    print("  o -> shift up")
    print("  p -> shift down")
    print("  space -> reset steer/throttle/brake to neutral")
    print("  Esc or Ctrl+C -> exit")
    print()


def read_key_nonblocking():
    if os.name != "nt":
        raise RuntimeError("Interactive vJoy keyboard testing currently requires Windows")

    import msvcrt

    if not msvcrt.kbhit():
        return None

    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        msvcrt.getwch()
        return None
    return key.lower()


def apply_state(controls, state, shift_up=False, shift_down=False):
    controls.set_controls(
        steer=state["steer"],
        acc=state["acc"],
        brake=state["brake"],
        enable_gear_shift=shift_up or shift_down,
        shift_up=shift_up,
        shift_down=shift_down,
    )


def reset_state():
    return {"steer": 0.0, "acc": -1.0, "brake": -1.0}


def handle_key(key, controls, state):
    if key == "a":
        state["steer"] = -1.0
        print("steer left")
    elif key == "s":
        state["steer"] = 0.0
        print("steer center")
    elif key == "d":
        state["steer"] = 1.0
        print("steer right")
    elif key == "1":
        state["acc"] = 1.0
        print("full throttle")
    elif key == "2":
        state["acc"] = 0.0
        print("half throttle")
    elif key == "3":
        state["acc"] = -1.0
        print("no throttle")
    elif key == "q":
        state["brake"] = 1.0
        print("full brake")
    elif key == "w":
        state["brake"] = 0.0
        print("half brake")
    elif key == "e":
        state["brake"] = -1.0
        print("no brake")
    elif key == "o":
        print("shift up / vJoy button 1")
        apply_state(controls, state, shift_up=True)
        return
    elif key == "p":
        print("shift down / vJoy button 3")
        apply_state(controls, state, shift_down=True)
        return
    elif key == " ":
        state.update(reset_state())
        print("neutral controls")
    else:
        return

    apply_state(controls, state)


def main():
    args = parse_args()
    if args.dll_path:
        dll_path = str(Path(args.dll_path).expanduser().resolve())
    else:
        dll_path = resolve_vjoy_dll_path()

    print(f"Using vJoy DLL: {dll_path}")
    print("Open Windows 'Set up USB game controllers' or Assetto Corsa bindings to watch vJoy device input.")
    print_keymap()

    controls = Controls(backend="vjoy", vjoy_dll_path=dll_path)
    state = reset_state()
    interval_s = 1.0 / args.rate_hz
    try:
        apply_state(controls, state)
        while True:
            key = read_key_nonblocking()
            if key == "\x1b":
                print("Escape pressed; exiting.")
                break
            if key is not None:
                handle_key(key, controls, state)
            else:
                apply_state(controls, state)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nInterrupted; releasing vJoy controls.")
    finally:
        state.update(reset_state())
        apply_state(controls, state)
        controls.close()
        print("vJoy test finished.")


if __name__ == "__main__":
    main()
