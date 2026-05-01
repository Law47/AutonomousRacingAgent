"""
Human Demonstration Recorder for Assetto Corsa

Records human driving sessions as (state, action, reward, next_state, done)
transitions directly from AC shared memory + physical gamepad input.
Saved .npz files can be loaded into the replay buffer via:

    python run.py --demo_path Demonstrations/my_lap.npz

USAGE:
    python record.py                          # Record and save to Demonstrations/
    python record.py --out my_lap.npz         # Custom output path
    python record.py --joystick 1             # Use joystick index 1 (default: 0)

CONTROLLER AXIS MAPPING (Xbox / standard gamepad):
    Axis 0 : Left stick X  → Steer     (-1 = left, +1 = right)
    Axis 4 : Left trigger  → Brake     (-1 = released, +1 = fully pressed)
    Axis 5 : Right trigger → Throttle  (-1 = released, +1 = fully pressed)

    If your controller uses different axes, adjust AXIS_STEER / AXIS_THROTTLE /
    AXIS_BRAKE below. Run with --list_axes to see live axis values.

CONTROLS DURING RECORDING:
    Press Ctrl+C to stop and save.
"""

import argparse
import ctypes
import mmap
import os
import sys
import time

import numpy as np
import pygame

# Ensure the workspace root is on the path so imports match run.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sharedMemoryStructs import SPageFilePhysics, SPageFileGraphic
from acEnv import ACEnv  # Used only for class-level constants (no instance created)

# ---------------------------------------------------------------------------
# Controller axis indices – adjust for your controller if needed
# ---------------------------------------------------------------------------
AXIS_STEER    = 0   # Left stick X
AXIS_THROTTLE = 5   # Right trigger
AXIS_BRAKE    = 4   # Left trigger

RECORD_HZ     = 25  # Frames per second to record (matches env step rate)

# ---------------------------------------------------------------------------
# Shared memory helpers (mirrors ACEnv.getObservation, no class instance needed)
# ---------------------------------------------------------------------------
_FIELD_MAPPING = ACEnv._field_mapping
_OBS_INPUTS    = ACEnv.observation_inputs
_OBS_INFO      = ACEnv.observation_info


def _extract_field(physics, graphics, field_name):
    """Mirror of ACEnv.extractFieldValue without using 'self'."""
    wheel_suffixes  = ['FL', 'FR', 'RL', 'RR']
    vector_suffixes = ['X', 'Y', 'Z']

    for suffix in wheel_suffixes:
        if field_name.endswith(suffix):
            idx = _FIELD_MAPPING[suffix]
            base = field_name[:-len(suffix)]
            if hasattr(physics, base):
                return getattr(physics, base)[idx]

    for suffix in vector_suffixes:
        if field_name.endswith(suffix):
            idx = _FIELD_MAPPING[suffix]
            base = field_name[:-len(suffix)]
            if hasattr(physics, base):
                return getattr(physics, base)[idx]
            if hasattr(graphics, base):
                return getattr(graphics, base)[idx]

    if field_name.startswith('rideHeight'):
        idx = 0 if 'Front' in field_name else 1
        return physics.rideHeight[idx]

    if hasattr(graphics, field_name):
        return getattr(graphics, field_name)

    return None


def get_observation(physics_mmap, graphics_mmap):
    """Read one observation vector from AC shared memory."""
    physics  = SPageFilePhysics.from_buffer_copy(physics_mmap)
    graphics = SPageFileGraphic.from_buffer_copy(graphics_mmap)

    obs = np.zeros(len(_OBS_INPUTS), dtype=np.float32)
    for i, name in enumerate(_OBS_INPUTS):
        try:
            if hasattr(physics, name):
                value = getattr(physics, name)
            else:
                value = _extract_field(physics, graphics, name)

            if value is not None and name in _OBS_INFO:
                max_val = _OBS_INFO[name]
                obs[i] = float(value) / max_val if max_val != 0 else 0.0
        except Exception:
            obs[i] = 0.0

    return obs, physics, graphics


def get_reward(physics):
    """Compute reward from physics state (mirrors acEnv.getReward base signal)."""
    speed = float(physics.speedKmh)
    return np.float32(speed / 300.0)


def get_action(joystick):
    """Read and convert physical gamepad to [throttle_brake, steer] action."""
    pygame.event.pump()

    steer    = float(joystick.get_axis(AXIS_STEER))

    # Triggers: pygame reports [-1 (released) … +1 (fully pressed)]
    throttle = (float(joystick.get_axis(AXIS_THROTTLE)) + 1.0) / 2.0  # [0, 1]
    brake    = (float(joystick.get_axis(AXIS_BRAKE))    + 1.0) / 2.0  # [0, 1]

    throttle_brake = float(np.clip(throttle - brake, -1.0, 1.0))
    steer          = float(np.clip(steer,            -1.0, 1.0))

    return np.array([throttle_brake, steer], dtype=np.float32)


# ---------------------------------------------------------------------------
# Axis diagnostic mode
# ---------------------------------------------------------------------------
def list_axes(joystick):
    print("Live axis values – move your controller. Press Ctrl+C to stop.\n")
    try:
        while True:
            pygame.event.pump()
            vals = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            line = "  ".join(f"Axis{i}: {v:+.3f}" for i, v in enumerate(vals))
            print(f"\r{line}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------
def record(out_path, joystick_index):
    # Connect to AC shared memory
    try:
        physics_mmap  = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics),  "acpmf_physics")
        graphics_mmap = mmap.mmap(0, ctypes.sizeof(SPageFileGraphic), "acpmf_graphics")
        print("Connected to Assetto Corsa shared memory.")
    except Exception as e:
        print(f"ERROR: Could not connect to AC shared memory: {e}")
        print("Make sure Assetto Corsa is running with a session active.")
        sys.exit(1)

    # Connect to joystick
    pygame.init()
    pygame.joystick.init()
    num_joysticks = pygame.joystick.get_count()
    if num_joysticks == 0:
        print("ERROR: No joystick / controller detected.")
        sys.exit(1)
    if joystick_index >= num_joysticks:
        print(f"ERROR: Joystick index {joystick_index} not found. "
              f"Available indices: 0–{num_joysticks - 1}")
        sys.exit(1)

    joystick = pygame.joystick.Joystick(joystick_index)
    joystick.init()
    print(f"Using controller: {joystick.get_name()} (index {joystick_index})")
    print(f"Recording at {RECORD_HZ} Hz. Press Ctrl+C to stop and save.\n")

    states      = []
    actions     = []
    rewards     = []
    next_states = []
    dones       = []

    frame_duration = 1.0 / RECORD_HZ
    last_packet_id = -1

    try:
        # Prime the first observation
        obs, physics, _ = get_observation(physics_mmap, graphics_mmap)
        state = obs.copy()

        frame = 0
        t_next = time.perf_counter() + frame_duration

        while True:
            # Wait for a new physics packet (AC runs at ~333 Hz internally;
            # we poll until we see a new packet ID or hit our frame deadline)
            deadline = time.perf_counter() + frame_duration
            while time.perf_counter() < deadline:
                obs, physics, graphics = get_observation(physics_mmap, graphics_mmap)
                if physics.packetId != last_packet_id:
                    last_packet_id = physics.packetId
                    break
                time.sleep(0.001)

            next_state = obs.copy()
            action     = get_action(joystick)
            reward     = get_reward(physics)
            done       = bool(physics.numberOfTyresOut >= 3)

            states.append(state)
            actions.append(action)
            rewards.append(np.array([reward], dtype=np.float32))
            next_states.append(next_state)
            dones.append(np.array([float(done)], dtype=np.float32))

            frame += 1
            if frame % (RECORD_HZ * 5) == 0:  # Progress every 5 seconds
                elapsed = frame / RECORD_HZ
                speed   = float(physics.speedKmh)
                print(f"  {elapsed:5.0f}s | {frame:6d} frames | speed: {speed:5.1f} km/h | "
                      f"throttle_brake: {action[0]:+.2f} | steer: {action[1]:+.2f}")

            state = next_state

            # Sleep until next target frame
            sleep_for = t_next - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            t_next += frame_duration

    except KeyboardInterrupt:
        print(f"\nRecording stopped. Captured {len(states)} frames "
              f"({len(states) / RECORD_HZ:.1f} seconds).")

    if len(states) == 0:
        print("No frames recorded – nothing saved.")
        return

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        states      = np.array(states,      dtype=np.float32),
        actions     = np.array(actions,     dtype=np.float32),
        rewards     = np.array(rewards,     dtype=np.float32),
        next_states = np.array(next_states, dtype=np.float32),
        dones       = np.array(dones,       dtype=np.float32),
    )
    print(f"Saved {len(states)} transitions to: {out_path}")
    print(f"Load into training with:  python run.py --demo_path \"{out_path}\"")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record human demonstration laps in Assetto Corsa")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz file path (default: Demonstrations/demo_<timestamp>.npz)")
    parser.add_argument("--joystick", type=int, default=0,
                        help="Joystick device index (default: 0). Use --list_axes to identify axes.")
    parser.add_argument("--list_axes", action="store_true",
                        help="Print live axis values for controller identification, then exit.")
    args = parser.parse_args()

    if args.list_axes:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick detected.")
            sys.exit(1)
        joy = pygame.joystick.Joystick(args.joystick)
        joy.init()
        print(f"Controller: {joy.get_name()}")
        list_axes(joy)
        sys.exit(0)

    if args.out is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("Demonstrations", f"demo_{timestamp}.npz")

    record(args.out, args.joystick)
