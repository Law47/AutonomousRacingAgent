import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

LOCAL_IMPORT_PATHS = [
    os.path.join(ROOT_DIR, "assetto_corsa_gym"),
    os.path.join(ROOT_DIR, "algorithm", "discor"),
]
for path in reversed(LOCAL_IMPORT_PATHS):
    sys.path.insert(0, path)

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from AssettoCorsaEnv.ac_env import TERMINAL_JUDGE_TIMEOUT
from AssettoCorsaEnv.brake_map import BrakeMap
import Common.logging_config as logging_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record manual demonstrations for Assetto Corsa")
    parser.add_argument("--config", default="config.yml", type=str, help="Config path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where demonstration files will be written",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional maximum number of recorded telemetry steps (0 means unlimited)",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="OmegaConf dotlist overrides",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(ROOT_DIR, "demonstrations", datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3])
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def wait_for_record_start() -> None:
    print("\n" + "=" * 60)
    print("Launch Assetto Corsa and get the car ready on track.")
    print("Press SPACE to begin recording. Press Q while recording to stop.")
    print("=" * 60 + "\n")

    if os.name == "nt":
        import msvcrt

        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b" ":
                return
            time.sleep(0.05)

    input("Press Enter to begin recording...")


def should_stop_recording() -> bool:
    if os.name != "nt":
        return False

    import msvcrt

    if not msvcrt.kbhit():
        return False

    key = msvcrt.getch().lower()
    return key == b"q"


def validate_runtime_config(env) -> None:
    env.static_info = env.client.simulation_management.get_static_info()
    env.track_length = env.static_info["TrackLength"]
    env.ac_mod_config = env.client.simulation_management.get_config()

    if env.config.screen_capture_enable:
        assert env.config.final_image_height == env.ac_mod_config["final_image_height"]
        assert env.config.final_image_width == env.ac_mod_config["final_image_width"]
        assert env.config.color_mode == env.ac_mod_config["color_mode"]

    assert env.config.ego_sampling_freq == env.ac_mod_config["ego_sampling_freq"], "Ego sampling frequency mismatch"
    assert env.static_info["TrackFullName"] == env.track_name, (
        f"Track name mismatch. Running: {env.static_info['TrackFullName']} Configured: {env.track_name}"
    )
    assert env.static_info["CarName"] == env.car_name, (
        f"Car name mismatch. Running: {env.static_info['CarName']} Configured: {env.car_name}"
    )


def get_absolute_controls(state, steer_max, brake_map):
    steer = float(np.clip(state["steerAngle"] / steer_max, -1.0, 1.0))
    throttle = float(np.clip((state["accStatus"] - 0.5) * 2.0, -1.0, 1.0))
    brake = float(np.clip(brake_map.get_x(state["brakeStatus"]).item(), -1.0, 1.0))
    return np.array([steer, throttle, brake], dtype=np.float32)


def interpolate_controls(states, absolute_controls):
    if len(states) <= 1:
        return absolute_controls.astype(np.float32)

    timestamps = np.array([state["capture_time_s"] for state in states], dtype=np.float64)
    stable_timestamps = timestamps.copy()
    for i in range(1, len(stable_timestamps)):
        if stable_timestamps[i] <= stable_timestamps[i - 1]:
            stable_timestamps[i] = stable_timestamps[i - 1] + 1e-6

    interpolated = absolute_controls.astype(np.float64).copy()
    for channel in range(interpolated.shape[1]):
        series = pd.Series(interpolated[:, channel]).interpolate(method="linear", limit_direction="both")
        interpolated[:, channel] = np.interp(stable_timestamps, stable_timestamps, series.to_numpy())
    return interpolated.astype(np.float32)


def enrich_states_with_actions(env, recorded_states):
    if not recorded_states:
        return []

    brake_map_file = Path(env.ac_configs_path) / "cars" / env.config.car / "brake_map.csv"
    brake_map = BrakeMap.load(brake_map_file)
    absolute_controls = np.vstack(
        [get_absolute_controls(state, env.max_steer_deg, brake_map) for state in recorded_states]
    )
    absolute_controls = interpolate_controls(recorded_states, absolute_controls)

    processed_states = []
    previous_abs_controls = absolute_controls[0]
    previous_gear = int(recorded_states[0].get("actualGear", 0))

    for index, raw_state in enumerate(recorded_states):
        state = raw_state.copy()
        current_abs_controls = absolute_controls[index]

        if index == 0:
            model_actions = np.zeros(env.action_dim, dtype=np.float32)
        else:
            model_actions = np.zeros(env.action_dim, dtype=np.float32)
            model_actions[:env.control_action_dim] = env.inverse_preprocess_actions(
                previous_abs_controls,
                current_abs_controls,
            )

            current_gear = int(state.get("actualGear", previous_gear))
            gear_delta = current_gear - previous_gear
            if gear_delta > 0:
                model_actions[3] = 1.0
            elif gear_delta < 0:
                model_actions[4] = 1.0
            previous_gear = current_gear

        for control_index in range(env.control_action_dim):
            state[f"current_action_abs_{control_index}"] = float(current_abs_controls[control_index])
        for action_index in range(env.action_dim):
            state[f"actions_{action_index}"] = float(model_actions[action_index])
        state["shift_up"] = bool(model_actions[3] > 0.5)
        state["shift_down"] = bool(model_actions[4] > 0.5)

        _, actions_diff = env.get_obs(state, processed_states)
        state["reward"] = env.get_reward(state, actions_diff).item()

        processed_states.append(state)
        previous_abs_controls = current_abs_controls

    return processed_states


def save_demonstration(output_dir, states, static_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
    save_path = os.path.join(output_dir, f"{timestamp}_demo.pkl")
    payload = {"states": states, "static_info": static_info}
    with open(save_path, "wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved demonstration to %s", save_path)
    return save_path


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    output_dir = build_output_dir(args)
    logging_config.create_logging(level=logging.DEBUG, file_name=os.path.join(output_dir, "record_demo.log"))
    logging.getLogger().setLevel(logging.INFO)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=output_dir)
    wait_for_record_start()

    env.client.reset(send_reset=False)
    validate_runtime_config(env)
    env.shift_gate.reset()
    env.termination_counter = int(TERMINAL_JUDGE_TIMEOUT * env.ctrl_rate)

    recorded_states = []
    logger.info("Recording demonstration to %s", output_dir)

    try:
        while True:
            raw_state = env.client.step_sim()
            capture_time = time.perf_counter()
            raw_state["timestamp_env"] = capture_time
            raw_state["capture_time_s"] = capture_time
            expanded_state, _ = env.expand_state(raw_state)
            recorded_states.append(expanded_state)

            if args.max_steps and len(recorded_states) >= args.max_steps:
                logger.info("Reached max_steps=%s", args.max_steps)
                break
            if should_stop_recording():
                logger.info("Stop key detected. Finishing recording.")
                break
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    finally:
        env.client.close()

    if not recorded_states:
        raise ValueError("No telemetry frames were recorded")

    processed_states = enrich_states_with_actions(env, recorded_states)
    save_path = save_demonstration(output_dir, processed_states, env.static_info)
    print(f"Saved demonstration: {save_path}")


if __name__ == "__main__":
    main()
