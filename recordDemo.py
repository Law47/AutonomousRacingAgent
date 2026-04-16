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
from AssettoCorsaEnv.ac_env import (
    PAST_ACTIONS_WINDOW,
    STREAMING_DEMO_FORMAT,
    TERMINAL_JUDGE_TIMEOUT,
)
from AssettoCorsaEnv.brake_map import BrakeMap
from AssettoCorsaEnv.gear_shift_labels import (
    DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR,
    infer_shift_from_state,
)
import Common.logging_config as logging_config

logger = logging.getLogger(__name__)
DEFAULT_FLUSH_INTERVAL_S = 300.0


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
        "--flush_interval_s",
        type=float,
        default=DEFAULT_FLUSH_INTERVAL_S,
        help="How often to flush recorded demonstration chunks to disk",
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


def build_demo_file_path(output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
    return os.path.join(output_dir, f"{timestamp}_demo.pkl")


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


def enrich_states_with_actions(
    env,
    recorded_states,
    previous_abs_controls=None,
    previous_stable_gear=None,
    history_tail=None,
    shift_label_min_drive_gear=DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR,
):
    if not recorded_states:
        return [], previous_abs_controls, previous_stable_gear, history_tail or []

    if history_tail is None:
        history_tail = []

    brake_map_file = Path(env.ac_configs_path) / "cars" / env.config.car / "brake_map.csv"
    brake_map = BrakeMap.load(brake_map_file)
    absolute_controls = np.vstack(
        [get_absolute_controls(state, env.max_steer_deg, brake_map) for state in recorded_states]
    )
    absolute_controls = interpolate_controls(recorded_states, absolute_controls)

    processed_states = []
    for index, raw_state in enumerate(recorded_states):
        state = raw_state.copy()
        current_abs_controls = absolute_controls[index]
        model_actions = np.zeros(env.action_dim, dtype=np.float32)

        if previous_abs_controls is not None:
            model_actions[:env.control_action_dim] = env.inverse_preprocess_actions(
                previous_abs_controls,
                current_abs_controls,
            )
        shift_actions, previous_stable_gear, _ = infer_shift_from_state(
            state,
            previous_stable_gear,
            min_drive_gear=shift_label_min_drive_gear,
        )
        model_actions[3:5] = shift_actions

        for control_index in range(env.control_action_dim):
            state[f"current_action_abs_{control_index}"] = float(current_abs_controls[control_index])
        for action_index in range(env.action_dim):
            state[f"actions_{action_index}"] = float(model_actions[action_index])
        state["shift_up"] = bool(model_actions[3] > 0.5)
        state["shift_down"] = bool(model_actions[4] > 0.5)

        _, actions_diff = env.get_obs(state, history_tail + processed_states)
        state["reward"] = env.get_reward(state, actions_diff).item()

        processed_states.append(state)
        previous_abs_controls = current_abs_controls

    history_tail = (history_tail + processed_states)[-PAST_ACTIONS_WINDOW:]
    return processed_states, previous_abs_controls, previous_stable_gear, history_tail


def initialize_streaming_demo_file(save_path, static_info):
    payload = {
        "format": STREAMING_DEMO_FORMAT,
        "static_info": static_info,
        "created_at": datetime.now().isoformat(),
    }
    with open(save_path, "wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    logger.info("Initialized streaming demonstration file at %s", save_path)


def append_demo_chunk(save_path, states):
    if not states:
        return 0

    payload = {
        "states": states,
        "saved_at": datetime.now().isoformat(),
        "count": len(states),
    }
    with open(save_path, "ab") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    logger.info("Flushed %s recorded states to %s", len(states), save_path)
    return len(states)


def flush_recorded_states(
    env,
    save_path,
    pending_states,
    previous_abs_controls,
    previous_stable_gear,
    history_tail,
    shift_label_min_drive_gear,
):
    if not pending_states:
        return previous_abs_controls, previous_stable_gear, history_tail, 0

    processed_states, previous_abs_controls, previous_stable_gear, history_tail = enrich_states_with_actions(
        env,
        pending_states,
        previous_abs_controls=previous_abs_controls,
        previous_stable_gear=previous_stable_gear,
        history_tail=history_tail,
        shift_label_min_drive_gear=shift_label_min_drive_gear,
    )
    saved_count = append_demo_chunk(save_path, processed_states)
    pending_states.clear()
    return previous_abs_controls, previous_stable_gear, history_tail, saved_count


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    output_dir = build_output_dir(args)
    logging_config.create_logging(level=logging.DEBUG, file_name=os.path.join(output_dir, "record_demo.log"))
    logging.getLogger().setLevel(logging.INFO)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=output_dir)
    demo_config = getattr(config, "Demonstrations", None)
    shift_label_min_drive_gear = int(
        getattr(demo_config, "shift_label_min_drive_gear", DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR)
    )
    wait_for_record_start()

    env.client.reset(send_reset=False)
    validate_runtime_config(env)
    env.shift_gate.reset()
    env.termination_counter = int(TERMINAL_JUDGE_TIMEOUT * env.ctrl_rate)

    save_path = build_demo_file_path(output_dir)
    initialize_streaming_demo_file(save_path, env.static_info)

    recorded_states = []
    previous_abs_controls = None
    previous_stable_gear = None
    history_tail = []
    total_saved_states = 0
    total_seen_states = 0
    last_flush_time = time.perf_counter()
    logger.info("Recording demonstration to %s", output_dir)

    try:
        while True:
            raw_state = env.client.step_sim()
            capture_time = time.perf_counter()
            raw_state["timestamp_env"] = capture_time
            raw_state["capture_time_s"] = capture_time
            expanded_state, _ = env.expand_state(raw_state)
            recorded_states.append(expanded_state)
            total_seen_states += 1

            should_flush = (capture_time - last_flush_time) >= args.flush_interval_s
            if should_flush:
                previous_abs_controls, previous_stable_gear, history_tail, saved_count = flush_recorded_states(
                    env,
                    save_path,
                    recorded_states,
                    previous_abs_controls,
                    previous_stable_gear,
                    history_tail,
                    shift_label_min_drive_gear,
                )
                total_saved_states += saved_count
                last_flush_time = capture_time

            if args.max_steps and total_seen_states >= args.max_steps:
                logger.info("Reached max_steps=%s", args.max_steps)
                break
            if should_stop_recording():
                logger.info("Stop key detected. Finishing recording.")
                break
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    finally:
        previous_abs_controls, previous_stable_gear, history_tail, saved_count = flush_recorded_states(
            env,
            save_path,
            recorded_states,
            previous_abs_controls,
            previous_stable_gear,
            history_tail,
            shift_label_min_drive_gear,
        )
        total_saved_states += saved_count
        env.client.close()

    if total_saved_states == 0:
        raise ValueError("No telemetry frames were recorded")

    print(f"Saved demonstration: {save_path} ({total_saved_states} states)")


if __name__ == "__main__":
    main()
