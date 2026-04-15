import glob
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import yaml
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from AssettoCorsaEnv.brake_map import BrakeMap

def read_yml(f):
    with open(f, 'r') as file:
        return yaml.safe_load(file)

def seconds_to_mm_ss_mmm(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:2d}:{remaining_seconds:06.3f}"

class DataLoader():
    def __init__(self, env, data_set_path, log_steer_ratios=False):
        self.env = env

        # Find all .pkl and .parquet files in the dataset path
        self.trajectories_paths = sorted(
            glob.glob(data_set_path + '/*.pkl') + glob.glob(data_set_path + '/*.parquet')
        )
        self.trajectories_count = len(self.trajectories_paths)
        assert self.trajectories_count > 0, f"No trajectories found in {data_set_path}"
        logger.info(f"Found {self.trajectories_count} trajectories in the path: {data_set_path}")

        self.trajectory_number = 0
        self.current_step = 0
        self.prev_abs_actions = None
        self.log_steer_ratios = log_steer_ratios

        # load the brake and steer maps from the env config!!! -> check if using another car
        brake_map_file = Path(env.ac_configs_path) / "cars" / env.config.car / 'brake_map.csv'
        self.brake_map = BrakeMap.load(brake_map_file)
        self.steer_max = env.max_steer_deg

    def get_absolute_actions_from_state(self, state):
        steer = state["steerAngle"] / self.steer_max
        pedal = (state["accStatus"] - 0.5) * 2  # 0,1 -> -1,1
        brake = self.brake_map.get_x(state["brakeStatus"]).item() # map
        return np.array( [steer, pedal, brake] )

    def get_actions_from_state(self, state):
        return self.get_absolute_actions_from_state(state)

    def get_recorded_model_actions(self, state):
        action_dim = getattr(self.env, "action_dim", 5)
        action_keys = [f"actions_{i}" for i in range(action_dim)]
        if all(key in state for key in action_keys):
            return np.array([state[key] for key in action_keys], dtype='float32')

        legacy_action_keys = [f"actions_{i}" for i in range(min(3, action_dim))]
        if all(key in state for key in legacy_action_keys):
            actions = np.zeros(action_dim, dtype='float32')
            actions[:len(legacy_action_keys)] = np.array([state[key] for key in legacy_action_keys], dtype='float32')
            return actions

        return None

    def infer_shift_actions(self, current_state, previous_state):
        if previous_state is None:
            return np.zeros(2, dtype='float32')

        current_gear = int(current_state.get("actualGear", 0))
        previous_gear = int(previous_state.get("actualGear", current_gear))
        gear_delta = current_gear - previous_gear

        if gear_delta > 0:
            return np.array([1.0, 0.0], dtype='float32')
        if gear_delta < 0:
            return np.array([0.0, 1.0], dtype='float32')
        return np.zeros(2, dtype='float32')

    def pad_model_actions(self, actions):
        actions = np.asarray(actions, dtype='float32')
        action_dim = getattr(self.env, "action_dim", actions.shape[0])
        if action_dim == actions.shape[0]:
            return actions
        if actions.shape[0] > action_dim:
            raise ValueError(f"Expected at most {action_dim} actions, got {actions.shape[0]}")

        padded_actions = np.zeros(action_dim, dtype='float32')
        padded_actions[:actions.shape[0]] = actions
        return padded_actions

    def compute_steer_ratio_statistics(self, trajectory):
        # trajectory is a list of dictionaries
        lap_data = defaultdict(list)

        for entry in trajectory:
            lap_data[entry["LapCount"]].append(entry["steerAngle"])

        # Process each lap
        for lap, steer_angles in lap_data.items():
            steer_angles = np.array(steer_angles)  # Convert to NumPy array
            steer_ratio_change = np.diff(steer_angles) * self.env.config.ego_sampling_freq
            logger.info(f"Lap: {lap} steer ratio change: {np.max(np.abs(steer_ratio_change)):8.2f}deg/s max: {np.max(np.abs(steer_angles)):8.2f}deg")
            #if np.max(np.abs(steer_ratio_change)) > 1500:
                # to debug outliers
                # breakpoint()#pd.DataFrame({"steerAngle": steer_angles, "steer_ratio_change": np.append(steer_ratio_change, np.nan)}).to_csv("steer_ratio_data.csv", index=False)

    def load_next_trajectory(self):
        load_path = self.trajectories_paths[self.trajectory_number]
        assert Path(load_path).exists(), f"Trajectory file not found: {load_path}"
        try:
            self.trajectory, self.static_info = self.env.load_history(load_path)
            self.trajectory_number += 1
            self.current_step = 0
            if self.log_steer_ratios:
                self.compute_steer_ratio_statistics(self.trajectory)
        except Exception:
            logger.error(f"Error loading trajectory: {load_path}")
            raise

    def read_step(self):
        state = self.trajectory[self.current_step]
        history = self.trajectory[:self.current_step] # get the history seen so far
        previous_state = self.trajectory[self.current_step - 1] if self.current_step > 0 else None
        current_abs_actions = self.get_absolute_actions_from_state(state)

        if self.current_step == 0:
            self.prev_abs_actions = current_abs_actions

        recorded_model_actions = self.get_recorded_model_actions(state)
        if recorded_model_actions is not None:
            actions = recorded_model_actions
        else:
            controls_actions = self.env.inverse_preprocess_actions(self.prev_abs_actions, current_abs_actions)
            shift_actions = self.infer_shift_actions(state, previous_state)
            actions = self.pad_model_actions(np.concatenate([controls_actions, shift_actions]))
        self.prev_abs_actions = current_abs_actions

        # abs values or relative
        self.current_actions = self.pad_model_actions(current_abs_actions)
        self.action = self.pad_model_actions(actions)

        self.state = state
        # re build the observations and the reward using the current environment settings
        self.obs, self.actions_diff = self.env.get_obs(state, history)
        self.reward = self.env.get_reward(state, self.actions_diff).item()

        done = False
        terminated = False
        if self.state["out_of_track"]: # or AC oot
            terminated = True
            #self.reward = .0
            # don't end the episode on oot , human laps are full of them
            # but we set the termination signal which is needed to train the model
        self.current_step += 1
        if self.current_step == len(self.trajectory):
            done = True

        truncated = False
        if done and not terminated:
            truncated = True

        self.info = {"terminated": float(terminated),
                     "obs_extra": self.env.get_extra_observations(state),
                     "TimeLimit.truncated": float(truncated)
                     }
        self.done = float(done)

    def reset(self):
        self.load_next_trajectory()
        self.read_step()
        return self.obs

    def get_info(self):
        return self.info.copy()

    def act(self):
        self.read_step()   # set get obs to t+1
        return self.action # actions  that took the car from t-1 to t

    def step(self, action):
        # returns at t+1
        return self.obs, self.reward, self.done, self.info

    def get_task_id(self):
        return self.env.get_task_id()

def get_path_for_track_car(dataset_path, file, track, car):
    data = read_yml(file)
    paths = data[track][car]
    paths = [dataset_path / Path(f"{track}/{car}") / p["id"] for p in paths]
    return paths

def get_all_paths_in_file(file, dataset_path, filter_tags={}, filter_track=None, filter_car=None):
    data = read_yml(file)
    all_paths = []

    for track, cars in data.items():
        if filter_track and track != filter_track:
            continue
        for car, entries in cars.items():
            if filter_car and car != filter_car:
                continue
            for entry in entries:
                # Check if entry matches all specified filter tags
                if all(entry.get(tag) == value for tag, value in filter_tags.items()):
                    path = dataset_path / Path(f"{track}/{car}") / entry["id"]
                    all_paths.append((path.as_posix() + "/laps/", car, track))
    return all_paths

def lap_times_list(df):
    lap_times = []
    for i in list(set(df["LapCount"])):
        if i == 0: # discard out lap
            continue
        l = df[df["LapCount"] == i]
        t = l["lastLapTime"].values[-1]
        lap_times.append(t)
    return lap_times
