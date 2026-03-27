from __future__ import annotations

from collections import deque
from logging import Logger
from typing import Optional

import ctypes
import keyboard
import mmap
import numpy as np
import socket
import threading
import time
from gymnasium import Env
from gymnasium import utils as gym_utils
from gymnasium.spaces import Box
from omegaconf import OmegaConf

from Common.controller_vjoy import VJoyController
from curriculum_scheduler import CurriculumScheduler
from racing_line_manager import RacingLineManager
from sharedMemoryStructs import SPageFileGraphic, SPageFilePhysics


TOP_SPEED_MS = 80
MAX_EPISODE_STEPS = 5000


def safe_clip(value: float, minimum: float, maximum: float) -> float:
    return float(np.clip(value, minimum, maximum))


class ACEnv(Env, gym_utils.EzPickle):
    raw_observation_info = {
        'gas': 1.0,
        'brake': 1.0,
        'pedalCommand': 1.0,
        'appliedAccel': 1.0,
        'appliedBrake': 1.0,
        'gear': 6.0,
        'rpms': 10000.0,
        'steerAngle': np.pi,
        'speedKmh': TOP_SPEED_MS * 3.6,
        'velocityX': TOP_SPEED_MS,
        'velocityY': 20.0,
        'velocityZ': 5.0,
        'accGX': 5.0,
        'accGY': 5.0,
        'accGZ': 5.0,
        'wheelSlipFL': 1.0,
        'wheelSlipFR': 1.0,
        'wheelSlipRL': 1.0,
        'wheelSlipRR': 1.0,
        'localAngularVelX': np.pi,
        'localAngularVelY': np.pi,
        'localAngularVelZ': np.pi,
        'numberOfTyresOut': 4.0,
        'normalizedCarPosition': 1.0,
        'finalFF': 1.0,
    }

    raw_observation_inputs = [
        'gas', 'brake', 'pedalCommand', 'appliedAccel', 'appliedBrake', 'gear', 'rpms', 'steerAngle', 'speedKmh',
        'velocityX', 'velocityY', 'velocityZ',
        'accGX', 'accGY', 'accGZ',
        'wheelSlipFL', 'wheelSlipFR', 'wheelSlipRL', 'wheelSlipRR',
        'localAngularVelX', 'localAngularVelY', 'localAngularVelZ',
        'numberOfTyresOut', 'normalizedCarPosition', 'finalFF',
    ]

    track_feature_names = ['line_gap', 'heading_error', 'forward_progress', 'off_track', 'target_speed']

    _field_mapping = {
        'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3,
        'X': 0, 'Y': 1, 'Z': 2,
    }

    def __init__(self, config: OmegaConf, logger: Logger):
        self.config = config
        self.logger = logger
        self._max_episode_steps = MAX_EPISODE_STEPS
        self.is_metaworld = False

        self.assetto_cfg = config.AssettoCorsa
        self.obs_cfg = config.get('observation', {})
        self.reward_cfg = config.get('reward', {})
        self.termination_cfg = config.get('termination', {})
        self.controller_cfg = config.get('controller', {})

        self._history_length = int(self.obs_cfg.get('history_length', 3))
        self._include_previous_obs = bool(self.assetto_cfg.get('add_previous_obs_to_state', True))
        self._include_previous_actions = bool(self.obs_cfg.get('include_previous_actions', True))
        self._include_track_relative_features = bool(self.obs_cfg.get('include_track_relative_features', True))
        self._curvature_points = int(self.obs_cfg.get('curvature_lookahead_points', 12))

        action_low = np.array([-1.0, -1.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_dim = 2
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        self._base_obs_dim = len(self.raw_observation_inputs) + len(self.track_feature_names) + self._curvature_points
        self.state_dim = self._base_obs_dim
        if self._include_previous_obs:
            self.state_dim += self._history_length * self._base_obs_dim
        if self._include_previous_actions:
            self.state_dim += self._history_length * self.action_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.physicsMMAP = None
        self.physics = None
        self.physicsConnected = False
        self.graphicsMMAP = None
        self.graphics = None
        self.graphicsConnected = False

        self.state = {'packetID': 0}
        self._episode_step = 0
        self._prev_norm_pos = None
        self._position_after_reset = None
        self._stuck_start_time = None
        self._low_speed_start_time = None
        self._last_reset_time = None
        self._reset_cooldown = 2.0
        self._p_key_pressed = False
        self._last_reward_breakdown = {}
        self._last_track_features = {}
        self._current_core_obs = np.zeros(self._base_obs_dim, dtype=np.float32)
        self._core_obs_history = deque(maxlen=self._history_length)
        self._action_history = deque(maxlen=self._history_length)
        self._last_applied_action = np.zeros(self.action_dim, dtype=np.float32)
        self._current_pedal_command = 0.0

        self._stuck_threshold = float(self.termination_cfg.get('stuck_distance_threshold_m', 2.0))
        self._stuck_timeout = float(self.termination_cfg.get('stuck_timeout_s', 5.0))
        self._low_speed_threshold = float(self.termination_cfg.get('low_speed_threshold_kmh', 5.0))
        self._low_speed_timeout = float(self.termination_cfg.get('low_speed_timeout_s', 10.0))

        self._reset_host = '127.0.0.1'
        self._reset_port = 65432
        self._reset_client_sock = None
        self._reset_client_lock = threading.Lock()
        self._reset_server_sock = None
        self._reset_server_thread = None
        self._start_reset_server()

        self.curriculum_scheduler = CurriculumScheduler(config, logger)
        self.racing_line_manager = RacingLineManager(config, logger)
        controller_device_id = int(self.controller_cfg.get('device_id', 1))
        controller_dll_path = self.controller_cfg.get('dll_path', None)
        self.controller = VJoyController(device_id=controller_device_id, dll_path=controller_dll_path)
        self.controller.neutral()
        self.logger.info(
            "vJoy controller initialized on device %s%s",
            controller_device_id,
            f" using {controller_dll_path}" if controller_dll_path else "",
        )

        self.logger.info(f"Action dim={self.action_dim}  Obs dim={self.state_dim}")
        self.connect()

    def _start_reset_server(self):
        try:
            self._reset_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._reset_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._reset_server_sock.bind((self._reset_host, self._reset_port))
            self._reset_server_sock.listen(1)
            self._reset_server_sock.settimeout(1.0)
            self._reset_server_thread = threading.Thread(target=self._reset_server_loop, daemon=True)
            self._reset_server_thread.start()
            self.logger.info(f"ACReset server listening on {self._reset_host}:{self._reset_port}")
        except Exception as exc:
            self.logger.error(f"Failed to start ACReset server: {exc}")

    def _reset_server_loop(self):
        while True:
            try:
                client_sock, addr = self._reset_server_sock.accept()
                with self._reset_client_lock:
                    if self._reset_client_sock is not None:
                        try:
                            self._reset_client_sock.close()
                        except Exception:
                            pass
                    self._reset_client_sock = client_sock
                self.logger.info(f"ACReset plugin connected from {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

    def _send_reset_command(self, retries: int = 10, retry_interval: float = 0.5) -> bool:
        for attempt in range(retries):
            with self._reset_client_lock:
                if self._reset_client_sock is not None:
                    try:
                        self._reset_client_sock.sendall(b"RESET\n")
                        self.logger.info("Sent RESET command to ACReset plugin")
                        return True
                    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                        self.logger.warning(f"ACReset plugin connection lost: {exc}")
                        try:
                            self._reset_client_sock.close()
                        except Exception:
                            pass
                        self._reset_client_sock = None
            if attempt < retries - 1:
                time.sleep(retry_interval)
        self.logger.warning("ACReset plugin did not connect after %.1fs", retries * retry_interval)
        return False

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def extractFieldValue(self, field_name: str, field_mapping: dict):
        wheel_suffixes = ['FL', 'FR', 'RL', 'RR']
        vector_suffixes = ['X', 'Y', 'Z']

        for suffix in wheel_suffixes:
            if field_name.endswith(suffix):
                idx = field_mapping[suffix]
                base_name = field_name[:-len(suffix)]
                if hasattr(self.physics, base_name):
                    return getattr(self.physics, base_name)[idx]

        for suffix in vector_suffixes:
            if field_name.endswith(suffix):
                idx = field_mapping[suffix]
                base_name = field_name[:-len(suffix)]
                if hasattr(self.physics, base_name):
                    return getattr(self.physics, base_name)[idx]
                if hasattr(self.graphics, base_name):
                    return getattr(self.graphics, base_name)[idx]

        if hasattr(self.physics, field_name):
            return getattr(self.physics, field_name)
        if hasattr(self.graphics, field_name):
            return getattr(self.graphics, field_name)
        return None

    def connect(self) -> None:
        self.physicsMMAP = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), 'acpmf_physics')
        self.physicsConnected = True
        self.logger.info("Connected to physics shared memory")
        self.graphicsMMAP = mmap.mmap(0, ctypes.sizeof(SPageFileGraphic), 'acpmf_graphics')
        self.graphicsConnected = True
        self.logger.info("Connected to graphics shared memory")

    def _read_shared_memory(self) -> bool:
        try:
            self.physics = SPageFilePhysics.from_buffer_copy(self.physicsMMAP)
            self.graphics = SPageFileGraphic.from_buffer_copy(self.graphicsMMAP)
            return True
        except Exception as exc:
            self.logger.error(f"Failed to read shared memory: {exc}")
            return False

    def _normalize_raw_feature(self, feature_name: str) -> float:
        if feature_name == 'pedalCommand':
            value = self._current_pedal_command
        elif feature_name == 'appliedAccel':
            value = self.state.get('applied_accel', 0.0)
        elif feature_name == 'appliedBrake':
            value = self.state.get('applied_brake', 0.0)
        else:
            value = self.extractFieldValue(feature_name, self._field_mapping)
        if value is None:
            return 0.0
        scale = float(self.raw_observation_info.get(feature_name, 1.0))
        if scale == 0.0:
            return 0.0
        return safe_clip(float(value) / scale, -5.0, 5.0)

    def _get_track_position(self) -> tuple[float, float, float]:
        coordinates = self.graphics.carCoordinates
        return float(coordinates[2]), float(coordinates[0]), float(coordinates[1])

    def _compute_track_features(self) -> tuple[np.ndarray, dict]:
        if not self._include_track_relative_features:
            zeros = np.zeros(len(self.track_feature_names) + self._curvature_points, dtype=np.float32)
            return zeros, {
                'distance': 0.0,
                'signed_distance': 0.0,
                'heading_error': 0.0,
                'lookahead_curvature': np.zeros(self._curvature_points, dtype=np.float32),
                'target_speed': 0.0,
            }

        car_pos = self._get_track_position()
        heading = float(self.physics.heading)
        track_features = self.racing_line_manager.get_track_features(car_pos, heading=heading)
        signed_distance = float(track_features['signed_distance'])
        heading_error = float(track_features['heading_error'])
        speed_ms = max(float(self.physics.speedKmh) / 3.6, 0.0)
        forward_progress = speed_ms * np.cos(heading_error) / TOP_SPEED_MS
        off_track = 1.0 if int(self.physics.numberOfTyresOut) >= 3 else 0.0
        target_speed = float(track_features.get('target_speed', 0.0)) / (TOP_SPEED_MS * 3.6)

        scalar_features = np.array([
            safe_clip(signed_distance / max(self.racing_line_manager.line_distance_threshold, 1e-6), -2.0, 2.0),
            safe_clip(heading_error / np.pi, -1.0, 1.0),
            safe_clip(forward_progress, -2.0, 2.0),
            off_track,
            safe_clip(target_speed, 0.0, 2.0),
        ], dtype=np.float32)
        lookahead = np.asarray(track_features['lookahead_curvature'], dtype=np.float32)
        return np.concatenate([scalar_features, lookahead]).astype(np.float32), track_features

    def _build_core_observation(self) -> np.ndarray:
        raw_features = np.array([
            self._normalize_raw_feature(feature_name)
            for feature_name in self.raw_observation_inputs
        ], dtype=np.float32)
        derived_features, track_features = self._compute_track_features()
        self._last_track_features = track_features
        return np.concatenate([raw_features, derived_features]).astype(np.float32)

    def _compose_observation(self, current_core_obs: np.ndarray) -> np.ndarray:
        chunks = [current_core_obs]
        if self._include_previous_obs:
            padded_history = [np.zeros(self._base_obs_dim, dtype=np.float32) for _ in range(self._history_length - len(self._core_obs_history))]
            padded_history.extend(list(self._core_obs_history))
            chunks.append(np.concatenate(padded_history).astype(np.float32))
        if self._include_previous_actions:
            padded_actions = [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self._history_length - len(self._action_history))]
            padded_actions.extend(list(self._action_history))
            chunks.append(np.concatenate(padded_actions).astype(np.float32))
        observation = np.concatenate(chunks).astype(np.float32)
        if observation.shape[0] != self.state_dim:
            raise ValueError(f"Observation shape mismatch: expected {self.state_dim}, got {observation.shape[0]}")
        return observation

    def _update_state_metadata(self):
        self.state['packetID'] = int(self.physics.packetId)
        self.state['tyresOut'] = int(self.physics.numberOfTyresOut)
        self.state['speed'] = float(self.physics.speedKmh)
        self.state['normalizedCarPosition'] = float(self.graphics.normalizedCarPosition)
        self.state['line_gap'] = float(self._last_track_features.get('signed_distance', 0.0))
        self.state['heading_error'] = float(self._last_track_features.get('heading_error', 0.0))
        heading_error = float(self._last_track_features.get('heading_error', 0.0))
        speed_ms = max(float(self.physics.speedKmh) / 3.6, 0.0)
        self.state['forward_progress'] = float(speed_ms * np.cos(heading_error))
        self.state['off_track'] = bool(int(self.physics.numberOfTyresOut) >= 3)
        self.state['abs_steer'] = abs(float(self._last_applied_action[0]))

    def getObservation(self) -> np.ndarray:
        if not self._read_shared_memory():
            return np.zeros(self.state_dim, dtype=np.float32)
        self._current_core_obs = self._build_core_observation()
        self._update_state_metadata()
        return self._compose_observation(self._current_core_obs)

    def getInfo(self) -> dict:
        return dict(self.state)

    def _safe_keyboard_pressed(self, key: str) -> bool:
        try:
            return keyboard.is_pressed(key)
        except Exception:
            return False

    def _check_termination(self) -> tuple[bool, bool, str]:
        terminated = False
        truncated = self._episode_step >= self._max_episode_steps
        reason = 'running'
        tyres_out = self.state.get('tyresOut', 0) or 0
        now = time.time()

        if tyres_out >= 3:
            terminated = True
            reason = 'off_track'

        if not terminated:
            try:
                current_pos = np.array(self.graphics.carCoordinates, dtype=np.float32)
                timeout_mult = self.curriculum_scheduler.get_low_speed_timeout_multiplier()
                adjusted_stuck_timeout = self._stuck_timeout * timeout_mult
                if self._position_after_reset is None:
                    self._position_after_reset = current_pos.copy()
                    self._stuck_start_time = now
                else:
                    distance = float(np.linalg.norm(current_pos - self._position_after_reset))
                    if distance <= self._stuck_threshold and (now - self._stuck_start_time) > adjusted_stuck_timeout:
                        terminated = True
                        reason = 'stuck'
            except Exception as exc:
                self.logger.warning(f"Stuck detection error: {exc}")

        if not terminated:
            try:
                current_speed = float(self.state.get('speed', 0.0) or 0.0)
                timeout_mult = self.curriculum_scheduler.get_low_speed_timeout_multiplier()
                adjusted_timeout = self._low_speed_timeout * timeout_mult
                if current_speed < self._low_speed_threshold:
                    if self._low_speed_start_time is None:
                        self._low_speed_start_time = now
                    elif (now - self._low_speed_start_time) > adjusted_timeout:
                        terminated = True
                        reason = 'low_speed'
                else:
                    self._low_speed_start_time = None
            except Exception as exc:
                self.logger.warning(f"Low-speed detection error: {exc}")

        if truncated and not terminated:
            reason = 'time_limit'
        return terminated, truncated, reason

    def step(self, action: Optional[np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.physicsConnected or not self.graphicsConnected:
            self.connect()
            return self.step(action)

        self.curriculum_scheduler.step()
        if action is not None:
            self.set_actions(action)

        self._episode_step += 1
        last_packet = self.state.get('packetID', -1)
        for _ in range(100):
            observation = self.getObservation()
            if last_packet != self.state['packetID']:
                break
        else:
            observation = self._compose_observation(self._current_core_obs)
            self.logger.warning("Timeout waiting for next physics packet")

        terminated, truncated, reason = self._check_termination()
        reward = self.getReward(terminated=terminated, truncated=truncated, termination_reason=reason)

        if self._safe_keyboard_pressed('p'):
            if not self._p_key_pressed and self._last_reward_breakdown:
                self._p_key_pressed = True
                lines = [f"  {key:30s}: {value:+8.3f}" for key, value in self._last_reward_breakdown.items()]
                print("\nREWARD BREAKDOWN:\n" + "\n".join(lines) + f"\n  {'TOTAL':30s}: {reward:+8.3f}\n")
        else:
            self._p_key_pressed = False

        info = self.getInfo()
        info.update({
            'terminated': bool(terminated),
            'termination_reason': reason,
            'line_gap': float(self._last_track_features.get('signed_distance', 0.0)),
            'heading_error': float(self._last_track_features.get('heading_error', 0.0)),
            'forward_progress': float(self.state.get('forward_progress', 0.0)),
            'off_track': bool(self.state.get('off_track', False)),
            'pedal_command': float(self.state.get('pedal_command', 0.0)),
            'applied_accel': float(self.state.get('applied_accel', 0.0)),
            'applied_brake': float(self.state.get('applied_brake', 0.0)),
            'abs_steer': float(self.state.get('abs_steer', 0.0)),
        })
        info.update({f'reward_{key}': float(value) for key, value in self._last_reward_breakdown.items()})

        self._core_obs_history.append(self._current_core_obs.copy())
        return observation, float(reward), bool(terminated), bool(truncated), info

    def set_actions(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}")

        steer = safe_clip(float(action[0]), -1.0, 1.0)
        pedal_delta = safe_clip(float(action[1]), -1.0, 1.0)
        self._current_pedal_command = safe_clip(self._current_pedal_command + pedal_delta, -1.0, 1.0)

        if self._current_pedal_command >= 0.0:
            accel = self._current_pedal_command
            brake = 0.0
        else:
            accel = 0.0
            brake = -self._current_pedal_command

        applied_action = self.controller.apply(steer=steer, accel=accel, brake=brake)
        policy_action = np.array([steer, pedal_delta], dtype=np.float32)
        self._last_applied_action = policy_action
        self._action_history.append(policy_action.copy())
        self.state['pedal_command'] = float(self._current_pedal_command)
        self.state['applied_accel'] = float(applied_action[1])
        self.state['applied_brake'] = float(applied_action[2])
        self.state['abs_steer'] = abs(float(steer))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        self.logger.info("Reset requested")
        super().reset(seed=seed)

        current_time = time.time()
        if self._last_reset_time is not None:
            time_since_last_reset = current_time - self._last_reset_time
            if time_since_last_reset < self._reset_cooldown:
                self.logger.warning(
                    "Reset spam detected! Only %.2fs since last reset (cooldown: %.2fs). Ignoring reset command.",
                    time_since_last_reset,
                    self._reset_cooldown,
                )
                return self.getObservation()
        self._last_reset_time = current_time

        try:
            self.controller.neutral()
        except Exception as exc:
            self.logger.warning(f"Could not neutralize controller before reset: {exc}")

        self.logger.info("Sending RESET command to ACReset plugin via socket...")
        if not self._send_reset_command():
            self.logger.warning("Could not send reset command. Ensure the ACReset app is enabled in Assetto Corsa.")

        time.sleep(1.0)
        self._episode_step = 0
        self._position_after_reset = None
        self._stuck_start_time = None
        self._low_speed_start_time = None
        self._prev_norm_pos = None
        self._core_obs_history.clear()
        self._action_history.clear()
        self._last_applied_action = np.zeros(self.action_dim, dtype=np.float32)
        self._current_pedal_command = 0.0
        self.state['pedal_command'] = 0.0
        self.state['applied_accel'] = 0.0
        self.state['applied_brake'] = 0.0
        self.state['abs_steer'] = 0.0
        self.controller.neutral()

        observation = self.getObservation()
        self._core_obs_history.append(self._current_core_obs.copy())
        self._prev_norm_pos = float(self.graphics.normalizedCarPosition)
        return observation

    def getReward(self, terminated: bool = False, truncated: bool = False, termination_reason: Optional[str] = None) -> float:
        speed_ms = max(float(self.physics.speedKmh) / 3.6, 0.0)
        speed_kmh = max(float(self.physics.speedKmh), 0.0)
        line_gap = abs(float(self._last_track_features.get('signed_distance', 0.0)))
        heading_error = abs(float(self._last_track_features.get('heading_error', 0.0)))
        forward_progress = speed_ms * np.cos(float(self._last_track_features.get('heading_error', 0.0))) / TOP_SPEED_MS
        off_track = 1.0 if int(self.physics.numberOfTyresOut) >= 3 else 0.0
        pedal_delta_magnitude = abs(float(self._last_applied_action[1]))
        steer_penalty = abs(float(self._last_applied_action[0]))
        low_speed_target_kmh = float(self.reward_cfg.get('low_speed_target_kmh', 25.0))
        low_speed_penalty = max(0.0, (low_speed_target_kmh - speed_kmh) / max(low_speed_target_kmh, 1e-6))

        w_progress = float(self.reward_cfg.get('w_progress', 1.0))
        w_gap = float(self.reward_cfg.get('w_gap', 0.35))
        w_heading = float(self.reward_cfg.get('w_heading', 0.15))
        w_offtrack = float(self.reward_cfg.get('w_offtrack', 2.0))
        w_terminal_stuck = float(self.reward_cfg.get('w_terminal_stuck', 3.0))
        w_overlap = float(self.reward_cfg.get('w_overlap', 0.0))
        w_low_speed = float(self.reward_cfg.get('w_low_speed', 0.20))
        w_steer = float(self.reward_cfg.get('w_steer', 0.03))
        w_steer_delta = float(self.reward_cfg.get('w_steer_delta', 0.0))
        previous_steer = float(self._action_history[-1][0]) if self._action_history else 0.0
        steer_delta_penalty = abs(float(self._last_applied_action[0]) - previous_steer)

        gap_term = line_gap / max(self.racing_line_manager.line_distance_threshold, 1e-6)
        heading_term = heading_error / np.pi
        terminal_stuck_flag = 1.0 if terminated and termination_reason in {'stuck', 'low_speed'} else 0.0

        reward = (
            w_progress * forward_progress
            - w_gap * gap_term
            - w_heading * heading_term
            - w_offtrack * off_track
            - w_terminal_stuck * terminal_stuck_flag
            - w_low_speed * low_speed_penalty
            - w_steer * steer_penalty
            - w_steer_delta * steer_delta_penalty
            - w_overlap * pedal_delta_magnitude
        )

        if np.isnan(reward) or np.isinf(reward):
            self.logger.error(f"Invalid reward detected: {reward}, clamping to -10")
            reward = -10.0

        self._last_reward_breakdown = {
            'forward_progress': float(w_progress * forward_progress),
            'line_gap_penalty': float(-w_gap * gap_term),
            'heading_penalty': float(-w_heading * heading_term),
            'off_track_penalty': float(-w_offtrack * off_track),
            'terminal_stuck_penalty': float(-w_terminal_stuck * terminal_stuck_flag),
            'low_speed_penalty': float(-w_low_speed * low_speed_penalty),
            'steer_penalty': float(-w_steer * steer_penalty),
            'steer_delta_penalty': float(-w_steer_delta * steer_delta_penalty),
            'pedal_delta_penalty': float(-w_overlap * pedal_delta_magnitude),
        }
        return float(reward)

    def close(self):
        try:
            self.controller.neutral()
        except Exception as exc:
            self.logger.warning(f"Error during close: {exc}")

    def shutdown(self):
        try:
            if self._reset_server_sock is not None:
                self._reset_server_sock.close()
                self._reset_server_sock = None
            with self._reset_client_lock:
                if self._reset_client_sock is not None:
                    self._reset_client_sock.close()
                    self._reset_client_sock = None
            self.logger.info("ACReset socket server shut down")
        except Exception as exc:
            self.logger.warning(f"Error shutting down reset server: {exc}")
        self.close()
        try:
            self.controller.close()
        except Exception:
            pass
