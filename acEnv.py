from typing import Optional
from logging import Logger
from omegaconf import OmegaConf
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium import utils as gym_utils
import numpy as np
import ctypes
import mmap
import time
import os
import socket
import threading
import keyboard
import vgamepad as vg
from sharedMemoryStructs import SPageFilePhysics, SPageFileGraphic
from curriculum_scheduler import CurriculumScheduler
from racing_line_manager import RacingLineManager

TOP_SPEED_MS = 80
MAX_EPISODE_STEPS = 5000  # ~200 seconds at 25Hz, prevents hour-long episodes

class ACEnv(Env, gym_utils.EzPickle):
    observation_info = {
        'gas': 1.0,
        'brake': 1.0,
        'gear': 6.,
        'rpms': 10000.,
        'steerAngle': 450,
        'speedKmh': TOP_SPEED_MS * 3.6,
        'velocityX': TOP_SPEED_MS,
        'velocityY': 20.,
        'velocityZ': 5.,
        'accGX': 5.,
        'accGY': 5.,
        'accGZ': 5.,
        'wheelSlipFL': 1.0,
        'wheelSlipFR': 1.0,
        'wheelSlipRL': 1.0,
        'wheelSlipRR': 1.0,
        'wheelLoadFL': 10000.,
        'wheelLoadFR': 10000.,
        'wheelLoadRL': 10000.,
        'wheelLoadRR': 10000.,
        'wheelsPressureFL': 30.,
        'wheelsPressureFR': 30.,
        'wheelsPressureRL': 30.,
        'wheelsPressureRR': 30.,
        'wheelAngularSpeedFL': TOP_SPEED_MS / 3.6,
        'wheelAngularSpeedFR': TOP_SPEED_MS / 3.6,
        'wheelAngularSpeedRL': TOP_SPEED_MS / 3.6,
        'wheelAngularSpeedRR': TOP_SPEED_MS / 3.6,
        'tyreCoreTemperatureFL': 100.,
        'tyreCoreTemperatureFR': 100.,
        'tyreCoreTemperatureRL': 100.,
        'tyreCoreTemperatureRR': 100.,
        'camberRADFL': 0.5,
        'camberRADFR': 0.5,
        'camberRADRL': 0.5,
        'camberRADRR': 0.5,
        'suspensionTravelFL': 0.1,
        'suspensionTravelFR': 0.1,
        'suspensionTravelRL': 0.1,
        'suspensionTravelRR': 0.1,
        'heading': 2 * np.pi,
        'pitch': np.pi / 2,
        'roll': np.pi / 2,
        'cgHeight': 0.5,
        'numberOfTyresOut': 4.,
        'rideHeightFront': 0.1,
        'rideHeightRear': 0.1,
        'localAngularVelX': np.pi,
        'localAngularVelY': np.pi,
        'localAngularVelZ': np.pi,
        'normalizedCarPosition': 1.0,
        'carCoordinatesX': 5000.,
        'carCoordinatesY': 5000.,
        'carCoordinatesZ': 500.,
    }

    observation_inputs = [
        'gas', 'brake', 'gear', 'rpms', 'steerAngle', 'speedKmh',
        'velocityX', 'velocityY', 'velocityZ',
        'accGX', 'accGY', 'accGZ',
        'wheelSlipFL', 'wheelSlipFR', 'wheelSlipRL', 'wheelSlipRR',
        'wheelLoadFL', 'wheelLoadFR', 'wheelLoadRL', 'wheelLoadRR',
        'wheelsPressureFL', 'wheelsPressureFR', 'wheelsPressureRL', 'wheelsPressureRR',
        'wheelAngularSpeedFL', 'wheelAngularSpeedFR', 'wheelAngularSpeedRL', 'wheelAngularSpeedRR',
        'tyreCoreTemperatureFL', 'tyreCoreTemperatureFR', 'tyreCoreTemperatureRL', 'tyreCoreTemperatureRR',
        'camberRADFL', 'camberRADFR', 'camberRADRL', 'camberRADRR',
        'suspensionTravelFL', 'suspensionTravelFR', 'suspensionTravelRL', 'suspensionTravelRR',
        'heading', 'pitch', 'roll', 'cgHeight',
        'numberOfTyresOut',
        'rideHeightFront', 'rideHeightRear',
        'localAngularVelX', 'localAngularVelY', 'localAngularVelZ',
        'normalizedCarPosition',
        'carCoordinatesX', 'carCoordinatesY', 'carCoordinatesZ',
    ]

    # Suffix-to-index mapping for array fields (class-level constant)
    _field_mapping = {
        'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3,
        'X': 0, 'Y': 1, 'Z': 2,
    }

    def __init__(self, config: OmegaConf, logger: Logger):
        self.config = config
        self.logger = logger

        self._max_episode_steps = MAX_EPISODE_STEPS

        self.action_dim = 2
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.state_dim = len(self.observation_inputs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.logger.info(f"Action dim={self.action_dim}  Obs dim={self.state_dim}")

        # Shared memory handles
        self.physicsMMAP = None
        self.physics = None
        self.physicsConnected = False
        self.graphicsMMAP = None
        self.graphics = None
        self.graphicsConnected = False

        self.state = {"packetID": 0}

        # Last applied actions (for reward calculation)
        self._last_gas = 0.0
        self._last_brake = 0.0
        self._last_steer = 0.0

        # Progress tracking
        self._prev_norm_pos = None

        # Stuck detection: distance from reset position
        self._position_after_reset = None
        self._stuck_start_time = None
        self._stuck_threshold = 2.0   # meters
        self._stuck_timeout = 5.0     # seconds (scaled by curriculum multiplier)

        # Low-speed stuck detection
        self._low_speed_start_time = None
        self._low_speed_threshold = 5.0   # km/h
        self._low_speed_timeout = 5.0     # seconds (scaled by curriculum multiplier)

        # Extreme slip flag (set by getReward)
        self._extreme_slip_detected = False

        # Reset cooldown
        self._last_reset_time = None
        self._reset_cooldown = 2.0  # seconds
        
        # Initialize curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(config, logger)
        
        # Initialize racing line manager
        self.racing_line_manager = RacingLineManager(config, logger)
        
        # Initialize virtual Xbox 360 controller
        self.gamepad = vg.VX360Gamepad()
        self.logger.info("Virtual Xbox 360 controller initialized")
        
        # Set controller to neutral state
        try:
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            self.gamepad.update()
            self.logger.info("Controller set to neutral state")
        except Exception as e:
            self.logger.warning(f"Could not initialize controller state: {e}")
        
        # ===== ACReset Plugin Socket Server =====
        # The ACReset plugin (running inside AC) connects to this server as a TCP client.
        # When we need to reset, we send "RESET\n" over the socket and the plugin
        # calls ac.ext_resetCar().
        self._reset_host = "127.0.0.1"
        self._reset_port = 65432
        self._reset_client_sock = None       # connected ACReset plugin socket
        self._reset_client_lock = threading.Lock()
        self._reset_server_sock = None
        self._reset_server_thread = None
        self._start_reset_server()
        
        # Debug key debounce
        self._p_key_pressed = False
        
        self.connect()

    # ---------- ACReset socket server ----------
    def _start_reset_server(self):
        """Start a TCP server that the ACReset plugin connects to.
        
        The ACReset plugin (ACResetClient) connects to 127.0.0.1:65432 as a
        non-blocking TCP client and polls for newline-delimited commands.
        When it receives 'RESET', it calls ac.ext_resetCar().
        """
        try:
            self._reset_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._reset_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._reset_server_sock.bind((self._reset_host, self._reset_port))
            self._reset_server_sock.listen(1)
            self._reset_server_sock.settimeout(1.0)  # allow periodic shutdown checks

            self._reset_server_thread = threading.Thread(
                target=self._reset_server_loop, daemon=True
            )
            self._reset_server_thread.start()
            self.logger.info(
                f"ACReset server listening on {self._reset_host}:{self._reset_port}"
            )
        except Exception as e:
            self.logger.error(f"Failed to start ACReset server: {e}")

    def _reset_server_loop(self):
        """Accept loop running in a daemon thread.
        
        Accepts one client at a time (the ACReset plugin).  If the plugin
        disconnects and reconnects, the new connection replaces the old one.
        """
        while True:
            try:
                client_sock, addr = self._reset_server_sock.accept()
                with self._reset_client_lock:
                    # Close any previous connection
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
                # Server socket closed (during shutdown)
                break

    def _send_reset_command(self, retries: int = 10, retry_interval: float = 0.5) -> bool:
        """Send 'RESET\\n' to the connected ACReset plugin.
        
        Retries up to `retries` times (default 10 = 5 seconds) in case the
        plugin hasn't connected yet.
        Returns True if the command was sent successfully.
        """
        for attempt in range(retries):
            with self._reset_client_lock:
                if self._reset_client_sock is not None:
                    try:
                        self._reset_client_sock.sendall(b"RESET\n")
                        self.logger.info("Sent RESET command to ACReset plugin")
                        return True
                    except (BrokenPipeError, ConnectionResetError, OSError) as e:
                        self.logger.warning(f"ACReset plugin connection lost: {e}")
                        try:
                            self._reset_client_sock.close()
                        except Exception:
                            pass
                        self._reset_client_sock = None

            # Plugin not connected - wait and retry
            if attempt < retries - 1:
                self.logger.debug(
                    f"ACReset plugin not connected, retrying in {retry_interval}s "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(retry_interval)

        self.logger.warning(
            "ACReset plugin did not connect after {:.1f}s. "
            "Make sure the ACReset app is enabled in Assetto Corsa.".format(
                retries * retry_interval
            )
        )
        return False

    # algo wants this
    def seed(self, seed=None):
        pass

    def extractFieldValue(self, field_name: str, field_mapping: dict):
        """Extract field value from physics/graphics using suffix-based indexing"""
        # Check physics array fields with wheel suffixes
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
                # Try physics first
                if hasattr(self.physics, base_name):
                    return getattr(self.physics, base_name)[idx]
                # Then try graphics
                if hasattr(self.graphics, base_name):
                    return getattr(self.graphics, base_name)[idx]
        
        # Handle special cases like rideHeightFront/Rear
        if field_name.startswith('rideHeight'):
            idx = 0 if 'Front' in field_name else 1
            return self.physics.rideHeight[idx]
        
        # Try graphics fields
        if hasattr(self.graphics, field_name):
            return getattr(self.graphics, field_name)
        
        return None

    def connect(self) -> None:
        """Connect to AC shared memory buffers."""
        try:
            self.physicsMMAP = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), "acpmf_physics")
            self.logger.info("Connected to physics shared memory")
            self.physicsConnected = True
        except Exception as e:
            self.logger.info(f"Could not connect to physics shared memory: {e}")
            raise
        try:
            self.graphicsMMAP = mmap.mmap(0, ctypes.sizeof(SPageFileGraphic), "acpmf_graphics")
            self.logger.info("Connected to graphics shared memory")
            self.graphicsConnected = True
        except Exception as e:
            self.logger.info(f"Could not connect to graphics shared memory: {e}")
            raise

    def getObservation(self) -> np.ndarray:
        try:
            self.physics = SPageFilePhysics.from_buffer_copy(self.physicsMMAP)
            self.graphics = SPageFileGraphic.from_buffer_copy(self.graphicsMMAP)
        except Exception as e:
            self.logger.error(f"Failed to read shared memory: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

        self.state["packetID"] = self.physics.packetId
        self.state["tyresOut"] = self.physics.numberOfTyresOut
        self.state["speed"] = self.physics.speedKmh

        observation = np.empty(self.state_dim, dtype=np.float32)
        for i, input_name in enumerate(self.observation_inputs):
            try:
                if hasattr(self.physics, input_name):
                    value = getattr(self.physics, input_name)
                else:
                    value = self.extractFieldValue(input_name, self._field_mapping)

                if value is not None and input_name in self.observation_info:
                    max_val = self.observation_info[input_name]
                    observation[i] = float(value) / max_val if max_val != 0 else 0.0
                else:
                    observation[i] = 0.0
            except Exception as e:
                self.logger.warning(f"Could not extract {input_name}: {e}")
                observation[i] = 0.0

        return observation
    
    def getInfo(self) -> dict:
        return self.state

    #Await response from asseto corsa and collects current game state info
    #Returns observation, reward, terminated, truncated, info
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        if not self.physicsConnected or not self.graphicsConnected:
            self.connect()
            return self.step(action)

        self.curriculum_scheduler.step()

        if action is not None:
            self.set_actions(action)

        # Wait for next physics packet
        last_packet = self.state["packetID"]
        for _ in range(100):
            observations = self.getObservation()
            if last_packet != self.state["packetID"]:
                break
        else:
            self.logger.warning("Timeout waiting for next physics packet")

        info = self.getInfo()
        reward = self.getReward()

        # Debug: press 'P' to print reward breakdown
        if keyboard.is_pressed('p'):
            if not self._p_key_pressed:
                self._p_key_pressed = True
                if hasattr(self, '_last_reward_breakdown'):
                    lines = [f"  {k:30s}: {v:+8.2f}" for k, v in self._last_reward_breakdown.items()]
                    print("\nREWARD BREAKDOWN:\n" + "\n".join(lines) + f"\n  {'TOTAL':30s}: {reward[0]:+8.2f}\n")
        else:
            self._p_key_pressed = False

        # --- Termination checks ---
        terminated = False
        tyres_out = self.state.get("tyresOut", 0) or 0
        now = time.time()

        if tyres_out >= 3:
            terminated = True
            self.logger.info(f"Episode terminated: {tyres_out} tires off track")

        # Stuck detection: distance from reset position (curriculum-scaled)
        if not terminated:
            try:
                current_pos = self.graphics.carCoordinates
                timeout_mult = self.curriculum_scheduler.get_low_speed_timeout_multiplier()
                adjusted_stuck_timeout = self._stuck_timeout * timeout_mult
                if self._position_after_reset is None:
                    self._position_after_reset = current_pos[:]
                    self._stuck_start_time = now
                else:
                    dist = np.linalg.norm(np.array(current_pos) - np.array(self._position_after_reset))
                    if dist <= self._stuck_threshold and (now - self._stuck_start_time) > adjusted_stuck_timeout:
                        terminated = True
                        self.logger.info(f"Episode terminated: stuck ({dist:.2f}m in {now - self._stuck_start_time:.1f}s, timeout={adjusted_stuck_timeout:.1f}s)")
            except Exception as e:
                self.logger.warning(f"Stuck detection error: {e}")

        # Low-speed stuck detection (uses curriculum-based timeout multiplier)
        if not terminated:
            try:
                current_speed = self.state.get("speed", 0.0) or 0.0
                # Get curriculum multiplier (high at start, decays to 1.0)
                timeout_mult = self.curriculum_scheduler.get_low_speed_timeout_multiplier()
                adjusted_timeout = self._low_speed_timeout * timeout_mult
                
                if current_speed < self._low_speed_threshold:
                    if self._low_speed_start_time is None:
                        self._low_speed_start_time = now
                    elif (now - self._low_speed_start_time) > adjusted_timeout:
                        terminated = True
                        self.logger.info(f"Episode terminated: low speed ({current_speed:.1f} km/h for {now - self._low_speed_start_time:.1f}s, timeout={adjusted_timeout:.1f}s)")
                else:
                    self._low_speed_start_time = None
            except Exception as e:
                self.logger.warning(f"Low-speed detection error: {e}")

        return observations, reward, terminated, False, info

    def set_actions(self, action: np.ndarray) -> None:
        """Apply [throttle_brake, steer] action to the virtual Xbox 360 controller."""
        throttle_brake = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        if throttle_brake >= 0:
            gas, brake = throttle_brake, 0.0
        else:
            gas, brake = 0.0, -throttle_brake

        self._last_gas = gas
        self._last_brake = brake
        self._last_steer = steer

        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=gas)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.update()

    #Resets the game state via ACReset plugin over TCP socket
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        self.logger.info("Reset requested")
        super().reset(seed=seed)

        # Check reset cooldown to prevent spam resets
        current_time = time.time()
        if self._last_reset_time is not None:
            time_since_last_reset = current_time - self._last_reset_time
            if time_since_last_reset < self._reset_cooldown:
                self.logger.warning(f"Reset spam detected! Only {time_since_last_reset:.2f}s since last reset (cooldown: {self._reset_cooldown}s). Ignoring reset command.")
                observation = self.getObservation()
                return observation
        
        self._last_reset_time = current_time

        # Set controller to neutral before reset so the car doesn't keep driving
        try:
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            self.gamepad.update()
        except Exception as e:
            self.logger.warning(f"Could not neutralize controller before reset: {e}")

        # Send RESET command to the ACReset plugin via TCP socket
        self.logger.info("Sending RESET command to ACReset plugin via socket...")
        reset_sent = self._send_reset_command()
        
        if not reset_sent:
            self.logger.warning(
                "Could not send reset command. "
                "Ensure the ACReset app is enabled in Assetto Corsa and has connected."
            )
        
        # Give AC a moment to execute ac.ext_resetCar() and stabilize
        time.sleep(1.0)
        
        # Reset episode tracking state
        self._position_after_reset = None
        self._stuck_start_time = None
        self._low_speed_start_time = None
        self._extreme_slip_detected = False
        
        observation = self.getObservation()
        self._prev_norm_pos = self.graphics.normalizedCarPosition
        return observation
    
    def getReward(self) -> np.ndarray:
        """Compute reward for the current step.
        
        Simplified reward matching assetto_corsa_gym model:
        Base reward is speed-normalized, with optional racing line guidance.
        
        Racing line acts as a multiplier on speed reward (curriculum-based decay).
        
        Commented out components from original model:
        - Track progress reward (lap position)
        - Speed conditional penalties (idle/stalling)
        - Throttle bonus
        - Braking while stopped penalty
        - Pit lane penalty
        - Off-track penalty (via lap termination instead)
        """
        breakdown = {}
        self._extreme_slip_detected = False
        
        # --- Base speed reward (primary signal) ---
        # Matches assetto_corsa_gym: speed_kmh / 300
        # Note: self.physics.speedKmh is already in km/h, no 3.6x conversion needed
        # (assetto_corsa_gym multiplies by 3.6 because their speed is in m/s)
        speed = self.state.get("speed", 0.0) or 0.0
        speed_normalized = speed / 300.0
        reward = speed_normalized
        breakdown['Speed (km/h ÷ 300)'] = speed_normalized
        
        # --- Racing line distance reward (optional, curriculum-based) ---
        # Similar to assetto_corsa_gym's gap-based multiplier
        racing_line_penalty = 0.0
        
        try:
            rl_cfg = self.config.get('racing_line', {})
            if rl_cfg.get('enable', True) and self.racing_line_manager.racing_line_loaded:
                rl_weight = self.curriculum_scheduler.get_racing_line_weight()

                if not hasattr(self, '_logged_rl_diagnostic'):
                    self.logger.info(
                        f"Racing line: loaded=True, curriculum_weight={rl_weight:.2f}"
                    )
                    self._logged_rl_diagnostic = True

                if rl_weight > 0.0:
                    # AC shared memory carCoordinates = [x, y, z] where Y is vertical
                    # Racing line CSV pos_x/pos_y map to AC's (z, x) ground plane
                    # So car ground position = (carCoordinates[2], carCoordinates[0])
                    car_pos = (self.graphics.carCoordinates[2],
                               self.graphics.carCoordinates[0],
                               0.0)
                    racing_line_distance = self.racing_line_manager.distance_to_racing_line(car_pos)
                    rl_threshold = self.racing_line_manager.line_distance_threshold
                    
                    # Matches assetto_corsa_gym: r *= (1.0 - abs(gap) / gap_const)
                    normalized_distance = min(abs(racing_line_distance) / rl_threshold, 1.0)
                    
                    # Apply curriculum-weighted penalty: as weight increases, penalty increases
                    line_multiplier = 1.0 - (normalized_distance * rl_weight)
                    reward *= line_multiplier
                    
                    # Track penalty applied (for breakdown only)
                    racing_line_penalty = speed_normalized * (1.0 - line_multiplier)
                    breakdown['Racing line penalty'] = -racing_line_penalty
            elif rl_cfg.get('enable', True) and not self.racing_line_manager.racing_line_loaded:
                if not hasattr(self, '_logged_rl_missing'):
                    self.logger.warning("Racing line not loaded - check Racing Lines CSV files")
                    self._logged_rl_missing = True
        except Exception as e:
            self.logger.debug(f"Racing line penalty error: {e}")

        breakdown['Total reward'] = reward

        # ===== COMMENTED OUT COMPONENTS FROM ORIGINAL MODEL =====
        
        # --- Off-track penalty (commented out - termination handles this) ---
        # tyres_out = self.state.get("tyresOut", 0) or 0
        # off_track_penalty = 0.0
        # if tyres_out >= 3:
        #     off_track_penalty = 5.0 * tyres_out
        #     reward -= off_track_penalty
        # breakdown['Off-track penalty'] = -off_track_penalty

        # --- Track progress (commented out - now handled by speed reward) ---
        # norm_pos = self.graphics.normalizedCarPosition
        # progress_reward = 0.0
        # 
        # # Validate norm_pos is a valid number
        # if norm_pos is None or not isinstance(norm_pos, (int, float)) or np.isnan(norm_pos) or np.isinf(norm_pos):
        #     self.logger.warning("Invalid normalizedCarPosition: {}".format(norm_pos))
        #     norm_pos = self._prev_norm_pos if self._prev_norm_pos is not None else 0.0
        # 
        # if self._prev_norm_pos is not None:
        #     progress = norm_pos - self._prev_norm_pos
        #
        #     if progress < -0.5:        # Lap wrap-around
        #         progress += 1.0
        #         progress_reward = 50.0
        #     elif progress > 0.5:       # Went backwards significantly
        #         progress -= 1.0
        #
        #     progress_reward += progress * 200.0
        #     reward += progress_reward
        #
        # breakdown['Progress reward'] = progress_reward
        # self._prev_norm_pos = norm_pos

        # --- Conditional speed penalties (commented out) ---
        # speed_reward = 0.0
        # if speed > 5:
        #     speed_reward = speed / 100.0
        # elif self._last_gas < 0.3:
        #     speed_reward = -100.0     # Not pressing gas = wasting time
        # else:
        #     speed_reward = -30.0      # Pressing gas but not moving yet
        # reward += speed_reward
        # breakdown['Speed reward'] = speed_reward

        # --- Throttle bonus (commented out) ---
        # throttle_bonus = 0.0
        # if speed < 5 and self._last_gas > 0.3:
        #     throttle_bonus = 8.0 * self._last_gas
        #     reward += throttle_bonus
        # breakdown['Throttle bonus'] = throttle_bonus

        # --- Braking while stopped penalty (commented out) ---
        # braking_penalty = 0.0
        # if speed < 10 and self._last_brake > 0.3:
        #     braking_penalty = 10.0 * self._last_brake
        #     reward -= braking_penalty
        # breakdown['Braking w/o movement'] = -braking_penalty

        # --- Pit lane penalty (commented out) ---
        # pit_penalty = 0.0
        # if self.graphics.isInPit:
        #     pit_penalty = 1.0
        #     reward -= pit_penalty
        # breakdown['Pit lane penalty'] = -pit_penalty

        if np.isnan(reward) or np.isinf(reward):
            self.logger.error(f"Invalid reward detected: {reward}, clamping to -10")
            reward = -10.0

        self._last_reward_breakdown = breakdown
        return np.array([reward], dtype=np.float32)

    # Per-episode cleanup (Agent calls this after every episode).
    def close(self):
        try:
            # Set controller to neutral between episodes
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            self.gamepad.update()
        except Exception as e:
            self.logger.warning("Error during close: {}".format(e))
        return

    # Final shutdown — call once when training is completely done
    def shutdown(self):
        """Tear down the socket server and all resources.  Call once at exit."""
        try:
            if self._reset_server_sock is not None:
                self._reset_server_sock.close()
                self._reset_server_sock = None
            with self._reset_client_lock:
                if self._reset_client_sock is not None:
                    self._reset_client_sock.close()
                    self._reset_client_sock = None
            self.logger.info("ACReset socket server shut down")
        except Exception as e:
            self.logger.warning(f"Error shutting down reset server: {e}")
        self.close()
        return