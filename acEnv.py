from typing import Optional
from logging import Logger
from omegaconf import OmegaConf
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium import utils as gym_utils
import numpy as np
import ctypes
from ctypes import c_int32, c_float, c_wchar
import mmap
import time
import os
import keyboard
import vgamepad as vg
from sharedMemoryStructs import SPageFilePhysics, SPageFileGraphic
from curriculum_scheduler import CurriculumScheduler
from racing_line_manager import RacingLineManager
import socket

TOP_SPEED_MS = 80
MAX_EPISODE_STEPS = 5000  # ~200 seconds at 25Hz, prevents hour-long episodes

HOST = "127.0.0.1"
PORT = 65432

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
        'gas',
        'brake',
        #'fuel',
        'gear',
        'rpms',
        'steerAngle',
        'speedKmh',
        #'velocity',
        'velocityX',
        'velocityY',
        'velocityZ',
        #'accG',
        'accGX',
        'accGY',
        'accGZ',
        #'wheelSlip',
        'wheelSlipFL',
        'wheelSlipFR',
        'wheelSlipRL',
        'wheelSlipRR',
        #'wheelLoad',
        'wheelLoadFL',
        'wheelLoadFR',
        'wheelLoadRL',
        'wheelLoadRR',
        #'wheelsPressure',
        'wheelsPressureFL',
        'wheelsPressureFR',
        'wheelsPressureRL',
        'wheelsPressureRR',
        #'wheelAngularSpeed',
        'wheelAngularSpeedFL',
        'wheelAngularSpeedFR',
        'wheelAngularSpeedRL',
        'wheelAngularSpeedRR',
        #'tyreWear',
        #'tyreDirtyLevel',
        #'tyreCoreTemperature',
        'tyreCoreTemperatureFL',
        'tyreCoreTemperatureFR',
        'tyreCoreTemperatureRL',
        'tyreCoreTemperatureRR',
        #'camberRAD',
        'camberRADFL',
        'camberRADFR',
        'camberRADRL',
        'camberRADRR',
        #'suspensionTravel',
        'suspensionTravelFL',
        'suspensionTravelFR',
        'suspensionTravelRL',
        'suspensionTravelRR',
        #'drs',
        #'tc',
        'heading',
        'pitch',
        'roll',
        'cgHeight',
        #'carDamage',
        'numberOfTyresOut',
        #'pitLimiterOn',
        #'abs',
        #'kersCharge',
        #'kersInput',
        #'autoShifterOn',
        #'rideHeight',
        'rideHeightFront',
        'rideHeightRear',
        #'turboBoost',
        #'ballast',
        #'airDensity',
        #'airTemp',
        #'roadTemp',
        #'localAngularVel',
        'localAngularVelX',
        'localAngularVelY',
        'localAngularVelZ',
        #'finalFF',
        #'performanceMeter',
        #'engineBrake',
        #'ersRecoveryLevel',
        #'ersPowerLevel',
        #'ersHeatCharging',
        #'ersIsCharging',
        #'kersCurrentKJ',
        #'drsAvailable',
        #'drsEnabled',
        #'brakeTemp',
        #'clutch',
        #'tyreTempI',
        #'tyreTempM',
        #'tyreTempO',
        #'isAIControlled',
        #'tyreContactPoint',
        #'tyreContactNormal',
        #'tyreContactHeading',
        'normalizedCarPosition',
        #'carCoordinates',
        'carCoordinatesX',
        'carCoordinatesY',
        'carCoordinatesZ'
    ]

    def __init__(self, config: OmegaConf, logger: Logger):
        self.config = config
        self.logger = logger

        self._max_episode_steps = MAX_EPISODE_STEPS
        self.is_metaworld = False

        self.action_dim = 2
        # Action space: [throttle_brake, steer]
        # throttle_brake: -1 (full brake) to +1 (full gas)
        # steer: -1 (full left) to +1 (full right)
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.logger.info(f"Action Dimensions: {self.action_dim}")
        self.logger.info(f"Action Space: {self.action_space}")

        self.state_dim = len(self.observation_inputs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.logger.info("Observation Dimensions {}".format(self.state_dim))
        self.logger.info("Observation Space {}".format(self.observation_space))

        #Shared Memory Reading Maps
        self.physicsMMAP = None
        self.physics = None
        self.physicsConnected = False

        self.graphicsMMAP = None
        self.graphics = None
        self.graphicsConnected = False

        self.state = {}
        self.state["packetID"] = 0
        
        # Track last applied actions for reward calculation
        self._last_gas = 0.0
        self._last_brake = 0.0
        self._last_steer = 0.0
        
        # Track previous normalizedCarPosition for progress reward
        self._prev_norm_pos = None
        
        # Track stuck detection (no movement for 3+ seconds)
        # CRITICAL: Track from RESET position, not absolute position
        # This prevents reset regression (car going backwards on track)
        self._position_after_reset = None
        self._stuck_start_time = None
        self._stuck_threshold = 2.0  # meters - movement threshold (increased to allow reset settling)
        self._stuck_timeout = 5.0  # seconds - increased to 5s to allow time after reset
        
        # Track low-speed stuck detection (stuck against wall but moving slightly)
        self._low_speed_start_time = None
        self._low_speed_threshold = 5.0  # km/h - if below this for too long, stuck
        self._low_speed_timeout = 10.0  # seconds - stuck if slow for this long
        
        # Track previous speed for consistency bonus
        self._prev_speed = 0.0
        
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
        
        # Initialize reset toggle file (ensure it starts at "0")
        try:
            home = os.path.expanduser("~")
            toggle_file = os.path.join(home, "ac_reset_toggle.txt")
            if not os.path.exists(toggle_file):
                with open(toggle_file, 'w') as f:
                    f.write('0')
                self.logger.info(f"Initialized reset toggle file: {toggle_file}")
        except Exception as e:
            self.logger.warning(f"Could not initialize toggle file: {e}")
        
        # Debug key debounce
        self._p_key_pressed = False
        
        self.connect()

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((HOST, PORT))

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

    #Connect to shared memory buffers
    #Also connect to ac python app
    def connect(self) -> None:
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
            
            # Update racing line from Lua plugin cache if available
            if self.racing_line_manager.enabled:
                self.racing_line_manager.update_racing_line_from_lua_cache()
        except Exception as e:
            self.logger.error("Failed to read shared memory: {}".format(e))
            # Return last known observation or zeros
            return np.zeros(self.state_dim, dtype=np.float32)

        # Extract simple physics fields directly (no suffix indexing needed)
        self.state["packetID"] = self.physics.packetId
        self.state["tyresOut"] = self.physics.numberOfTyresOut
        self.state["speed"] = self.physics.speedKmh
        
        # Build normalized observation array
        observation = []
        field_mapping = {
            'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3,  # wheel indices
            'X': 0, 'Y': 1, 'Z': 2,  # vector axes
        }
        
        for input_name in self.observation_inputs:
            try:
                # Try to get from physics first
                if hasattr(self.physics, input_name):
                    value = getattr(self.physics, input_name)
                # Handle array fields with suffixes (wheelSlipFL, velocityX, etc.)
                else:
                    value = self.extractFieldValue(input_name, field_mapping)
                
                # Normalize the value
                if value is not None and input_name in self.observation_info:
                    max_val = self.observation_info[input_name]
                    normalized_value = float(value) / max_val if max_val != 0 else 0.0
                    observation.append(normalized_value)
            except Exception as e:
                self.logger.warning(f"Could not extract {input_name}: {e}")
                observation.append(0.0)
        
        return np.array(observation, dtype=np.float32)
    
    def getInfo(self) -> dict:
        return self.state

    #Await response from asseto corsa and collects current game state info
    #Returns observation, reward, terminated, truncated, info
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        if not self.physicsConnected or not self.graphicsConnected:
            self.connect()
            return self.step(action)

        # Track curriculum progress
        self.curriculum_scheduler.step()

        # Apply the action (controller input) to the game
        if action is not None:
            self.set_actions(action)

        # Wait for next physics packet with timeout
        lastPacketID = self.state["packetID"]
        nextPacketFound = False
        wait_count = 0
        max_wait = 100  # Max ~4 seconds at 25Hz

        while not nextPacketFound and wait_count < max_wait:
            observations = self.getObservation()
            nextPacketFound = lastPacketID != self.state["packetID"]
            wait_count += 1
            if wait_count >= max_wait:
                self.logger.warning("Timeout waiting for next physics packet")
                break

        info = self.getInfo()
        reward = self.getReward()
        
        # DEBUG: Press 'P' to print reward breakdown (with debounce to prevent freezing)
        if keyboard.is_pressed('p'):
            if not self._p_key_pressed:  # Only trigger on key press (transition)
                self._p_key_pressed = True
                print("\n" + "="*60)
                print("REWARD BREAKDOWN:")
                print("="*60)
                if hasattr(self, '_last_reward_breakdown'):
                    for key, value in self._last_reward_breakdown.items():
                        print(f"{key:30s}: {value:+8.2f}")
                    print("-"*60)
                    print(f"{'TOTAL REWARD':30s}: {reward[0]:+8.2f}")
                print("="*60 + "\n")
        else:
            self._p_key_pressed = False  # Reset when key is released
        
        # Termination conditions
        terminated = False
        tyres_out = self.state.get("tyresOut", 0) or 0
        
        # Check for extreme slip termination first (set in reward calc)
        if self._extreme_slip_detected:
            terminated = True
            self.logger.info("Episode terminated: extreme slip detected (unrecoverable loss of control)")
        # Severe off-track = all 4 tires off (episode ends)
        elif tyres_out >= 4:
            terminated = True
            self.logger.info("Episode terminated: all 4 tires off track")
        
        # Also terminate if 3 tires off (significant off-track)
        elif tyres_out >= 3:
            terminated = True
            self.logger.info("Episode terminated: {} tires off track".format(tyres_out))
        
        # ========== STUCK DETECTION (FIXED FOR RESET REGRESSION) ==========
        # Terminate if car hasn't moved from RESET POSITION in the timeout period
        # KEY: Track distance from position after reset, not from previous step
        # This prevents reset regression where car goes backwards on track
        try:
            current_pos = self.graphics.carCoordinates  # [x, y, z]
            current_time = time.time()
            
            if self._position_after_reset is None:
                # First step after reset - store this position as reference
                self._position_after_reset = current_pos[:]
                self._stuck_start_time = current_time
            else:
                # Calculate distance from RESET position (not from previous step)
                distance = np.linalg.norm(np.array(current_pos) - np.array(self._position_after_reset))
                
                if distance > self._stuck_threshold:
                    # Car has moved enough from reset position - not stuck
                    # Keep tracking from original reset position
                    pass
                else:
                    # Car hasn't moved much from reset position
                    stuck_time = current_time - self._stuck_start_time
                    if stuck_time > self._stuck_timeout:
                        # Car stuck for too long
                        terminated = True
                        self.logger.info("Episode terminated: car stuck (distance={:.2f}m in {:.2f}s from reset position)".format(distance, stuck_time))
        except Exception as e:
            self.logger.warning("Stuck detection error: {}".format(e))
        
        # ========== LOW-SPEED STUCK DETECTION ==========
        # Catch cases where car is stuck against wall but moving slightly
        # (e.g., pressing against barrier, wheels spinning)
        try:
            current_speed = self.state.get("speed", 0.0) or 0.0
            current_time = time.time()
            
            if current_speed < self._low_speed_threshold:
                # Car is going very slow
                if self._low_speed_start_time is None:
                    self._low_speed_start_time = current_time
                else:
                    # Check how long car has been slow
                    slow_duration = current_time - self._low_speed_start_time
                    if slow_duration > self._low_speed_timeout:
                        terminated = True
                        self.logger.info("Episode terminated: car stuck at low speed (speed={:.1f} km/h for {:.1f}s)".format(current_speed, slow_duration))
            else:
                # Car is moving at good speed, reset timer
                self._low_speed_start_time = None
        except Exception as e:
            self.logger.warning("Low-speed stuck detection error: {}".format(e))

        truncated = False

        return observations, reward, terminated, truncated, info

    #Applies action into asseto corsa
    #ActionSpace: [throttle_brake, steer]
    #  throttle_brake: -1 (full brake) to +1 (full gas)  
    #  steer: -1 (full left) to +1 (full right)
    def set_actions(self, action: np.ndarray) -> None:
        
        # Split combined throttle/brake axis
        throttle_brake = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        
        # Convert to separate gas/brake for the controller
        if throttle_brake >= 0:
            gas = throttle_brake
            brake = 0.0
        else:
            gas = 0.0
            brake = -throttle_brake
        
        # Store last applied actions for reward calculation
        self._last_gas = gas
        self._last_brake = brake
        self._last_steer = steer
        
        # Map to virtual Xbox 360 controller
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=gas)
        self.gamepad.left_trigger_float(value_float=brake)
        
        # Submit the input state to ViGEmBus
        self.gamepad.update()

    #Resets the game state via AC plugin (requires acgym plugin or compatible)
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        self.logger.info("Reseting")
        super().reset(seed=seed)

        self.srv.listen(1)
        conn, addr = self.srv.accept()
        print("Connected with ACReset")
        time.sleep(1)
        conn.sendall(b"RESET\n")
        print("Sent reset command")
        time.sleep(1)
        
        observation = self.getObservation()
        self._prev_norm_pos = self.graphics.normalizedCarPosition
        return observation
    
    def getReward(self) -> np.ndarray:
        """Reward function for RL training
        
        Reward components:
        1. FORWARD PROGRESS (DOMINANT SIGNAL) - major rewards for track progress
        2. Speed bonus (increases continuously with speed)
        3. Off-track penalty (significant deterrent)
        4. Standing still penalty (CATASTROPHIC - forces action)
        5. Steering without movement penalty (prevents wheel-spinning behavior)
        6. Slip penalty (prevents loss of control)
        7. Racing line distance penalty (curriculum learning - decays over time)
        8. Pit lane penalty
        
        KEY DESIGN: Balance progress signal with quality-of-driving signals.
        """
        
        # Store breakdown for debug printing
        breakdown = {}
        
        # Initialize extreme slip flag (will be set during slip calculation)
        self._extreme_slip_detected = False
        
        reward = 0.0
        tyres_out = self.state.get("tyresOut", 0) or 0
        
        # ========== OFF-TRACK DETECTION (SIGNIFICANT PENALTY) ==========
        # DESIGN: Off-track is bad but still recoverable
        off_track_penalty = 0.0
        if tyres_out >= 3:
            # Increased penalty - strongly discourages off-track driving
            off_track_penalty = 5.0 * tyres_out  # -15 for 3 tires, -20 for all 4
            reward -= off_track_penalty
            if tyres_out >= 4:
                self.logger.debug("[REWARD] ALL TIRES OFF! Penalty: {}".format(off_track_penalty))
        breakdown['Off-track penalty'] = -off_track_penalty
        
        # ========== TRACK PROGRESS (PRIMARY SIGNAL) ==========
        norm_pos = self.graphics.normalizedCarPosition
        progress_reward = 0.0
        
        # Validate norm_pos is a valid number
        if norm_pos is None or not isinstance(norm_pos, (int, float)) or np.isnan(norm_pos) or np.isinf(norm_pos):
            self.logger.warning("Invalid normalizedCarPosition: {}".format(norm_pos))
            norm_pos = self._prev_norm_pos if self._prev_norm_pos is not None else 0.0
        
        if self._prev_norm_pos is not None:
            # Calculate progress around track
            progress = norm_pos - self._prev_norm_pos
            
            # Handle lap wrap-around
            if progress < -0.5:  # Wrapped around (crossing line 0->1)
                progress += 1.0
                progress_reward = 50.0  # MAJOR milestone - completing a lap section
                self.logger.debug("[REWARD] Lap progress complete! +50")
            elif progress > 0.5:  # Went backwards significantly
                progress -= 1.0
            
            # Progress reward: moving forward is VERY important
            # Reduced to allow other signals to matter (efficiency, speed, line quality)
            # 0.01 progress = +2 reward
            # 0.1 progress = +20 reward
            progress_reward += progress * 200.0  # Reduced from 1000 to 200
            reward += progress_reward
        
        breakdown['Progress reward'] = progress_reward
        self._prev_norm_pos = norm_pos
        
        # ========== SPEED REWARD ==========
        # Simple linear reward based on speed (km/h)
        # In racing, faster = better, so reward should scale with speed
        speed = self.state.get("speed", 0.0) or 0.0
        speed_reward = 0.0
        
        if speed > 5:  # Moving
            # Linear reward: 50 km/h = 0.5, 100 km/h = 1.0, 200 km/h = 2.0
            speed_reward = speed / 100.0
        else:  # Standing still - but check if trying to accelerate
            # KEY FIX: Make inaction penalty conditional on gas input
            # If NOT pressing gas: catastrophic penalty (forces exploration of gas)
            # If pressing gas: much lighter penalty (encourage holding it)
            if self._last_gas < 0.3:
                # Not trying to accelerate = wasting time
                speed_reward = -100.0  # Catastrophic penalty for inaction
            else:
                # Trying to accelerate (gas >= 0.3) but not moving yet
                # Much lighter penalty - encourage the agent to keep pressing gas
                speed_reward = -30.0  # Reduced from -100
        
        self._prev_speed = speed  # Store for next frame
        reward += speed_reward
        breakdown['Speed reward'] = speed_reward
        
        # ========== THROTTLE BONUS (ENCOURAGES EXPLORATION) ==========
        # Early in training, agent needs incentive to discover that pressing gas reduces -100 penalty
        # Without this, agent learns "hold brake = stable -100" is safer than exploring
        throttle_bonus = 0.0
        if speed < 5 and self._last_gas > 0.3:
            # Reward for trying to accelerate (even if not moving yet)
            # This creates a clear path: "gas reduces penalty from -100"
            throttle_bonus = 8.0 * self._last_gas  # Up to +8 bonus for full throttle
            reward += throttle_bonus
        breakdown['Throttle bonus'] = throttle_bonus
        
        # ========== BRAKING WITHOUT MOVING PENALTY ==========
        # If agent is braking but not moving, it's wasting time (should accelerate)
        # This prevents the "hold brake" behavior and forces acceleration attempts
        braking_penalty = 0.0
        if speed < 10 and self._last_brake > 0.3:
            # Braking while not moving = wasting time, force acceleration
            braking_penalty = 10.0 * self._last_brake  # Harsh penalty for brake holding
            reward -= braking_penalty
        breakdown['Braking w/o movement'] = -braking_penalty
        
        # ========== RACING LINE DISTANCE PENALTY (CURRICULUM LEARNING) ==========
        # Penalize distance from racing line, with reward decaying over training
        # DESIGN: Heavy guidance early, gradually removed as policy improves
        racing_line_penalty = 0.0
        racing_line_distance = 0.0  # Track for debug output
        try:
            rl_cfg = self.config.get('racing_line', {})
            if rl_cfg.get('enable', True):
                # Get current racing line weight from curriculum
                rl_weight = self.curriculum_scheduler.get_racing_line_weight()
                
                # Diagnostic: Log status once at start
                if not hasattr(self, '_logged_rl_diagnostic'):
                    self.logger.info(f"[RACING_LINE_DIAGNOSTIC]")
                    self.logger.info(f"  Racing line enabled: True")
                    self.logger.info(f"  Racing line loaded: {self.racing_line_manager.racing_line_loaded}")
                    self.logger.info(f"  Curriculum weight (should be ~1.0 at start): {rl_weight:.2f}")
                    self.logger.info(f"  Config check: enable={rl_cfg.get('enable', True)}")
                    if self.racing_line_manager.racing_line_loaded:
                        self.logger.info(f"  Racing line points: {len(self.racing_line_manager.racing_line_points)}")
                    self._logged_rl_diagnostic = True
                
                if rl_weight > 0.0 and self.racing_line_manager.racing_line_loaded:
                    car_pos = (self.graphics.carCoordinates[0], 
                              self.graphics.carCoordinates[1], 
                              self.graphics.carCoordinates[2])
                    
                    racing_line_distance = self.racing_line_manager.distance_to_racing_line(car_pos)
                    
                    # Convert distance to normalized penalty (0 to 1)
                    # Maximum penalty at RL threshold distance
                    rl_threshold = self.racing_line_manager.line_distance_threshold
                    normalized_distance = min(racing_line_distance / rl_threshold, 1.0)
                    
                    # Apply curriculum weight - starts high, decays to 0
                    line_weight = self.racing_line_manager.line_distance_weight
                    racing_line_penalty = line_weight * normalized_distance * rl_weight
                    reward -= racing_line_penalty
                    
                    # Periodic diagnostic: log actual distance every 100 frames
                    if not hasattr(self, '_rl_debug_counter'):
                        self._rl_debug_counter = 0
                    self._rl_debug_counter += 1
                    if self._rl_debug_counter % 100 == 0:
                        self.logger.debug(f"[RL_DISTANCE] Distance to line: {racing_line_distance:.2f}m, "
                                        f"Penalty: {racing_line_penalty:.2f}, Weight: {rl_weight:.2f}")
                elif not self.racing_line_manager.racing_line_loaded:
                    # Log why racing line is not active
                    if not hasattr(self, '_logged_rl_missing'):
                        self.logger.warning(f"[RACING_LINE] Not loaded yet - waiting for Lua cache file")
                        self.logger.warning(f"  Cache file path: {os.path.expanduser('~')}/ac_racing_line_cache.bin")
                        self.logger.warning(f"  Ensure CSP_Reset script is ENABLED in Content Manager")
                        self._logged_rl_missing = True
        except Exception as e:
            self.logger.debug(f"Racing line penalty calculation error: {e}")
        
        breakdown['Racing line penalty'] = -racing_line_penalty
        breakdown['Racing line distance'] = racing_line_distance  # Add distance to breakdown
        
        # ========== SLIP ANGLE & WHEELSPIN PENALTY ==========
        # Penalize car for losing traction (high slip) or wheelspin (spinning wheels vs forward motion)
        # This prevents learning unstable/drifting driving behaviors
        slip_penalty = 0.0
        try:
            # Get wheel slip values (0-1 scale, higher = more slip)
            wheel_slips = [
                abs(self.state.get('wheelSlipFL', 0.0) or 0.0),
                abs(self.state.get('wheelSlipFR', 0.0) or 0.0),
                abs(self.state.get('wheelSlipRL', 0.0) or 0.0),
                abs(self.state.get('wheelSlipRR', 0.0) or 0.0),
            ]
            max_wheel_slip = max(wheel_slips) if wheel_slips else 0.0
            
            # EXTREME slip = terminate episode (unrecoverable loss of control)
            if max_wheel_slip > 0.8:
                self._extreme_slip_detected = True
                slip_penalty = 50.0
            # Penalize excessive slip (>0.2 = significant loss of traction)
            elif max_wheel_slip > 0.2:
                slip_penalty = 10.0 * (max_wheel_slip - 0.2)  # scales with severity
            
            reward -= slip_penalty
        except Exception as e:
            self.logger.debug(f"Slip penalty calculation error: {e}")
        
        breakdown['Slip penalty'] = -slip_penalty
        
        # ========== REMOVED: Smooth driving penalties ==========
        # REASON: These discouraged the agent from taking any action.
        # The agent would rather hold brakes (no input changes = no penalties)
        # than risk acceleration or steering changes.
        # Let the agent learn smoothness naturally from lap times and off-track penalties.
        
        # ========== PIT LANE PENALTY ==========
        pit_penalty = 0.0
        if self.graphics.isInPit:
            pit_penalty = 1.0
            reward -= pit_penalty
        breakdown['Pit lane penalty'] = -pit_penalty
        
        # Safety: Clip reward to prevent NaN/inf
        if np.isnan(reward) or np.isinf(reward):
            self.logger.error("Invalid reward detected: {}, setting to -10".format(reward))
            reward = -10.0
        
        # Store breakdown for debug printing (press 'P' during training)
        self._last_reward_breakdown = breakdown
        
        return np.array([reward], dtype=np.float32)
    
    # On exit - cleanup resources
    def close(self):
        try:
            # Set controller to neutral before closing
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            self.gamepad.update()
            self.logger.info("ACEnv closed, controller set to neutral")
        except Exception as e:
            self.logger.warning("Error during close: {}".format(e))
        return