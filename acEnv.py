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
import keyboard
import vgamepad as vg
from sharedMemoryStructs import SPageFilePhysics, SPageFileGraphic

TOP_SPEED_MS = 80
MAX_EPISODE_STEPS = 100000

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

        self.action_dim = 3
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]))
        
        self.logger.info(f"Action Dimensions: {self.action_dim}")
        self.logger.info(f"Action Space: {self.action_space}")

        # self.state_dim = len(self.observation_inputs)
        self.state_dim = 53
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.logger.info(f"Observation Dimensions {self.state_dim}")
        self.logger.info(f"Observation Space {self.observation_space}")

        #Shared Memory Reading Maps
        self.physicsMMAP = None
        self.physics = None
        self.physicsConnected = False

        self.graphicsMMAP = None
        self.graphics = None
        self.graphicsConnected = False

        self.state = {}
        self.state["packetID"] = 0
        
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
        
        self.connect()

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
        self.physics = SPageFilePhysics.from_buffer_copy(self.physicsMMAP)
        self.graphics = SPageFileGraphic.from_buffer_copy(self.graphicsMMAP)

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
        
        return np.array(observation, dtype=np.float32)[1:]
    
    def getInfo(self) -> dict:
        return self.state

    #Await response from asseto corsa and collects current game state info
    #Returns observation, reward, terminated, truncated, info
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        if not self.physicsConnected or not self.graphicsConnected:
            self.connect()
            return self.step(action)

        lastPacketID = self.state["packetID"]
        nextPacketFound = False

        while not nextPacketFound:
            observations = self.getObservation()
            nextPacketFound = lastPacketID != self.state["packetID"]

        info = self.getInfo()
        reward = self.getReward()
        
        terminated = False
        if self.state["tyresOut"] != None:
            terminated = self.state["tyresOut"] > 2
            if terminated:
                self.logger.info(f"Terminating")

        truncated = False

        return observations, reward, terminated, info

    #Applies action into asseto corsa
    #ActionSpace:<gas, brake, steer> 
    #ActionSpaceRanges: <0 - 1, 0 - 1, -1 - 1>
    def set_actions(self, action: np.ndarray) -> None:
        
        # Extract actions: [gas, brake, steer]
        gas = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        steer = float(np.clip(action[2], -1.0, 1.0))
        
        # Map to virtual Xbox 360 controller
        # Gas -> Right trigger (0.0 to 1.0)
        # Brake -> Left trigger (0.0 to 1.0)
        # Steer -> Left joystick X-axis (-1.0 to 1.0)
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=gas)
        self.gamepad.left_trigger_float(value_float=brake)
        
        # Submit the input state to ViGEmBus
        self.gamepad.update()

    #Resets the game state (ctrl + r) + enter (when settings up make sure to already hover drive in the pit menu)
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        self.logger.info(f"Reseting")
        super().reset(seed=seed)
        
        # Simulate Ctrl+R to reset Assetto Corsa
        self.logger.info("Sending Ctrl+R to reset game")
        keyboard.press('ctrl')
        time.sleep(0.05)
        keyboard.press('r')
        time.sleep(0.05)
        keyboard.release('r')
        time.sleep(0.05)
        keyboard.release('ctrl')
        time.sleep(0.2)
        # Press Enter
        self.logger.info("Sending Enter to confirm reset")
        keyboard.press('enter')
        time.sleep(0.05)
        keyboard.release('enter')
        time.sleep(0.5)
        
        observation = self.getObservation()
        info = self.getInfo()

        return observation
    
    def getReward(self) -> np.ndarray:
        if self.state["tyresOut"] != None:
            if self.state["tyresOut"] > 2:
                return -1
        
        if self.state["speed"] != None:
            speedReward = 3.6 * self.state["speed"] / 300
            return np.array([speedReward], dtype=np.float32)
    
        return np.array([0], dtype=np.float32)
    
    # On exit
    def close(self):
        return