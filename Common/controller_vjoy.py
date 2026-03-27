from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


VJOY_SCALE = 16384
DEFAULT_DLL_CANDIDATES = (
    r"C:\Program Files\vJoy\x64\vJoyInterface.dll",
    r"C:\Program Files\vJoy\x86\vJoyInterface.dll",
)


def clamp_controls(steer: float, accel: float, brake: float) -> Tuple[float, float, float]:
    steer = float(np.clip(steer, -1.0, 1.0))
    accel = float(np.clip(accel, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    return steer, accel, brake


def convert_to_vjoy_axes(steer: float, accel: float, brake: float, scale: int = VJOY_SCALE) -> Tuple[int, int, int]:
    steer, accel, brake = clamp_controls(steer, accel, brake)
    steer_pos = int((steer + 1.0) * scale)
    accel_pos = int((accel + 1.0) * scale)
    brake_pos = int((brake + 1.0) * scale)
    return steer_pos, accel_pos, brake_pos


@dataclass
class VJoyState:
    steer: float = 0.0
    accel: float = 0.0
    brake: float = 0.0


class _VJoyDLL:
    _JOY_POS_FORMAT = "BlllllllllllllllllllIIII"

    def __init__(self, device_id: int, dll_path: str | None = None):
        self.device_id = int(device_id)
        self.dll_path = self._resolve_dll_path(dll_path)
        self.dll = ctypes.CDLL(self.dll_path)
        self.acquired = False

    @staticmethod
    def _resolve_dll_path(dll_path: str | None) -> str:
        candidates = []
        if dll_path:
            candidates.append(dll_path)
        env_path = os.environ.get("VJOY_DLL_PATH")
        if env_path:
            candidates.append(env_path)
        candidates.extend(DEFAULT_DLL_CANDIDATES)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        raise RuntimeError(
            "Could not find vJoyInterface.dll. Set controller.dll_path in config or VJOY_DLL_PATH."
        )

    def open(self) -> None:
        if not self.dll.AcquireVJD(self.device_id):
            raise RuntimeError(
                f"Failed to acquire vJoy device {self.device_id}. Check that vJoy is installed and device {self.device_id} exists."
            )
        self.acquired = True

    def close(self) -> None:
        if self.acquired:
            self.dll.RelinquishVJD(self.device_id)
            self.acquired = False

    def generate_joystick_position(self, wAxisX=0, wAxisY=0, wAxisZ=0, lButtons=0) -> bytes:
        return struct.pack(
            self._JOY_POS_FORMAT,
            self.device_id,
            0, 0, 0,
            wAxisX, wAxisY, wAxisZ,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            lButtons,
            0, 0, 0, 0,
        )

    def update(self, joystick_position: bytes) -> None:
        if not self.dll.UpdateVJD(self.device_id, joystick_position):
            raise RuntimeError(f"UpdateVJD failed for vJoy device {self.device_id}")


class VJoyController:
    def __init__(self, device_id: int = 1, scale: int = VJOY_SCALE, dll_path: str | None = None):
        self._scale = int(scale)
        self._device = _VJoyDLL(device_id=device_id, dll_path=dll_path)
        self._device.open()
        self.state = VJoyState()

    def apply(self, steer: float, accel: float, brake: float) -> np.ndarray:
        steer, accel, brake = clamp_controls(steer, accel, brake)
        axis_x, axis_y, axis_z = convert_to_vjoy_axes(steer, accel, brake, self._scale)
        joystick_position = self._device.generate_joystick_position(
            wAxisX=axis_x,
            wAxisY=axis_y,
            wAxisZ=axis_z,
            lButtons=0,
        )
        self._device.update(joystick_position)
        self.state = VJoyState(steer=steer, accel=accel, brake=brake)
        return np.array([steer, accel, brake], dtype=np.float32)

    def neutral(self) -> np.ndarray:
        return self.apply(0.0, 0.0, 0.0)

    def close(self) -> None:
        try:
            self.neutral()
        finally:
            self._device.close()
