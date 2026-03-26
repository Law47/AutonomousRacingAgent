from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


VJOY_SCALE = 16384


def clamp_controls(steer: float, accel: float, brake: float) -> Tuple[float, float, float]:
    steer = float(np.clip(steer, -1.0, 1.0))
    accel = float(np.clip(accel, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    return steer, accel, brake


def convert_to_vjoy_axes(steer: float, accel: float, brake: float, scale: int = VJOY_SCALE) -> Tuple[int, int, int]:
    steer, accel, brake = clamp_controls(steer, accel, brake)

    steer_pos = int((steer + 1.0) * scale)
    accel_pos = int(accel * 2.0 * scale)
    brake_pos = int(brake * 2.0 * scale)
    return steer_pos, accel_pos, brake_pos


@dataclass
class VJoyState:
    steer: float = 0.0
    accel: float = 0.0
    brake: float = 0.0


class VJoyController:
    def __init__(self, device_id: int = 1, scale: int = VJOY_SCALE):
        try:
            import pyvjoy
        except ImportError as exc:
            raise RuntimeError(
                "pyvjoy is required for the vJoy controller backend. Install vJoy and the pyvjoy package."
            ) from exc

        self._pyvjoy = pyvjoy
        self._scale = scale
        self._device = pyvjoy.VJoyDevice(device_id)
        self.state = VJoyState()

    def apply(self, steer: float, accel: float, brake: float) -> np.ndarray:
        steer, accel, brake = clamp_controls(steer, accel, brake)
        axis_x, axis_y, axis_z = convert_to_vjoy_axes(steer, accel, brake, self._scale)
        self._device.set_axis(self._pyvjoy.HID_USAGE_X, axis_x)
        self._device.set_axis(self._pyvjoy.HID_USAGE_Y, axis_y)
        self._device.set_axis(self._pyvjoy.HID_USAGE_Z, axis_z)
        self.state = VJoyState(steer=steer, accel=accel, brake=brake)
        return np.array([steer, accel, brake], dtype=np.float32)

    def neutral(self) -> np.ndarray:
        return self.apply(0.0, 0.0, 0.0)

    def close(self) -> None:
        self.neutral()
