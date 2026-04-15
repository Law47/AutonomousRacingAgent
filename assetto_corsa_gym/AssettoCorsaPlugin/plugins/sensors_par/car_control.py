import logging
logger = logging.getLogger(__name__)

import os

try:
    import vgamepad as vg
except Exception:
    vg = None

if os.name == 'posix':
    try:
        from AssettoCorsaEnv.vjoy_linux import vJoy
    except Exception:
        try:
            from vjoy_linux import vJoy
        except Exception:
            vJoy = None
else:
    try:
        from AssettoCorsaEnv.vjoy import vJoy
    except Exception:
        try:
            from vjoy import vJoy
        except Exception:
            vJoy = None

SCALE = 16384
VJOY_SHIFT_UP_BUTTON = 0x00000001
VJOY_SHIFT_DOWN_BUTTON = 0x00000004
SHIFT_BUTTON_HOLD_UPDATES = 3


class Controls(object):
    def __init__(self, backend="vigem"):
        self.backend = (backend or "vigem").lower()
        if self.backend not in ["vigem", "vjoy"]:
            raise ValueError(f"Unsupported control backend '{self.backend}'. Use 'vigem' or 'vjoy'.")

        self.onButtons = 0
        self._shift_up_hold_remaining = 0
        self._shift_down_hold_remaining = 0
        self.gamepad = None
        self.vj = None

        if self.backend == "vigem":
            if vg is None:
                raise RuntimeError("vgamepad is not installed but control_backend='vigem' was requested")
            self.gamepad = vg.VX360Gamepad()
            logger.info("Using ViGEmBus backend (vgamepad)")
        else:
            if vJoy is None:
                raise RuntimeError("vJoy backend requested but vJoy module is unavailable")
            self.vj = vJoy()
            self.vj.open()
            logger.info("Using vJoy backend")

        # internal state
        self.steer = 1.0        # [0, 2], center=1.0
        self.acc = 0.0          # [0, 1]
        self.brake = 0.0        # [0, 1]
        self.enable_gear_shift = 0.
        self.shift_up = 0.
        self.shift_down = 0.

        # commands
        self.steer_cmd = 0.0 # [-1,1]
        self.pedal_cmd = 0.0 # [-1,1]
        self.brake_cmd = 0.0 # [-1,1]

        self.ct_12_stop = False
        self.update()

    def close(self):
        try:
            self.steer = 1.0
            self.acc = 0.0
            self.brake = 0.0
            self._clear_shift_buttons()
            self.update()
            if self.vj is not None:
                self.vj.close()
        except Exception:
            pass

    def trigger_emergency_stop(self):
        self.ct_12_stop = True
        self.steer = 1.0
        self.acc = 0.0
        self.brake = 0.5
        self._clear_shift_buttons()
        logger.info("CT12 triggered")
        self.update()

    def set_controls(self, steer, acc, brake, enable_gear_shift=False, shift_up=False, shift_down=False):
        self.steer_cmd = steer
        self.pedal_cmd = acc
        self.brake_cmd = brake

        if not self.ct_12_stop:
            self.steer = self.steer_cmd + 1.0
            if self.steer < 0.0:
                self.steer = 0.0
            elif self.steer > 2.0:
                self.steer = 2.0

            self.acc = (self.pedal_cmd + 1) / 2
            if self.acc < 0.0:
                self.acc = 0.0
            elif self.acc > 1.0:
                self.acc = 1.0

            self.brake = (self.brake_cmd + 1) / 2
            if self.brake < 0.0:
                self.brake = 0.0
            elif self.brake > 1.0:
                self.brake = 1.0

            if enable_gear_shift:
                if shift_up and shift_down:
                    self._clear_shift_buttons()
                elif shift_up:
                    self._shift_up_hold_remaining = SHIFT_BUTTON_HOLD_UPDATES
                    self._shift_down_hold_remaining = 0
                elif shift_down:
                    self._shift_down_hold_remaining = SHIFT_BUTTON_HOLD_UPDATES
                    self._shift_up_hold_remaining = 0
                else:
                    pass
            else:
                self._clear_shift_buttons()
        self.update()

    def update(self):
        self.onButtons = self._shift_buttons_for_update()
        if self.backend == "vigem":
            steer_normalized = float(self.steer) - 1.0
            self.gamepad.left_joystick_float(x_value_float=steer_normalized, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=float(self.acc))
            self.gamepad.left_trigger_float(value_float=float(self.brake))
            self._update_vigem_shift_buttons()
            self.gamepad.update()
        else:
            self.setJoy(self.steer, self.acc, self.brake, self.onButtons, SCALE)

    def _clear_shift_buttons(self):
        self._shift_up_hold_remaining = 0
        self._shift_down_hold_remaining = 0
        self.onButtons = 0

    def _shift_buttons_for_update(self):
        buttons = 0
        if self._shift_up_hold_remaining > 0:
            buttons |= VJOY_SHIFT_UP_BUTTON
            self._shift_up_hold_remaining -= 1
        if self._shift_down_hold_remaining > 0:
            buttons |= VJOY_SHIFT_DOWN_BUTTON
            self._shift_down_hold_remaining -= 1
        return buttons

    def _update_vigem_shift_buttons(self):
        shift_up_button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        shift_down_button = vg.XUSB_BUTTON.XUSB_GAMEPAD_X

        if self.onButtons & VJOY_SHIFT_UP_BUTTON:
            self.gamepad.press_button(button=shift_up_button)
        else:
            self.gamepad.release_button(button=shift_up_button)

        if self.onButtons & VJOY_SHIFT_DOWN_BUTTON:
            self.gamepad.press_button(button=shift_down_button)
        else:
            self.gamepad.release_button(button=shift_down_button)

    def setJoy(self, valueX, valueY, valueZ, onButtons, scale):
        xPos = int(valueX * scale)
        yPos = int(valueY * 2 * scale)
        zPos = int(valueZ * 2 * scale)
        joystickPosition = self.vj.generateJoystickPosition(wAxisX=xPos, wAxisY=yPos, wAxisZ=zPos, lButtons=onButtons)
        self.vj.update(joystickPosition)
