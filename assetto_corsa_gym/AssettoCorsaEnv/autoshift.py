import logging
from collections.abc import Mapping

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_GAS_THRESHOLDS = {
    "normal": (0.95, 0.40, 12.0, 0.15),
    "sport": (0.80, 0.40, 24.0, 0.50),
    "eco": (1.00, 0.50, 6.0, 0.00),
}


def _cfg_get(config, key, default):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class AutoShifter:
    """Stateful automatic gearbox rule set adapted to env telemetry."""

    def __init__(self, config=None, ctrl_rate=25):
        self.enabled = bool(_cfg_get(config, "enabled", True))
        self.mode = str(_cfg_get(config, "mode", "normal")).lower()
        self.ctrl_rate = float(ctrl_rate)
        self.gear_index_offset = int(_cfg_get(config, "gear_index_offset", 0))
        self.max_rpm = float(_cfg_get(config, "max_rpm", 0.0))
        self.max_shift_rpm_ratio = float(_cfg_get(config, "max_shift_rpm_ratio", 0.95))
        self.idle_rpm = float(_cfg_get(config, "idle_rpm", _cfg_get(config, "idle_rpm_fallback", 1000.0)))
        self.rpm_range_divisor = float(_cfg_get(config, "rpm_range_divisor", 3.0))
        self.min_shift_interval_s = float(_cfg_get(config, "min_shift_interval_s", 0.10))
        self.upshift_cooldown_s = float(_cfg_get(config, "upshift_cooldown_s", 1.0))
        self.downshift_cooldown_s = float(_cfg_get(config, "downshift_cooldown_s", 2.0))
        self.braking_downshift_cooldown_s = float(_cfg_get(config, "braking_downshift_cooldown_s", 1.0))
        self.up_after_downshift_cooldown_s = float(_cfg_get(config, "up_after_downshift_cooldown_s", 1.0))
        self.downshift_to_first_aggression = float(_cfg_get(config, "downshift_to_first_aggression", 0.95))
        self.min_downshift_to_first_speed_kmh = float(_cfg_get(config, "min_downshift_to_first_speed_kmh", 15.0))
        self.overdrive_downshift_gear = int(_cfg_get(config, "overdrive_downshift_gear", 4))
        self.slip_threshold = float(_cfg_get(config, "slip_threshold", 1.0))
        self.max_shift_rpm = self.max_rpm * self.max_shift_rpm_ratio if self.max_rpm > 0 else 0.0
        self.reset()

    def reset(self):
        self.elapsed_time = 0.0
        self.aggressiveness = self.thresholds[3]
        self.last_inc_aggr_time = 0.0
        self.last_shift_time = -1e9
        self.last_shift_up_time = -1e9
        self.last_shift_down_time = -1e9
        self.rpm_range_top = self.idle_rpm + 1000.0
        self.rpm_range_bottom = self.idle_rpm

    @property
    def thresholds(self):
        return DEFAULT_GAS_THRESHOLDS.get(self.mode, DEFAULT_GAS_THRESHOLDS["normal"])

    @property
    def rpm_range_size(self):
        usable_range = max(self.max_rpm - self.idle_rpm, 1.0)
        return usable_range / max(self.rpm_range_divisor, 1.0)

    def configure_static_info(self, static_info):
        max_rpm = static_info.get("maxRpm", static_info.get("MaxRpm", 0)) if static_info else 0
        if max_rpm:
            self.max_rpm = float(max_rpm)
            self.max_shift_rpm = self.max_rpm * self.max_shift_rpm_ratio
            logger.info("AutoShift max_rpm configured from static info: %.0f", self.max_rpm)

    def update(self, state, dt):
        dt = float(dt)
        self.elapsed_time += dt
        if not self.enabled:
            return np.zeros(2, dtype=np.float32), self.info_dict()

        telemetry = self._extract_telemetry(state)
        if self.max_rpm <= 0:
            self.max_rpm = max(telemetry["rpm"] * 1.25, self.idle_rpm + 3000.0)
            self.max_shift_rpm = self.max_rpm * self.max_shift_rpm_ratio

        self._update_aggressiveness(telemetry, dt)
        shift_up, shift_down = self._make_decision(telemetry)
        if shift_up:
            self.last_shift_time = self.elapsed_time
            self.last_shift_up_time = self.elapsed_time
        elif shift_down:
            self.last_shift_time = self.elapsed_time
            self.last_shift_down_time = self.elapsed_time

        action = np.array([float(shift_up), float(shift_down)], dtype=np.float32)
        return action, self.info_dict(telemetry)

    def _extract_telemetry(self, state):
        gear = int(state.get("actualGear", 0)) - self.gear_index_offset
        gas = float(state.get("accStatus", state.get("gas", 0.0)))
        if gas < 0.0:
            gas = (gas + 1.0) * 0.5
        brake = float(state.get("brakeStatus", state.get("brake", 0.0)))
        if brake < 0.0:
            brake = (brake + 1.0) * 0.5
        if "speed_kmh" in state:
            speed = float(state.get("speed_kmh", 0.0))
        else:
            speed = float(state.get("speed", 0.0)) * 3.6
        rpm = float(state.get("RPM", state.get("rpm", 0.0)))
        return {
            "gear": gear,
            "gas": float(np.clip(gas, 0.0, 1.0)),
            "brake": float(np.clip(brake, 0.0, 1.0)),
            "speed_kmh": max(speed, 0.0),
            "rpm": max(rpm, 0.0),
            "slipping": self._is_slipping(state),
        }

    def _is_slipping(self, state):
        if "NdSlip" in state:
            try:
                return max(np.asarray(state["NdSlip"], dtype=np.float32).reshape(-1)) > self.slip_threshold
            except Exception:
                pass
        slip_keys = [
            "tyre_slip_ratio_fl",
            "tyre_slip_ratio_fr",
            "tyre_slip_ratio_rl",
            "tyre_slip_ratio_rr",
        ]
        values = [abs(float(state[key])) for key in slip_keys if key in state]
        return bool(values and max(values) > self.slip_threshold)

    def _update_aggressiveness(self, telemetry, dt):
        gas_top, gas_bottom, calm_decay, minimum_aggr = self.thresholds
        denom = max(gas_top - gas_bottom, 1e-6)
        gas_aggr = (telemetry["gas"] - gas_bottom) / denom
        brake_aggr = (telemetry["brake"] - (gas_bottom - 0.3)) / denom * 1.6
        new_aggr = float(np.clip(max(gas_aggr, brake_aggr), 0.0, 1.0))

        if new_aggr > self.aggressiveness and telemetry["gear"] > 0:
            self.aggressiveness = new_aggr
            self.last_inc_aggr_time = self.elapsed_time

        if self.elapsed_time > self.last_inc_aggr_time + 2.0:
            self.aggressiveness -= dt / max(calm_decay, 1e-6)

        self.aggressiveness = float(np.clip(max(self.aggressiveness, minimum_aggr), 0.0, 1.0))
        self.rpm_range_top = (
            self.idle_rpm + 1000.0
            + ((self.max_shift_rpm - self.idle_rpm - 1000.0) * self.aggressiveness)
        )
        self.rpm_range_bottom = max(
            self.idle_rpm + (min(max(telemetry["gear"], 0), 6) * 80.0),
            self.rpm_range_top - self.rpm_range_size,
        )

    def _make_decision(self, telemetry):
        downshift_cooldown = (
            self.braking_downshift_cooldown_s
            if telemetry["brake"] > 0.0
            else self.downshift_cooldown_s
        )

        if (
            self.elapsed_time < self.last_shift_time + self.min_shift_interval_s
            or telemetry["gear"] < 1
            or self.elapsed_time < self.last_shift_up_time + self.upshift_cooldown_s
            or self.elapsed_time < self.last_shift_down_time + downshift_cooldown
        ):
            return False, False

        if (
            telemetry["rpm"] > self.rpm_range_top
            and not telemetry["slipping"]
            and self.elapsed_time > self.last_shift_down_time + self.up_after_downshift_cooldown_s
            and telemetry["brake"] <= 0.0
            and telemetry["gas"] > 0.0
        ):
            return True, False

        can_downshift_to_first = (
            telemetry["gear"] == 2
            and (
                self.aggressiveness >= self.downshift_to_first_aggression
                or telemetry["speed_kmh"] <= self.min_downshift_to_first_speed_kmh
            )
        )
        overdrive_braking = telemetry["gear"] >= self.overdrive_downshift_gear and telemetry["brake"] > 0.0
        if (
            telemetry["rpm"] < self.rpm_range_bottom
            and not telemetry["slipping"]
            and telemetry["gear"] > 1
            and self.elapsed_time > self.last_shift_down_time + downshift_cooldown
            and (telemetry["gear"] > 2 or can_downshift_to_first or overdrive_braking)
        ):
            return False, True

        return False, False

    def info_dict(self, telemetry=None):
        info = {
            "auto_aggressiveness": float(self.aggressiveness),
            "auto_rpm_range_top": float(self.rpm_range_top),
            "auto_rpm_range_bottom": float(self.rpm_range_bottom),
        }
        if telemetry:
            info.update({
                "auto_gear": int(telemetry["gear"]),
                "auto_slipping": bool(telemetry["slipping"]),
            })
        return info
