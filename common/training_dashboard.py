import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """Live matplotlib dashboard for the training loop.

    The dashboard is intentionally best-effort: if the plotting backend is not
    available or a draw call fails, callers can keep training without it.
    """

    _REWARD_SERIES = (
        ("reward", "total reward"),
        ("base_reward_estimate", "base reward est."),
        ("reverse_gear_penalty", "reverse gear penalty"),
        ("out_of_track_penalty", "off-track penalty"),
    )
    _MODEL_OUTPUT_SERIES = (
        ("model_steer", "steer out"),
        ("model_throttle", "throttle out"),
        ("model_brake", "brake out"),
        ("model_shift_up", "shift up out"),
        ("model_shift_down", "shift down out"),
    )
    _APPLIED_CONTROL_SERIES = (
        ("applied_steer", "applied steer"),
        ("applied_throttle", "applied throttle"),
        ("applied_brake", "applied brake"),
        ("shift_up_pulse", "shift up pulse"),
        ("shift_down_pulse", "shift down pulse"),
    )
    _VEHICLE_INPUT_SERIES = (
        ("speed_scaled", "speed / 300 kph"),
        ("rpm_scaled", "rpm / 10000"),
        ("gear_scaled", "gear / 8"),
        ("gap_scaled", "gap / 12 m"),
        ("out_of_track", "out of track"),
    )

    def __init__(self, history=500, update_interval_steps=5, title="Training Dashboard"):
        self.history = max(int(history), 2)
        self.update_interval_steps = max(int(update_interval_steps), 1)
        self.title = title
        self.enabled = False
        self._closed = False
        self._steps = deque(maxlen=self.history)
        self._series = {
            name: deque(maxlen=self.history)
            for panel in self._panels()
            for name, _ in panel["series"]
        }
        self._latest_stats = {}

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Live training dashboard disabled; matplotlib could not be imported: %s", exc)
            return

        self._plt = plt
        self._plt.ion()
        self._fig, axes = self._plt.subplots(2, 2, figsize=(14, 8), num=title)
        self._axes = list(np.asarray(axes).reshape(-1))
        self._lines = {}
        self._latest_text = self._fig.text(
            0.01,
            0.01,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
        )

        manager = getattr(self._fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)

        for ax, panel in zip(self._axes, self._panels()):
            ax.set_title(panel["title"])
            ax.set_xlabel("training step")
            ax.grid(True, alpha=0.25)
            for name, label in panel["series"]:
                line, = ax.plot([], [], label=label, linewidth=1.4)
                self._lines[name] = line
            if panel.get("ylim") is not None:
                ax.set_ylim(*panel["ylim"])
            ax.legend(loc="upper left", fontsize="small")

        self._fig.tight_layout(rect=(0, 0.05, 1, 1))
        backend = str(self._plt.get_backend()).lower()
        if "agg" not in backend:
            try:
                self._fig.show()
            except Exception:
                self._plt.show(block=False)
        self.enabled = True
        logger.info(
            "Live training dashboard enabled with history=%s update_interval_steps=%s",
            self.history,
            self.update_interval_steps,
        )

    def _panels(self):
        return (
            {
                "title": "Rewards / Punishments",
                "series": self._REWARD_SERIES,
                "ylim": None,
            },
            {
                "title": "Raw Model Outputs",
                "series": self._MODEL_OUTPUT_SERIES,
                "ylim": (-1.05, 1.05),
            },
            {
                "title": "Applied Controls",
                "series": self._APPLIED_CONTROL_SERIES,
                "ylim": (-1.05, 1.05),
            },
            {
                "title": "Vehicle Inputs (Scaled)",
                "series": self._VEHICLE_INPUT_SERIES,
                "ylim": (-1.05, 1.05),
            },
        )

    def update(
        self,
        *,
        step,
        episode,
        episode_step,
        action=None,
        reward=None,
        env_state=None,
        info=None,
        train_stats=None,
    ):
        if not self.enabled or self._closed:
            return

        state = env_state if isinstance(env_state, dict) else {}
        info = info if isinstance(info, dict) else {}
        train_stats = train_stats if isinstance(train_stats, dict) else {}
        metrics = self._extract_metrics(action, reward, state)

        self._steps.append(int(step))
        for name in self._series:
            self._series[name].append(metrics.get(name, np.nan))

        self._latest_stats = {
            "step": int(step),
            "episode": int(episode),
            "episode_step": int(episode_step),
            "reward": self._to_float(reward, default=np.nan),
            "speed_kmh": self._to_float(state.get("speed"), default=np.nan) * 3.6,
            "rpm": self._to_float(state.get("RPM"), default=np.nan),
            "gear": self._to_float(state.get("actualGear"), default=np.nan),
            "gap": self._to_float(state.get("gap"), default=np.nan),
            "out_of_track": self._to_float(state.get("out_of_track"), default=np.nan),
            "terminated": bool(info.get("terminated", False)),
            "policy_loss": self._to_float(train_stats.get("policy_loss"), default=np.nan),
            "alpha": self._to_float(train_stats.get("alpha"), default=np.nan),
        }

        if int(step) % self.update_interval_steps == 0:
            self.redraw()

    def redraw(self):
        if not self.enabled or self._closed:
            return

        try:
            if not self._plt.fignum_exists(self._fig.number):
                self.close()
                return

            x = np.asarray(self._steps, dtype=np.float32)
            for name, line in self._lines.items():
                line.set_data(x, np.asarray(self._series[name], dtype=np.float32))

            if len(x):
                left = x[0]
                right = x[-1] if x[-1] > x[0] else x[0] + 1
                for ax, panel in zip(self._axes, self._panels()):
                    ax.set_xlim(left, right)
                    if panel.get("ylim") is None:
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)
                    else:
                        ax.set_ylim(*panel["ylim"])

            self._latest_text.set_text(self._format_latest_text())
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            logger.exception("Live training dashboard failed while drawing; disabling it.")
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.enabled = False
        try:
            if hasattr(self, "_plt") and hasattr(self, "_fig"):
                self._plt.close(self._fig)
        except Exception:
            logger.exception("Failed to close live training dashboard.")

    def _extract_metrics(self, action, reward, state):
        action_values = self._action_values(action, state)
        applied = self._applied_controls(state)
        speed_kmh = self._to_float(state.get("speed"), default=np.nan) * 3.6
        reward_value = self._to_float(reward, state.get("reward", np.nan))
        reverse_gear_penalty = self._to_float(state.get("reverse_gear_penalty"), 0.0)
        out_of_track_penalty = self._to_float(state.get("out_of_track_penalty"), 0.0)
        base_reward_estimate = reward_value - reverse_gear_penalty - out_of_track_penalty

        return {
            "reward": reward_value,
            "base_reward_estimate": base_reward_estimate,
            "reverse_gear_penalty": reverse_gear_penalty,
            "out_of_track_penalty": out_of_track_penalty,
            "model_steer": action_values[0],
            "model_throttle": action_values[1],
            "model_brake": action_values[2],
            "model_shift_up": action_values[3],
            "model_shift_down": action_values[4],
            "applied_steer": applied[0],
            "applied_throttle": applied[1],
            "applied_brake": applied[2],
            "shift_up_pulse": self._to_float(state.get("shift_up"), 0.0),
            "shift_down_pulse": self._to_float(state.get("shift_down"), 0.0),
            "speed_scaled": self._scale(speed_kmh, 300.0),
            "rpm_scaled": self._scale(state.get("RPM"), 10000.0),
            "gear_scaled": self._scale(state.get("actualGear"), 8.0),
            "gap_scaled": self._scale(state.get("gap"), 12.0),
            "out_of_track": self._to_float(state.get("out_of_track"), 0.0),
        }

    def _action_values(self, action, state):
        values = []
        flat_action = None
        if action is not None:
            try:
                flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
            except Exception:
                flat_action = None

        for index in range(5):
            value = np.nan
            if flat_action is not None and index < flat_action.shape[0]:
                value = flat_action[index]
            else:
                value = state.get(f"actions_{index}")
                if value is None:
                    value = state.get(f"actions_{index:01d}")
            values.append(self._to_float(value, np.nan))
        return values

    def _applied_controls(self, state):
        steer = state.get("current_action_abs_0")
        throttle = state.get("current_action_abs_1")
        brake = state.get("current_action_abs_2")

        if steer is None:
            steer = self._scale(state.get("steerAngle"), 450.0)
        if throttle is None:
            throttle = state.get("accStatus")
        if brake is None:
            brake = state.get("brakeStatus")

        return (
            self._to_float(steer, np.nan),
            self._to_float(throttle, np.nan),
            self._to_float(brake, np.nan),
        )

    def _format_latest_text(self):
        if not self._latest_stats:
            return ""

        policy_loss = self._latest_stats["policy_loss"]
        alpha = self._latest_stats["alpha"]
        optimizer_text = ""
        if np.isfinite(policy_loss) or np.isfinite(alpha):
            optimizer_text = f" | policy_loss={policy_loss:.4f} alpha={alpha:.4f}"

        return (
            f"step={self._latest_stats['step']} "
            f"episode={self._latest_stats['episode']} "
            f"episode_step={self._latest_stats['episode_step']} | "
            f"reward={self._latest_stats['reward']:.4f} "
            f"speed={self._latest_stats['speed_kmh']:.1f}kph "
            f"rpm={self._latest_stats['rpm']:.0f} "
            f"gear={self._latest_stats['gear']:.0f} "
            f"gap={self._latest_stats['gap']:.2f} "
            f"oot={self._latest_stats['out_of_track']:.0f} "
            f"terminated={int(self._latest_stats['terminated'])}"
            f"{optimizer_text}"
        )

    @staticmethod
    def _scale(value, divisor):
        return TrainingDashboard._to_float(value, np.nan) / float(divisor)

    @staticmethod
    def _to_float(value, default=np.nan):
        if value is None:
            return default
        try:
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return default
                value = value.reshape(-1)[0]
            return float(value)
        except (TypeError, ValueError):
            return default
