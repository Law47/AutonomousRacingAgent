import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STREAMING_DEMO_FORMAT = "streaming_demo_v1"
DEFAULT_MAX_POINTS = 30000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize recorded Assetto Corsa demonstrations")
    parser.add_argument(
        "demo_path",
        type=str,
        help="Path to a demo .pkl/.parquet file or a directory containing recorded demos",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the generated visualization image",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of points to plot per series before downsampling",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open an interactive matplotlib window",
    )
    return parser.parse_args()


def resolve_demo_path(demo_path: str) -> Path:
    path = Path(demo_path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            list(path.glob("*.pkl")) + list(path.glob("*.parquet")),
            key=lambda item: item.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No .pkl or .parquet demonstrations found in {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Demonstration path not found: {path}")


def load_demo(path: Path):
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
        static_info_path = path.parent / "static_info.json"
        static_info = {}
        if static_info_path.exists():
            static_info = pd.read_json(static_info_path, typ="series").to_dict()
        return df, static_info

    if path.suffix != ".pkl":
        raise ValueError(f"Unsupported demonstration format: {path.suffix}")

    with open(path, "rb") as file_handle:
        payload = pickle.load(file_handle)
        if isinstance(payload, dict) and payload.get("format") == STREAMING_DEMO_FORMAT:
            static_info = payload.get("static_info", {})
            trajectory = []
            while True:
                try:
                    chunk = pickle.load(file_handle)
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    print(
                        f"Warning: detected an incomplete trailing chunk in {path}. "
                        "Showing the last fully-saved data."
                    )
                    break
                trajectory.extend(chunk.get("states", []))
            return pd.DataFrame(trajectory), static_info

        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected pickle payload in {path}")

        return pd.DataFrame(payload.get("states", [])), payload.get("static_info", {})


def downsample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride].reset_index(drop=True)


def get_time_axis(df: pd.DataFrame) -> np.ndarray:
    if "currentTime" in df:
        time_axis = df["currentTime"].to_numpy(dtype=float)
    elif "capture_time_s" in df:
        time_axis = df["capture_time_s"].to_numpy(dtype=float)
    else:
        time_axis = np.arange(len(df), dtype=float)
    return time_axis - time_axis[0]


def get_control_series(df: pd.DataFrame, control_index: int, fallback_column: str) -> np.ndarray:
    action_column = f"current_action_abs_{control_index}"
    if action_column in df:
        return df[action_column].to_numpy(dtype=float)
    if fallback_column in df:
        return df[fallback_column].to_numpy(dtype=float)
    return np.zeros(len(df), dtype=float)


def shade_binary_mask(ax, time_axis: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    if len(mask) == 0 or not mask.any():
        return
    mask = np.asarray(mask, dtype=bool)
    start_idx = None
    for idx, is_active in enumerate(mask):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            ax.axvspan(time_axis[start_idx], time_axis[idx - 1], color=color, alpha=alpha)
            start_idx = None
    if start_idx is not None:
        ax.axvspan(time_axis[start_idx], time_axis[-1], color=color, alpha=alpha)


def summarize_demo(df: pd.DataFrame, static_info) -> str:
    duration_s = float(get_time_axis(df)[-1]) if len(df) > 1 else 0.0
    speed_kph = df["speed"].to_numpy(dtype=float) * 3.6 if "speed" in df else np.zeros(len(df))
    off_track_count = int(df["out_of_track"].astype(bool).sum()) if "out_of_track" in df else 0
    shift_up_count = int(df["shift_up"].astype(bool).sum()) if "shift_up" in df else 0
    shift_down_count = int(df["shift_down"].astype(bool).sum()) if "shift_down" in df else 0
    lap_count = len(set(df["LapCount"])) if "LapCount" in df and len(df) else 0
    track_name = static_info.get("TrackFullName", "unknown track")
    car_name = static_info.get("CarName", "unknown car")
    return (
        f"Track: {track_name}\n"
        f"Car: {car_name}\n"
        f"Samples: {len(df)}\n"
        f"Duration: {duration_s:.1f}s\n"
        f"Laps observed: {lap_count}\n"
        f"Mean speed: {speed_kph.mean() if len(speed_kph) else 0.0:.1f} km/h\n"
        f"Max speed: {speed_kph.max() if len(speed_kph) else 0.0:.1f} km/h\n"
        f"Shift up count: {shift_up_count}\n"
        f"Shift down count: {shift_down_count}\n"
        f"Off-track samples: {off_track_count}"
    )


def create_figure(df: pd.DataFrame, static_info, title: str):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

    ax_path = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_controls = fig.add_subplot(grid[1, 1])

    time_axis = get_time_axis(df)
    off_track_mask = df["out_of_track"].astype(bool).to_numpy() if "out_of_track" in df else np.zeros(len(df), dtype=bool)

    if {"world_position_x", "world_position_y"}.issubset(df.columns):
        x = df["world_position_x"].to_numpy(dtype=float)
        y = df["world_position_y"].to_numpy(dtype=float)
        speed_for_color = df["speed"].to_numpy(dtype=float) * 3.6 if "speed" in df else np.zeros(len(df))
        ax_path.plot(x, y, color="0.75", linewidth=1.0, alpha=0.6)
        scatter = ax_path.scatter(x, y, c=speed_for_color, cmap="viridis", s=10, linewidths=0)
        if off_track_mask.any():
            ax_path.scatter(x[off_track_mask], y[off_track_mask], color="crimson", s=12, label="off track")
            ax_path.legend(loc="best")
        colorbar = fig.colorbar(scatter, ax=ax_path, fraction=0.046, pad=0.04)
        colorbar.set_label("Speed (km/h)")
        ax_path.set_aspect("equal", adjustable="box")
        ax_path.set_xlabel("World X")
        ax_path.set_ylabel("World Y")
        ax_path.set_title("Driven Path")
    else:
        ax_path.text(0.5, 0.5, "No path data available", ha="center", va="center")
        ax_path.set_axis_off()

    if "speed" in df:
        ax_speed.plot(time_axis, df["speed"].to_numpy(dtype=float) * 3.6, color="tab:blue", label="speed")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_xlabel("Time (s)")
    shade_binary_mask(ax_speed, time_axis, off_track_mask, color="crimson", alpha=0.08)
    ax_speed.grid(alpha=0.25)

    speed_lines, speed_labels = ax_speed.get_legend_handles_labels()
    if "RPM" in df:
        ax_rpm = ax_speed.twinx()
        ax_rpm.plot(time_axis, df["RPM"].to_numpy(dtype=float), color="tab:orange", alpha=0.8, label="RPM")
        ax_rpm.set_ylabel("RPM")
        rpm_lines, rpm_labels = ax_rpm.get_legend_handles_labels()
        ax_speed.legend(speed_lines + rpm_lines, speed_labels + rpm_labels, loc="upper left")
    elif speed_lines:
        ax_speed.legend(loc="upper left")
    ax_speed.set_title("Speed and RPM")

    steer = get_control_series(df, 0, "steerAngle")
    throttle = get_control_series(df, 1, "accStatus")
    brake = get_control_series(df, 2, "brakeStatus")
    ax_controls.plot(time_axis, steer, label="steer", color="tab:purple", linewidth=1.0)
    ax_controls.plot(time_axis, throttle, label="throttle", color="tab:green", linewidth=1.0)
    ax_controls.plot(time_axis, brake, label="brake", color="tab:red", linewidth=1.0)
    ax_controls.set_ylim(-1.05, 1.05)
    ax_controls.set_xlabel("Time (s)")
    ax_controls.set_ylabel("Control Value")
    ax_controls.grid(alpha=0.25)
    shade_binary_mask(ax_controls, time_axis, off_track_mask, color="crimson", alpha=0.08)

    reward_lines = []
    reward_labels = []
    if "actualGear" in df:
        ax_controls.step(
            time_axis,
            df["actualGear"].to_numpy(dtype=float),
            where="post",
            color="tab:gray",
            linewidth=1.0,
            alpha=0.75,
            label="gear",
        )
    if "shift_up" in df:
        ax_controls.step(
            time_axis,
            df["shift_up"].astype(float).to_numpy(),
            where="post",
            color="tab:cyan",
            linewidth=1.0,
            alpha=0.9,
            label="shift up",
        )
    if "shift_down" in df:
        ax_controls.step(
            time_axis,
            -df["shift_down"].astype(float).to_numpy(),
            where="post",
            color="tab:brown",
            linewidth=1.0,
            alpha=0.9,
            label="shift down",
        )
    if "reward" in df:
        ax_reward = ax_controls.twinx()
        ax_reward.plot(time_axis, df["reward"].to_numpy(dtype=float), color="tab:orange", alpha=0.35, label="reward")
        ax_reward.set_ylabel("Reward")
        reward_lines, reward_labels = ax_reward.get_legend_handles_labels()

    control_lines, control_labels = ax_controls.get_legend_handles_labels()
    ax_controls.legend(control_lines + reward_lines, control_labels + reward_labels, loc="upper right", ncol=2)
    ax_controls.set_title("Controls, Gear, Shifts, and Reward")

    fig.suptitle(title, fontsize=15)
    return fig


def main() -> None:
    args = parse_args()
    demo_file = resolve_demo_path(args.demo_path)
    df, static_info = load_demo(demo_file)
    if df.empty:
        raise ValueError(f"No demonstration states found in {demo_file}")

    df = downsample_frame(df, args.max_points)
    summary = summarize_demo(df, static_info)
    print(summary)

    title = f"Demonstration Viewer: {demo_file.name}"
    fig = create_figure(df, static_info, title=title)
    fig.text(
        0.015,
        0.015,
        summary,
        ha="left",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    if args.save_path:
        save_path = Path(args.save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"Saved figure to {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
