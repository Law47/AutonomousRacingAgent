"""
Track-reference utilities backed by CSV racing-line data.
"""

from __future__ import annotations

import os
from logging import Logger
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class RacingLineManager:
    def __init__(self, config: OmegaConf, logger: Logger):
        self.config = config
        self.logger = logger

        rl_cfg = config.get('racing_line', {})
        obs_cfg = config.get('observation', {})

        self.enabled = rl_cfg.get('enable', True)
        self.track_name = rl_cfg.get('track_name', 'monza')
        self.line_distance_weight = float(rl_cfg.get('line_distance_weight', 1.0))
        self.line_distance_threshold = float(rl_cfg.get('line_distance_threshold', 12.0))
        self.curvature_lookahead_points = int(obs_cfg.get('curvature_lookahead_points', 12))
        self.curvature_lookahead_distance_m = float(obs_cfg.get('curvature_lookahead_distance_m', 300.0))

        self.racing_line_loaded = False
        self.racing_line_points = None
        self.left_border_points = None
        self.right_border_points = None
        self.reference_yaw = None
        self.reference_target_speed = None
        self.segment_lengths = None
        self.cumulative_lengths = None
        self.total_length = 0.0

        if self.enabled:
            self._load_racing_line_from_csv()

    def _load_racing_line_from_csv(self) -> bool:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            racing_lines_dir = os.path.join(current_dir, 'Racing Lines')
            csv_file = os.path.join(racing_lines_dir, f'{self.track_name}_racing_line.csv')

            if not os.path.exists(csv_file):
                self.logger.warning(f"[RACING_LINE] CSV file not found: {csv_file}")
                return False

            df = pd.read_csv(csv_file)
            self.racing_line_points = df[['pos_x', 'pos_y']].values.astype(np.float32)
            self.left_border_points = df[['left_border_x', 'left_border_y']].values.astype(np.float32)
            self.right_border_points = df[['right_border_x', 'right_border_y']].values.astype(np.float32)
            self.reference_yaw = np.deg2rad(df['yaw'].values.astype(np.float32))
            self.reference_target_speed = df['target_speed'].values.astype(np.float32)

            diffs = np.roll(self.racing_line_points, -1, axis=0) - self.racing_line_points
            self.segment_lengths = np.linalg.norm(diffs, axis=1)
            self.cumulative_lengths = np.concatenate([[0.0], np.cumsum(self.segment_lengths[:-1])]).astype(np.float32)
            self.total_length = float(np.sum(self.segment_lengths))

            self.racing_line_loaded = len(self.racing_line_points) > 0
            self.logger.info(f"Loaded racing line '{self.track_name}' with {len(self.racing_line_points)} points")
            return self.racing_line_loaded
        except Exception as exc:
            self.logger.error(f"[RACING_LINE] Error loading CSV: {type(exc).__name__}: {exc}")
            return False

    def distance_to_racing_line(self, car_pos: Tuple[float, float, float]) -> float:
        return abs(self.get_track_features(car_pos).get('signed_distance', self.line_distance_threshold))

    def _car_xy(self, car_pos: Tuple[float, float, float]) -> np.ndarray:
        return np.array([car_pos[0], car_pos[1]], dtype=np.float32)

    def _nearest_index(self, car_xy: np.ndarray) -> int:
        distances = np.linalg.norm(self.racing_line_points - car_xy, axis=1)
        return int(np.argmin(distances))

    def _tangent_at(self, index: int) -> np.ndarray:
        prev_idx = (index - 1) % len(self.racing_line_points)
        next_idx = (index + 1) % len(self.racing_line_points)
        tangent = self.racing_line_points[next_idx] - self.racing_line_points[prev_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return tangent / tangent_norm

    def _distance_to_index(self, start_index: int, distance_m: float) -> int:
        if self.total_length <= 0.0:
            return start_index
        target_distance = (float(self.cumulative_lengths[start_index]) + distance_m) % self.total_length
        return int(np.searchsorted(self.cumulative_lengths, target_distance, side='left') % len(self.racing_line_points))

    def get_curvature_lookahead(self, nearest_index: int, num_points: int, distance_m: float) -> np.ndarray:
        if not self.racing_line_loaded:
            return np.zeros(num_points, dtype=np.float32)

        sample_distances = np.linspace(0.0, distance_m, num_points + 1, dtype=np.float32)[1:]
        base_heading = float(self.reference_yaw[nearest_index])
        lookahead = np.zeros(num_points, dtype=np.float32)
        for idx, sample_distance in enumerate(sample_distances):
            sample_index = self._distance_to_index(nearest_index, float(sample_distance))
            lookahead[idx] = wrap_angle(float(self.reference_yaw[sample_index]) - base_heading) / np.pi
        return lookahead

    def get_track_features(self, car_pos: Tuple[float, float, float], heading: float | None = None) -> Dict[str, object]:
        if not self.enabled or not self.racing_line_loaded:
            return {
                'nearest_index': -1,
                'distance': self.line_distance_threshold,
                'signed_distance': self.line_distance_threshold,
                'heading_error': 0.0,
                'tangent': np.array([1.0, 0.0], dtype=np.float32),
                'point': np.zeros(2, dtype=np.float32),
                'lookahead_curvature': np.zeros(self.curvature_lookahead_points, dtype=np.float32),
                'target_speed': 0.0,
            }

        car_xy = self._car_xy(car_pos)
        nearest_index = self._nearest_index(car_xy)
        nearest_point = self.racing_line_points[nearest_index]
        tangent = self._tangent_at(nearest_index)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        delta = car_xy - nearest_point
        signed_distance = float(np.dot(delta, normal))
        distance = float(abs(signed_distance))

        if heading is None:
            heading_error = 0.0
        else:
            reference_heading = float(self.reference_yaw[nearest_index])
            heading_error = wrap_angle(float(heading) - reference_heading)

        return {
            'nearest_index': nearest_index,
            'distance': distance,
            'signed_distance': signed_distance,
            'heading_error': heading_error,
            'tangent': tangent,
            'point': nearest_point,
            'lookahead_curvature': self.get_curvature_lookahead(
                nearest_index,
                self.curvature_lookahead_points,
                self.curvature_lookahead_distance_m,
            ),
            'target_speed': float(self.reference_target_speed[nearest_index]),
        }

    def get_line_visualization_points(self, max_points: int = 100):
        if not self.racing_line_loaded or self.racing_line_points is None:
            return []
        if len(self.racing_line_points) <= max_points:
            return self.racing_line_points.tolist()
        indices = np.linspace(0, len(self.racing_line_points) - 1, max_points, dtype=int)
        return self.racing_line_points[indices].tolist()
