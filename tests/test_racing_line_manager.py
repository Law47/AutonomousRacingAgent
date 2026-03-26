from logging import getLogger

import numpy as np
from omegaconf import OmegaConf

from racing_line_manager import RacingLineManager


def test_racing_line_manager_exposes_track_features():
    config = OmegaConf.load('config.yml')
    manager = RacingLineManager(config, getLogger(__name__))

    assert manager.racing_line_loaded
    first_point = manager.racing_line_points[0]
    features = manager.get_track_features((float(first_point[0]), float(first_point[1]), 0.0), heading=float(manager.reference_yaw[0]))

    assert abs(features['signed_distance']) < 1e-4
    assert abs(features['heading_error']) < 1e-4
    assert len(features['lookahead_curvature']) == config.observation.curvature_lookahead_points
