from logging import getLogger

from omegaconf import OmegaConf

from racing_line_manager import RacingLineManager


def test_racing_line_manager_track_features_shape():
    config = OmegaConf.load('config.yml')
    manager = RacingLineManager(config, getLogger(__name__))
    point = manager.racing_line_points[0]
    features = manager.get_track_features((float(point[0]), float(point[1]), 0.0), heading=manager.reference_yaw[0])

    assert manager.racing_line_loaded
    assert abs(features['signed_distance']) < 1e-3
    assert len(features['lookahead_curvature']) == config.observation.curvature_lookahead_points
