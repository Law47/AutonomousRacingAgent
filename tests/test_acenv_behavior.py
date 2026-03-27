from types import SimpleNamespace

import numpy as np

from acEnv import ACEnv


class DummyController:
    def apply(self, steer, accel, brake):
        return np.array([steer, accel, brake], dtype=np.float32)


def make_env_stub():
    env = object.__new__(ACEnv)
    env.action_dim = 2
    env.controller = DummyController()
    env._current_pedal_command = 0.0
    env._last_applied_action = np.zeros(2, dtype=np.float32)
    env._action_history = []
    env.state = {}
    env.reward_cfg = {
        'w_progress': 1.0,
        'w_gap': 0.35,
        'w_heading': 0.15,
        'w_offtrack': 2.0,
        'w_terminal_stuck': 3.0,
        'w_low_speed': 0.20,
        'low_speed_target_kmh': 25.0,
        'w_steer': 0.03,
        'w_steer_delta': 0.0,
        'w_overlap': 0.0,
    }
    env.racing_line_manager = SimpleNamespace(line_distance_threshold=12.0)
    env._last_track_features = {'signed_distance': 0.0, 'heading_error': 0.0}
    env.physics = SimpleNamespace(speedKmh=0.0, numberOfTyresOut=0)
    env._last_reward_breakdown = {}
    return env


def test_set_actions_tracks_signed_pedal_state_and_brake():
    env = make_env_stub()

    env.set_actions(np.array([0.2, 0.6], dtype=np.float32))
    assert env.state['pedal_command'] == 0.6
    assert env.state['applied_accel'] == 0.6
    assert env.state['applied_brake'] == 0.0

    env.set_actions(np.array([0.2, -1.0], dtype=np.float32))
    assert env.state['pedal_command'] == -0.4
    assert env.state['applied_accel'] == 0.0
    assert env.state['applied_brake'] == 0.4


def test_normalize_raw_feature_reads_actuator_state_channels():
    env = make_env_stub()
    env.state['applied_accel'] = 0.25
    env.state['applied_brake'] = 0.75
    env._current_pedal_command = -0.5

    assert env._normalize_raw_feature('pedalCommand') == -0.5
    assert env._normalize_raw_feature('appliedAccel') == 0.25
    assert env._normalize_raw_feature('appliedBrake') == 0.75


def test_reward_penalizes_low_speed_and_high_steer():
    env = make_env_stub()

    env._last_applied_action = np.array([0.1, 0.0], dtype=np.float32)
    env.physics = SimpleNamespace(speedKmh=60.0, numberOfTyresOut=0)
    fast_reward = env.getReward(terminated=False, termination_reason='running')

    env._last_applied_action = np.array([1.0, 0.0], dtype=np.float32)
    env.physics = SimpleNamespace(speedKmh=5.0, numberOfTyresOut=0)
    slow_steer_reward = env.getReward(terminated=False, termination_reason='running')

    assert slow_steer_reward < fast_reward
    assert 'low_speed_penalty' in env._last_reward_breakdown
    assert 'steer_penalty' in env._last_reward_breakdown


def test_reward_terminal_stuck_penalty_applies():
    env = make_env_stub()
    env._last_applied_action = np.array([0.0, 0.0], dtype=np.float32)
    env.physics = SimpleNamespace(speedKmh=0.0, numberOfTyresOut=0)
    reward = env.getReward(terminated=True, termination_reason='low_speed')

    assert reward < 0.0
    assert env._last_reward_breakdown['terminal_stuck_penalty'] < 0.0
