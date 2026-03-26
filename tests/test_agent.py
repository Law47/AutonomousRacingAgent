import numpy as np
from gymnasium.spaces import Box

from Model.discor.agent import Agent
from discor.replay_buffer import EnsembleBuffer, ReplayBuffer


class DummyAlgo:
    gamma = 0.99
    nstep = 1

    def save_models(self, path):
        return None

    def update_target_networks(self):
        return None

    def update_online_networks(self, batch, writer):
        return {'ok': 1.0}

    def explore(self, state):
        return np.array([0.0, 0.5, 0.0], dtype=np.float32), None

    def exploit(self, state):
        return np.array([0.0, 0.5, 0.0], dtype=np.float32), None


class DummyEnv:
    def __init__(self):
        self.action_space = Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._max_episode_steps = 2
        self.steps = 0

    def seed(self, seed=None):
        return None

    def reset(self):
        self.steps = 0
        return np.zeros(4, dtype=np.float32)

    def set_actions(self, action):
        self.last_action = action

    def step(self, action=None):
        self.steps += 1
        return np.ones(4, dtype=np.float32), 1.0, False, self.steps >= 2, {'terminated': False}

    def close(self):
        return None


def test_agent_keeps_offline_buffer_when_requested(tmp_path):
    env = DummyEnv()
    agent = Agent(env=env, test_env=env, algo=DummyAlgo(), log_dir=str(tmp_path), device='cpu', use_offline_buffer=True)
    assert isinstance(agent._replay_buffer, EnsembleBuffer)


def test_agent_uses_replay_buffer_by_default(tmp_path):
    env = DummyEnv()
    agent = Agent(env=env, test_env=env, algo=DummyAlgo(), log_dir=str(tmp_path), device='cpu', use_offline_buffer=False)
    assert isinstance(agent._replay_buffer, ReplayBuffer)
