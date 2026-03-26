import numpy as np
import torch
from gymnasium.spaces import Box

from Model.discor.agent import Agent
from discor.replay_buffer import EnsembleBuffer, ReplayBuffer


class DummyEnv:
    def __init__(self):
        self.observation_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self._max_episode_steps = 3
        self.reset_count = 0
        self.step_count = 0

    def seed(self, seed=None):
        return seed

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        return np.zeros(4, dtype=np.float32)

    def set_actions(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)

    def step(self, action=None):
        if action is not None:
            self.set_actions(action)
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self._max_episode_steps
        return np.ones(4, dtype=np.float32), np.array([1.0], dtype=np.float32), terminated, truncated, {'terminated': terminated}

    def close(self):
        return None


class DummyAlgo:
    gamma = 0.99
    nstep = 1

    def explore(self, state):
        return np.array([0.0, 0.5, 0.0], dtype=np.float32), torch.zeros((1, 1))

    def exploit(self, state):
        return np.array([0.0, 0.5, 0.0], dtype=np.float32), torch.zeros((1, 1))

    def update_online_networks(self, batch, writer):
        return {'dummy': 1.0}

    def update_target_networks(self):
        return None

    def save_models(self, save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)


def test_agent_keeps_ensemble_buffer_when_requested(tmp_path):
    env = DummyEnv()
    agent = Agent(env, env, DummyAlgo(), str(tmp_path), torch.device('cpu'), use_offline_buffer=True, logger=None)
    assert isinstance(agent._replay_buffer, EnsembleBuffer)


def test_agent_uses_replay_buffer_by_default(tmp_path):
    env = DummyEnv()
    agent = Agent(env, env, DummyAlgo(), str(tmp_path), torch.device('cpu'), use_offline_buffer=False, logger=None)
    assert isinstance(agent._replay_buffer, ReplayBuffer)


def test_agent_casts_scalar_rewards_and_handles_truncation(tmp_path):
    env = DummyEnv()
    agent = Agent(env, env, DummyAlgo(), str(tmp_path), torch.device('cpu'), start_steps=0, batch_size=1, logger=None)
    agent.train_episode()
    assert len(agent._replay_buffer) == env._max_episode_steps
    assert isinstance(agent.episodes_stats[-1]['ep_reward'], float)
