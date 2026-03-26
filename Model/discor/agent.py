import os
import pickle
import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from discor.replay_buffer import EnsembleBuffer, ReplayBuffer
from discor.utils import RunningMeanStats


class Agent:
    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1_000_000,
                 update_interval=1, start_steps=10000, log_interval=10, checkpoint_freq=0,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_offline_buffer=False, offline_buffer_size=1_000_000,
                 save_final_buffer=False, wandb_logger=None, logger=None):
        self._env = env
        self._test_env = test_env
        self._algo = algo
        self._device = device
        self._start_time = time.time()

        self.checkpoint_freq = checkpoint_freq
        self.save_final_buffer = save_final_buffer
        self.wandb_logger = wandb_logger
        self.logger = logger

        self._env.seed(seed)
        self._test_env.seed(2**31 - 1 - seed)

        if use_offline_buffer:
            self._replay_buffer = EnsembleBuffer(
                memory_size=memory_size,
                state_shape=self._env.observation_space.shape,
                action_shape=self._env.action_space.shape,
                gamma=self._algo.gamma,
                nstep=self._algo.nstep,
                offline_buffer_size=offline_buffer_size,
            )
        else:
            self._replay_buffer = ReplayBuffer(
                memory_size=memory_size,
                state_shape=self._env.observation_space.shape,
                action_shape=self._env.action_space.shape,
                gamma=self._algo.gamma,
                nstep=self._algo.nstep,
            )

        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._summary_dir, exist_ok=True)

        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self.episodes_stats = []
        self._steps = 0
        self._episodes = 0
        self._best_eval_score = -np.inf
        self._train_return = RunningMeanStats(log_interval)

        self._num_steps = num_steps
        self._batch_size = batch_size
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes

    def save(self, path, save_buffer=True):
        self._algo.save_models(path)
        if save_buffer:
            with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as handle:
                pickle.dump(self._replay_buffer, handle)
            if self.logger:
                self.logger.info("saved replay buffer to %s", path)
        if self.logger:
            self.logger.info("saved models to %s", path)

    def run(self):
        try:
            while self._steps <= self._num_steps:
                self.train_episode()
                if self._eval_interval and self._steps and (self._steps % self._eval_interval == 0):
                    if self.logger:
                        self.logger.info("Evaluating")
                    self.evaluate()
        finally:
            self.save(os.path.join(self._model_dir, 'final'), save_buffer=self.save_final_buffer)

    def update_model(self):
        train_stats = None
        if self._steps % self._update_interval == 0:
            batch = self._replay_buffer.sample(self._batch_size, self._device)
            train_stats = self._algo.update_online_networks(batch, self._writer)
        self._algo.update_target_networks()
        return train_stats

    @staticmethod
    def _safe_mean(values):
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _safe_max(values):
        if not values:
            return 0.0
        return float(np.max(values))

    @staticmethod
    def _safe_std(values):
        if not values:
            return 0.0
        return float(np.std(values))

    @staticmethod
    def _safe_q99(values):
        if not values:
            return 0.0
        return float(np.quantile(np.asarray(values), 0.99))

    def train_episode(self):
        self._episodes += 1
        episode_return = 0.0
        episode_steps = 0
        ep_start_time = time.time()
        train_stats = None
        step_perf, action_perf, update_model_perf = [], [], []

        try:
            state = self._env.reset()
            done = False
            step_start_time = time.perf_counter()

            while not done and self._steps <= self._num_steps:
                start_profile = time.perf_counter()
                if self._steps < self._start_steps:
                    action = self._env.action_space.sample()
                else:
                    action, _ = self._algo.explore(state)
                action_perf.append(time.perf_counter() - start_profile)

                self._env.set_actions(action)

                start_profile = time.perf_counter()
                if self._steps >= self._start_steps and len(self._replay_buffer) >= self._batch_size:
                    train_stats = self.update_model()
                update_model_perf.append(time.perf_counter() - start_profile)

                next_state, reward, terminated, truncated, info = self._env.step(action=None)
                reward = float(reward)
                done = bool(terminated or truncated)
                step_perf.append(time.perf_counter() - step_start_time)
                step_start_time = time.perf_counter()

                masked_done = bool(terminated)
                replay_episode_done = bool(done)

                self._replay_buffer.append(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated=masked_done,
                    episode_done=replay_episode_done,
                )

                self._steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                if self.checkpoint_freq and (self._steps % self.checkpoint_freq == 0):
                    checkpoint_dir = os.path.join(self._model_dir, 'checkpoints', f'step_{self._steps:08d}')
                    if self.logger:
                        self.logger.info("checkpointing model %s steps", self._steps)
                    self.save(checkpoint_dir, save_buffer=False)
        except TimeoutError:
            if self.logger:
                self.logger.exception("Agent TimeoutError")
        finally:
            self._env.close()

        self._train_return.append(float(episode_return))
        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar('reward/train', self._train_return.get(), self._steps)

        metrics = self.common_metrics()
        metrics.update({
            'total_steps': self._steps,
            'episode': self._episodes,
            'ep_reward': float(episode_return),
            'ep_steps': episode_steps,
            'update_model_perf_mean': self._safe_mean(update_model_perf),
            'update_model_perf_max': self._safe_max(update_model_perf),
            'update_model_perf_std': self._safe_std(update_model_perf),
            'step_perf_mean': self._safe_mean(step_perf),
            'step_perf_max': self._safe_max(step_perf),
            'step_perf_std': self._safe_std(step_perf),
            'step_perf_q99': self._safe_q99(step_perf),
            'step_perf_> thres': int(np.sum(np.asarray(step_perf) > 0.041)) if step_perf else 0,
            'action_perf_mean': self._safe_mean(action_perf),
            'action_perf_max': self._safe_max(action_perf),
            'action_perf_std': self._safe_std(action_perf),
        })
        if train_stats:
            metrics.update(train_stats)

        if self.logger:
            self.logger.info(
                "Episode done. Took %.2fs. Steps per episode: %s. Buffer size: %s fps: %.2f",
                time.time() - ep_start_time,
                episode_steps,
                len(self._replay_buffer),
                episode_steps / max(time.time() - ep_start_time, 1e-6),
            )
        if self.wandb_logger:
            self.wandb_logger.log(metrics, 'episodes')
        self.episodes_stats.append(metrics)
        pd.DataFrame(self.episodes_stats).to_csv(os.path.join(self._log_dir, 'summary.csv'), index=None)

    def evaluate(self):
        total_return = 0.0
        try:
            for _ in range(self._num_eval_episodes):
                state = self._test_env.reset()
                episode_return = 0.0
                done = False

                while not done:
                    action, _ = self._algo.exploit(state)
                    next_state, reward, terminated, truncated, _ = self._test_env.step(action)
                    reward = float(reward)
                    done = bool(terminated or truncated)
                    episode_return += reward
                    state = next_state

                total_return += episode_return
        except TimeoutError:
            if self.logger:
                self.logger.exception("Agent TimeoutError")
        finally:
            self._test_env.close()

        mean_return = total_return / max(self._num_eval_episodes, 1)
        if mean_return > self._best_eval_score:
            self._best_eval_score = mean_return
            self._algo.save_models(os.path.join(self._model_dir, 'best'))
        self._writer.add_scalar('reward/test', mean_return, self._steps)

        if self.logger:
            self.logger.info("Evaluation mean return: %.3f", mean_return)

    def common_metrics(self):
        return dict(
            step=self._steps,
            episode=self._episodes,
            buffer_size=len(self._replay_buffer),
            total_time=time.time() - self._start_time,
        )

    def __del__(self):
        try:
            self._env.close()
            self._test_env.close()
            self._writer.close()
        except Exception:
            pass
