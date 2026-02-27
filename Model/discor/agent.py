import os
import time
import numpy as np
import pandas as pd
import pickle
from torch.utils.tensorboard import SummaryWriter

from discor.replay_buffer import ReplayBuffer, EnsembleBuffer
from discor.utils import RunningMeanStats

class Agent:

    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1_000_000,
                 update_interval=1, start_steps=10000, log_interval=10, checkpoint_freq=0,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_offline_buffer=False, offline_buffer_size=1_000_000,
                 save_final_buffer=False, wandb_logger=None, logger = None):

        # Environment.
        self._env = env
        self._test_env = test_env
        self.checkpoint_freq = checkpoint_freq
        self.wandb_logger = wandb_logger
        self.save_final_buffer = save_final_buffer

        self._env.seed(seed)
        self._test_env.seed(2**31-1-seed)

        self.wandb_logger = wandb_logger
        self.logger = logger

        self._start_time = time.time()

        # Algorithm.
        self._algo = algo

        if use_offline_buffer:
            self._replay_buffer = EnsembleBuffer(memory_size=memory_size, state_shape=self._env.observation_space.shape,
                                                 action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep, offline_buffer_size=offline_buffer_size)
        else:
            # Replay buffer with n-step return.
            self._replay_buffer = ReplayBuffer(memory_size=memory_size, state_shape=self._env.observation_space.shape,
                                               action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep)


        # Replay buffer with n-step return.
        self._replay_buffer = ReplayBuffer(
            memory_size=memory_size,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            gamma=self._algo.gamma, nstep=self._algo.nstep)

        # Directory to log.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self.episodes_stats = []
        self._steps = 0
        self._episodes = 0
        self._train_return = RunningMeanStats(log_interval)
        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self._best_eval_score = -np.inf

        self._device = device
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
            with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as f:
                pickle.dump(self._replay_buffer, f)
            self.logger.info("saved replay buffer to {}".format(path))
        self.logger.info("saved models to {}".format(path))

    def run(self):
        try:
            while True:
                self.train_episode()
                if self._steps > self._num_steps:
                    break
                if self._eval_interval and (self._steps % self._eval_interval == 0):
                    self.logger.info("Evaluating")
                    self.evaluate()
        finally:
            self.save(os.path.join(self._model_dir, 'final'), save_buffer=self.save_final_buffer)

    def update_model(self):
        train_stats = None
        # Update online networks.
        if self._steps % self._update_interval == 0:
            batch = self._replay_buffer.sample(self._batch_size, self._device)
            train_stats = self._algo.update_online_networks(batch, self._writer)

        # Update target networks.
        self._algo.update_target_networks()
        return train_stats

    def train_episode(self):
        """
        Train only one episode
        """
        self._episodes += 1
        episode_return = 0.
        episode_steps = 0

        ep_start_time = time.time()
        ep_stats = {}
        train_stats = None

        try:
            done = False
            step_perf, action_perf, update_model_perf = [], [], []
            state = self._env.reset()
            step_start_time = time.perf_counter()

            while (not done):
                start_profile = time.perf_counter()
                if self._start_steps > self._steps:
                    action = self._env.action_space.sample()
                else:
                    action, _ = self._algo.explore(state)
                action_perf.append(time.perf_counter() - start_profile)

                # apply actions right away without blocking
                self._env.set_actions(action)

                # update model
                start_profile = time.perf_counter()
                if self._steps >= self._start_steps:
                    train_stats = self.update_model()
                update_model_perf.append(time.perf_counter() - start_profile)

                # get observations
                next_state, reward, done, info = self._env.step(action=None)  # action is already applied
                step_perf.append(time.perf_counter() - step_start_time)
                step_start_time = time.perf_counter()

                # Set done=True only when the agent fails, ignoring done signal
                # if the agent reach time horizons.
                if (episode_steps + 1 >= self._env._max_episode_steps):
                    masked_done = False
                else:
                    masked_done = done

                if done:
                    rb_done = True
                else:
                    rb_done = False

                self._replay_buffer.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=rb_done)

                self._steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                if self.checkpoint_freq and (self._steps % self.checkpoint_freq == 0):
                    self.logger.info(f"checkpointing model {self._steps} steps")
                    self.save(os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"), save_buffer=False)
        except TimeoutError:
            self.logger.exception("Agent TimeoutError")
        finally:
            env_ep_stats = self._env.close()

        # We log running mean of training rewards.
        self._train_return.append(episode_return)

        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar(
                'reward/train', self._train_return.get(), self._steps)

        print(f'Episode: {self._episodes}  '
              f'Episode steps: {episode_steps}  '
              f'Return: {episode_return}')

        ep_time = time.time() - ep_start_time
        ep_stats['total_steps'] = self._steps
        ep_stats['episode'] = self._episodes
        ep_stats['ep_reward'] = episode_return
        ep_stats['ep_steps'] = episode_steps
        eval_metrics = self.common_metrics()
        eval_metrics.update(ep_stats)
        if train_stats:
            eval_metrics.update(train_stats)
        eval_metrics["update_model_perf_mean"] = np.array(update_model_perf).mean()
        eval_metrics["update_model_perf_max"] = np.array(update_model_perf).max()
        eval_metrics["update_model_perf_std"] = np.array(update_model_perf).std()
        eval_metrics["step_perf_mean"] = np.array(step_perf).mean()
        eval_metrics["step_perf_max"] = np.array(step_perf).max()
        eval_metrics["step_perf_std"] = np.array(step_perf).std()
        eval_metrics["step_perf_q99"] = np.quantile(np.array(step_perf), 0.99)
        eval_metrics["step_perf_> thres"] = np.sum(np.array(step_perf) > 0.041)
        eval_metrics["action_perf_mean"] = np.array(action_perf).mean()
        eval_metrics["action_perf_max"] = np.array(action_perf).max()
        eval_metrics["action_perf_std"] = np.array(action_perf).std()
        self.logger.info(f"Avr step time: {eval_metrics['step_perf_mean']:.3f}s, actions: {eval_metrics['action_perf_mean']:.4f}s, update: {eval_metrics['update_model_perf_mean']:.3f}s")
        self.logger.info(f"Max step time: {eval_metrics['step_perf_max']:.3f}s, actions: {eval_metrics['action_perf_max']:.4f}s, update: {eval_metrics['update_model_perf_max']:.3f}s")
        self.logger.info(f"std step time: {eval_metrics['step_perf_std']:.3f}s, actions: {eval_metrics['action_perf_std']:.4f}s, update: {eval_metrics['update_model_perf_std']:.3f}s")
        self.logger.info(f"step_perf_> thres: {eval_metrics['step_perf_> thres']} / {len(step_perf)}")
        if self.wandb_logger:
            self.wandb_logger.log(eval_metrics, 'episodes')
        self.episodes_stats.append(eval_metrics)
        pd.DataFrame(self.episodes_stats).to_csv(os.path.join(self._log_dir, 'summary.csv'), index=None)
        self.logger.info(f'Episode done. Took {ep_time:.2f}s.  Steps per episode: {episode_steps}. Buffer size: {len(self._replay_buffer)} fps: {episode_steps/ep_time:.2f}')

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._steps,
            episode=self._episodes,
            buffer_size=len(self._replay_buffer),
            total_time=time.time() - self._start_time,
        )

    def evaluate(self):
        total_return = 0.0
        if self._test_env.is_metaworld:
            total_success = 0.0

        for _ in range(self._num_eval_episodes):
            state = self._test_env.reset()
            episode_return = 0.0
            done = False
            if self._test_env.is_metaworld:
                success = 0.0

            while (not done):
                action = self._algo.exploit(state)
                next_state, reward, done, info = self._test_env.step(action)
                episode_return += reward
                state = next_state

                if self._test_env.is_metaworld and info['success'] > 1e-8:
                    success = 1.0

            total_return += episode_return
            if self._test_env.is_metaworld:
                total_success += success

        mean_return = total_return / self._num_eval_episodes
        if self._test_env.is_metaworld:
            success_rate = total_success / self._num_eval_episodes
            self._writer.add_scalar(
                'reward/success_rate', success_rate, self._steps)

        if mean_return > self._best_eval_score:
            self._best_eval_score = mean_return
            self._algo.save_models(os.path.join(self._model_dir, 'best'))

        self._writer.add_scalar(
            'reward/test', mean_return, self._steps)
        print('-' * 60)
        print(f'Num steps: {self._steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def __del__(self):
        self._env.close()
        self._test_env.close()
        self._writer.close()
