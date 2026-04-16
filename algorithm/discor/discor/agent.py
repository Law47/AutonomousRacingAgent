import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from pathlib import Path
from tqdm import tqdm

from discor.replay_buffer import ReplayBuffer, EnsembleBuffer
from discor.utils import RunningMeanStats
from AssettoCorsaEnv.data_loader import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time

class Agent:
    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1_000_000,
                 update_interval=1, start_steps=10000, log_interval=10, checkpoint_freq=0,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_offline_buffer=False, offline_buffer_size=1_000_000,
                 wandb_logger=None, save_final_buffer=False):

        # Environment.
        self._env = env
        self._test_env = test_env
        self.checkpoint_freq = checkpoint_freq
        self.wandb_logger = wandb_logger
        self.save_final_buffer = save_final_buffer

        self._env.seed(seed)
        self._test_env.seed(2**31-1-seed)

        # Algorithm.
        self._algo = algo
        self._demo_transition_count = 0

        if use_offline_buffer:
            self._replay_buffer = EnsembleBuffer(memory_size=memory_size, state_shape=self._env.observation_space.shape,
                                                 action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep, offline_buffer_size=offline_buffer_size)
        else:
            # Replay buffer with n-step return.
            self._replay_buffer = ReplayBuffer(memory_size=memory_size, state_shape=self._env.observation_space.shape,
                                               action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep)

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
        self._start_time = time.time()

        self.best_lap_time = np.inf
        self.best_reward = -np.inf

        logger.info(f'num_steps: {num_steps}')
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'update_interval: {update_interval}')
        logger.info(f'start_steps: {start_steps}')
        logger.info(f'log_interval: {log_interval}')
        logger.info(f'eval_interval: {eval_interval}')
        logger.info(f'num_eval_episodes: {num_eval_episodes}')
        logger.info(f'seed: {seed}')
        logger.info(f'gamma: {self._algo.gamma}')
        logger.info(f'nstep: {self._algo.nstep}')
        logger.info(f'memory_size: {memory_size}')

    def save(self, path, save_buffer=True):
        self._algo.save_models(path)
        if save_buffer:
            replay_buffer_path = os.path.join(path, 'replay_buffer.pkl')
            tmp_path = replay_buffer_path + ".tmp"
            try:
                with open(tmp_path, 'wb') as f:
                    pickle.dump(self._replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, replay_buffer_path)
                logger.info("saved replay buffer to %s", path)
            except Exception:
                logger.exception("Failed to save replay buffer to %s", replay_buffer_path)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    logger.exception("Failed to remove temp replay buffer at %s", tmp_path)
        logger.info("saved models to {}".format(path))

    def load(self, path, load_buffer=True):
        try:
            self._algo.load_models(path)
        except ValueError:
            logger.exception("Unable to load model from %s", path)
            raise
        logger.info(f"loaded model from {path}")
        if load_buffer:
            replay_buffer_path = os.path.join(path, 'replay_buffer.pkl')
            if os.path.exists(replay_buffer_path):
                try:
                    with open(replay_buffer_path, 'rb') as f:
                        loaded_replay_buffer = pickle.load(f)
                except (pickle.UnpicklingError, EOFError) as exc:
                    logger.warning(
                        "Unable to load replay buffer from %s (%s). "
                        "Continuing with model weights only.",
                        replay_buffer_path,
                        exc,
                    )
                    return
                if getattr(loaded_replay_buffer, "_action_shape", None) != self._env.action_space.shape:
                    raise ValueError(
                        "Replay buffer is incompatible with the current action shape. "
                        "This project now uses 5 actions, so old 3-action replay buffers cannot be resumed; "
                        "start a fresh run or load weights without the old buffer."
                    )
                self._replay_buffer = loaded_replay_buffer
                self._steps = self._replay_buffer._n
                logger.info(f"loaded buffer from {path}. Number of steps: {len(self._replay_buffer)}")
            else:
                logger.warning("replay_buffer.pkl not found in %s; continuing with model weights only", path)

    def run(self):
        try:
            while True:
                self.train_episode()
                if self._steps > self._num_steps:
                    break
                if self._eval_interval and (self._steps % self._eval_interval == 0):
                    logger.info("Evaluating")
                    self.evaluate()
        finally:
            self.save(os.path.join(self._model_dir, 'final'), save_buffer=self.save_final_buffer)

    def update_model(self):
        train_stats = None
        # Update online networks.
        if self._steps % self._update_interval == 0:
            batch = self._replay_buffer.sample(self._batch_size, self._device)
            train_stats = self.update_model_from_batch(batch)
        return train_stats

    def update_model_from_batch(self, batch):
        train_stats = self._algo.update_online_networks(batch, self._writer)
        self._algo.update_target_networks()
        return train_stats

    def has_min_experience(self):
        return len(self._replay_buffer) >= self._start_steps

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
                if not self.has_min_experience():
                    action = self._env.action_space.sample()
                else:
                    action, _ = self._algo.explore(state)
                action_perf.append(time.perf_counter() - start_profile)

                # apply actions right away without blocking
                self._env.set_actions(action)

                # update model
                start_profile = time.perf_counter()
                if self.has_min_experience():
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

                if info['terminated']:
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
                    logger.info(f"checkpointing model {self._steps} steps")
                    self.save(os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"), save_buffer=False)
        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            env_ep_stats = self._env.close()

        # We log running mean of training rewards.
        self._train_return.append(episode_return)

        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar(
                'reward/train', self._train_return.get(), self._steps)

        print(f'Episode: {self._episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {episode_return:<5.1f}')

        ep_time = time.time() - ep_start_time
        ep_stats['total_steps'] = self._steps
        ep_stats['episode'] = self._episodes
        ep_stats['ep_reward'] = episode_return
        ep_stats['ep_steps'] = episode_steps
        ep_stats.update(env_ep_stats if isinstance(env_ep_stats, dict) else {})

        if env_ep_stats["BestLap"] < self.best_lap_time:
            logger.info(f"new best lap time {env_ep_stats['BestLap']}")
            self.best_lap_time = env_ep_stats["BestLap"]
            self.save(os.path.join(self._model_dir, 'best_lap_time'), save_buffer=False)

        if env_ep_stats["ep_reward"] > self.best_reward:
            logger.info(f"new best reward {env_ep_stats['ep_reward']}")
            self.best_reward = env_ep_stats["ep_reward"]
            self.save(os.path.join(self._model_dir, 'best_reward'), save_buffer=False)

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
        logger.info(f"Avr step time: {eval_metrics['step_perf_mean']:.3f}s, actions: {eval_metrics['action_perf_mean']:.4f}s, update: {eval_metrics['update_model_perf_mean']:.3f}s")
        logger.info(f"Max step time: {eval_metrics['step_perf_max']:.3f}s, actions: {eval_metrics['action_perf_max']:.4f}s, update: {eval_metrics['update_model_perf_max']:.3f}s")
        logger.info(f"std step time: {eval_metrics['step_perf_std']:.3f}s, actions: {eval_metrics['action_perf_std']:.4f}s, update: {eval_metrics['update_model_perf_std']:.3f}s")
        logger.info(f"step_perf_> thres: {eval_metrics['step_perf_> thres']} / {len(step_perf)}")
        if self.wandb_logger:
            self.wandb_logger.log(eval_metrics, 'episodes')
        self.episodes_stats.append(eval_metrics)
        pd.DataFrame(self.episodes_stats).to_csv(os.path.join(self._log_dir, 'summary.csv'), index=None)
        logger.info(f'Episode done. Took {ep_time:.2f}s.  Steps per episode: {episode_steps}. Buffer size: {len(self._replay_buffer)} fps: {episode_steps/ep_time:.2f}')

    def evaluate(self):
        try:
            total_return = 0.0
            for _ in range(self._num_eval_episodes):
                state = self._test_env.reset()
                episode_return = 0.0
                done = False

                while (not done):
                    action, entropies = self._algo.exploit(state)
                    next_state, reward, done, info = self._test_env.step(action)
                    self._test_env.states[-1]["entropies"] = entropies.cpu().numpy().item()
                    episode_return += reward
                    state = next_state
                total_return += episode_return
        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            env_ep_stats = self._env.close()
            pd.DataFrame([env_ep_stats]).to_csv(os.path.join(self._log_dir, 'eval_summary.csv'), index=None)

    def __del__(self):
        # Best-effort cleanup during interpreter shutdown.
        for closer in (
            lambda: self._env.close(),
            lambda: self._test_env.close(),
            lambda: self._writer.close(),
            lambda: self.wandb_logger.finish() if self.wandb_logger else None,
        ):
            try:
                closer()
            except Exception:
                pass

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._steps,
            episode=self._episodes,
            buffer_size=len(self._replay_buffer),
            total_time=time.time() - self._start_time,
        )

    def load_pre_train_data(self, trajs_path, env, log_steer_ratios=False):
        total_added_episodes = 0

        buffer_size_before = len(self._replay_buffer)
        env_data = DataLoader(env, trajs_path, log_steer_ratios=log_steer_ratios)
        for ep in tqdm(range(env_data.trajectories_count)[:]):
            state = env_data.reset()

            total_added_episodes += 1
            for i in range(len(env_data.trajectory) - 1):
                action = env_data.act()
                next_state, reward, done, info = env_data.step(action)

                if info['terminated']:
                    terminated = True
                else:
                    terminated = False

                # end of trajectory
                if i >= len(env_data.trajectory) - 2:
                    episode_done = True # will add done as zero to the RB
                else:
                    episode_done = False # use the termination signal from the environment

                self._replay_buffer.append(state, action, reward, next_state, terminated=terminated, episode_done=episode_done)
                state = next_state
                if episode_done:
                    break
        added_transitions = len(self._replay_buffer) - buffer_size_before
        self._demo_transition_count += added_transitions
        logger.info(
            "Loaded %s demonstration episodes from %s. Added %s transitions. Buffer size: %s",
            total_added_episodes,
            trajs_path,
            added_transitions,
            len(self._replay_buffer),
        )
        return added_transitions

    def pre_train_epochs(self, num_epochs, num_samples=None):
        if len(self._replay_buffer) == 0:
            raise ValueError("Cannot pre-train without demonstration data in the replay buffer")

        num_epochs = int(num_epochs)
        if num_epochs <= 0:
            logger.info("Skipping demonstration pre-training because num_epochs=%s", num_epochs)
            return 0

        if num_samples is None:
            num_samples = self._demo_transition_count or len(self._replay_buffer)
        num_samples = min(int(num_samples), len(self._replay_buffer))
        if num_samples <= 0:
            raise ValueError("Cannot pre-train without demonstration transitions")

        total_updates = 0
        self._algo.update_entropy = False
        logger.info(
            "Pre-training from demonstrations for %s epochs over %s samples "
            "(batch_size=%s)",
            num_epochs,
            num_samples,
            self._batch_size,
        )
        try:
            for epoch in range(num_epochs):
                epoch_updates = 0
                progress = tqdm(
                    self._replay_buffer.iter_batches(
                        self._batch_size,
                        self._device,
                        num_samples=num_samples,
                        shuffle=True,
                    ),
                    desc=f"Demo epoch {epoch + 1}/{num_epochs}",
                )
                for batch in progress:
                    self.update_model_from_batch(batch)
                    epoch_updates += 1
                    total_updates += 1
                logger.info(
                    "Finished demonstration epoch %s/%s with %s updates",
                    epoch + 1,
                    num_epochs,
                    epoch_updates,
                )
        finally:
            self._algo.update_entropy = True

        logger.info("Finished demonstration pre-training with %s total updates", total_updates)
        return total_updates

    def pre_train(self, num_updates=None):
        self._algo.update_entropy = False
        if len(self._replay_buffer) == 0:
            raise ValueError("Cannot pre-train without demonstration data in the replay buffer")

        if num_updates is None:
            num_updates = len(self._replay_buffer)

        logger.info("Pre-training for %s updates...", num_updates)
        try:
            for _ in tqdm(range(num_updates)):
                self.update_model()
        finally:
            self._algo.update_entropy = True
