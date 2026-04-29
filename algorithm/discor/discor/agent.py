import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import re
from pathlib import Path
from tqdm import tqdm
from collections.abc import Mapping

from discor.replay_buffer import ReplayBuffer, EnsembleBuffer
from discor.utils import RunningMeanStats
from AssettoCorsaEnv.data_loader import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time

TRAINING_STATE_FILENAME = 'training_state.pkl'
CTRL_P = b'\x10'
SPACE = b' '


def summary_fallback_candidates(path):
    load_path = Path(path)
    if load_path.is_file():
        load_path = load_path.parent

    candidates = []
    for candidate_dir in (load_path, *load_path.parents):
        candidate = candidate_dir / 'summary.csv'
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def checkpoint_step_from_path(path):
    for part in reversed(Path(path).parts):
        match = re.fullmatch(r"step[_-](\d+)", part)
        if match:
            return int(match.group(1))
    return None


def merge_env_episode_stats(agent_stats, env_stats):
    merged = dict(agent_stats)
    if not isinstance(env_stats, Mapping):
        return merged

    for key, value in env_stats.items():
        if key == 'total_steps':
            merged['env_total_steps'] = value
        else:
            merged[key] = value
    return merged


def sync_env_training_counters(envs, steps, episodes=None):
    seen = set()
    for env in envs:
        if env is None or id(env) in seen:
            continue
        seen.add(id(env))

        if hasattr(env, "total_steps"):
            env.total_steps = max(int(getattr(env, "total_steps", 0)), int(steps))
        if episodes is not None and hasattr(env, "n_episodes"):
            env.n_episodes = max(int(getattr(env, "n_episodes", 0)), int(episodes))
        if hasattr(env, "set_training_step"):
            env.set_training_step(int(steps))


def cfg_get(config, key, default):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def demo_lap_sample_counts(trajectory):
    counts = {}
    for state in trajectory:
        lap = state.get("LapCount", None)
        counts[lap] = counts.get(lap, 0) + 1
    return counts


def demo_progress_delta(previous_state, current_state, track_length):
    if previous_state is None or current_state is None:
        return 0.0, False

    current_dist = float(
        current_state.get(
            "LapDist",
            track_length * float(current_state.get("NormalizedSplinePosition", 0.0)),
        )
    )
    previous_dist = float(
        previous_state.get(
            "LapDist",
            track_length * float(previous_state.get("NormalizedSplinePosition", 0.0)),
        )
    )
    raw_delta = current_dist - previous_dist
    wrapped_forward = False
    if raw_delta < -0.5 * track_length:
        raw_delta += track_length
        wrapped_forward = True
    elif raw_delta > 0.5 * track_length:
        raw_delta -= track_length
    return raw_delta, wrapped_forward


def should_keep_demo_transition(previous_state, current_state, lap_counts, filter_config, track_length):
    if not cfg_get(filter_config, "enabled", False):
        return True, "kept"

    max_lap_samples = cfg_get(filter_config, "max_lap_samples", None)
    if max_lap_samples is not None:
        max_lap_samples = int(max_lap_samples)
        previous_lap = previous_state.get("LapCount", None) if previous_state else None
        current_lap = current_state.get("LapCount", None) if current_state else None
        if lap_counts.get(previous_lap, 0) > max_lap_samples or lap_counts.get(current_lap, 0) > max_lap_samples:
            return False, "long_lap"

    min_speed_ms = cfg_get(filter_config, "min_speed_ms", None)
    if min_speed_ms is not None:
        previous_speed = float(previous_state.get("speed", 0.0)) if previous_state else 0.0
        current_speed = float(current_state.get("speed", 0.0)) if current_state else 0.0
        if previous_speed < float(min_speed_ms) and current_speed < float(min_speed_ms):
            return False, "low_speed"

    max_abs_progress_delta_m = cfg_get(filter_config, "max_abs_progress_delta_m", None)
    if max_abs_progress_delta_m is not None and previous_state is not None:
        progress_delta, _wrapped_forward = demo_progress_delta(
            previous_state,
            current_state,
            track_length,
        )
        if abs(progress_delta) > float(max_abs_progress_delta_m):
            return False, "progress_jump"

    return True, "kept"


class Agent:
    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1_000_000,
                 update_interval=1, start_steps=10000, log_interval=10, checkpoint_freq=0,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_offline_buffer=False, offline_buffer_size=1_000_000,
                 start_steps_count="online_steps", offline_sampling_config=None,
                 shift_curriculum_config=None, shift_exploration_config=None,
                 wandb_logger=None, save_final_buffer=False,
                 save_checkpoint_buffer=False):

        # Environment.
        self._env = env
        self._test_env = test_env
        self.checkpoint_freq = checkpoint_freq
        self.wandb_logger = wandb_logger
        self.save_final_buffer = save_final_buffer
        self.save_checkpoint_buffer = save_checkpoint_buffer

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
        self._start_steps_count = str(start_steps_count)
        self._use_offline_buffer = bool(use_offline_buffer)
        self._offline_sampling_enabled = bool(
            cfg_get(offline_sampling_config, "enabled", self._use_offline_buffer)
        )
        self._offline_sampling_initial_ratio = float(
            cfg_get(offline_sampling_config, "initial_offline_ratio", 0.5)
        )
        self._offline_sampling_final_ratio = float(
            cfg_get(offline_sampling_config, "final_offline_ratio", 0.0)
        )
        self._offline_sampling_transition_steps = max(
            int(cfg_get(offline_sampling_config, "transition_steps", 3_000_000)), 1
        )
        self._shift_curriculum_enabled = bool(
            cfg_get(shift_curriculum_config, "enabled", True)
        )
        self._shift_auto_only_steps = int(
            cfg_get(shift_curriculum_config, "auto_only_steps", self._start_steps)
        )
        self._shift_eval_use_manual = bool(
            cfg_get(shift_curriculum_config, "eval_use_manual", True)
        )
        self._shift_handoff_prompt = str(
            cfg_get(
                shift_curriculum_config,
                "handoff_prompt",
                "Turn off Assetto Corsa automatic shifting, then press SPACE to continue training.",
            )
        )
        self._shift_handoff_bc_updates = max(
            int(cfg_get(shift_curriculum_config, "handoff_bc_updates", 0)),
            0,
        )
        self._shift_execute_deterministic = bool(
            cfg_get(shift_exploration_config, "execute_deterministic", True)
        )
        self._shift_initial_epsilon = max(
            float(cfg_get(shift_exploration_config, "initial_epsilon", 0.0)),
            0.0,
        )
        self._shift_final_epsilon = max(
            float(cfg_get(shift_exploration_config, "final_epsilon", 0.0)),
            0.0,
        )
        self._shift_epsilon_decay_steps = max(
            int(cfg_get(shift_exploration_config, "epsilon_decay_steps", 1)),
            1,
        )
        self._shift_prevent_low_gear_downshift = bool(
            cfg_get(shift_exploration_config, "prevent_low_gear_downshift", True)
        )
        self._shift_min_downshift_gear = int(
            cfg_get(shift_exploration_config, "min_downshift_gear", 2)
        )
        max_shift_gear = cfg_get(shift_exploration_config, "max_gear", 6)
        self._shift_max_gear = int(max_shift_gear) if max_shift_gear is not None else None
        self._shift_handoff_completed = (not self._shift_curriculum_enabled) or self._shift_auto_only_steps <= 0
        self._paused = False
        self._loaded_model = False
        self._assetto_auto_shift_source_gear = None
        self._collect_online_experience = False
        self._configure_replay_source_for_shift_phase()

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

    def _training_state(self):
        return {
            'version': 1,
            'steps': int(self._steps),
            'episodes': int(self._episodes),
            'demo_transition_count': int(self._demo_transition_count),
            'best_lap_time': float(self.best_lap_time),
            'best_reward': float(self.best_reward),
            'best_eval_score': float(self._best_eval_score),
            'shift_handoff_completed': bool(self._shift_handoff_completed),
            'replay_buffer_size': int(len(self._replay_buffer)),
            'online_buffer_size': int(getattr(self._replay_buffer, "online_size", len(self._replay_buffer))),
            'offline_buffer_size': int(getattr(self._replay_buffer, "offline_size", 0)),
        }

    def _save_training_state(self, path):
        os.makedirs(path, exist_ok=True)
        state_path = os.path.join(path, TRAINING_STATE_FILENAME)
        tmp_path = state_path + ".tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(self._training_state(), f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, state_path)
            logger.info("saved training state to %s", state_path)
        except Exception:
            logger.exception("Failed to save training state to %s", state_path)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.exception("Failed to remove temp training state at %s", tmp_path)

    def _load_training_state(self, path):
        state_path = os.path.join(path, TRAINING_STATE_FILENAME)
        if not os.path.exists(state_path):
            return False

        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.warning(
                "Unable to load training state from %s (%s). "
                "Falling back to replay-buffer-derived counters.",
                state_path,
                exc,
            )
            return False

        if not isinstance(state, Mapping):
            logger.warning(
                "Training state at %s was %s, expected mapping. "
                "Falling back to replay-buffer-derived counters.",
                state_path,
                type(state).__name__,
            )
            return False

        self._steps = int(state.get('steps', self._steps))
        self._episodes = int(state.get('episodes', self._episodes))
        self._demo_transition_count = int(
            state.get('demo_transition_count', self._demo_transition_count)
        )
        self.best_lap_time = float(state.get('best_lap_time', self.best_lap_time))
        self.best_reward = float(state.get('best_reward', self.best_reward))
        self._best_eval_score = float(state.get('best_eval_score', self._best_eval_score))
        self._shift_handoff_completed = bool(
            state.get('shift_handoff_completed', self._shift_handoff_completed)
        )
        self._configure_replay_source_for_shift_phase()
        self._sync_env_training_counters()
        logger.info(
            "loaded training state from %s. steps=%s episodes=%s",
            state_path,
            self._steps,
            self._episodes,
        )
        return True

    def _load_summary_training_state(self, path):
        candidates = summary_fallback_candidates(path)
        summary_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if summary_path is None:
            return False

        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            logger.exception("Unable to load summary fallback from %s", summary_path)
            return False

        if summary.empty:
            logger.warning("Summary fallback at %s was empty", summary_path)
            return False

        last_row = summary.iloc[-1]
        if 'step' in summary.columns and pd.notna(last_row.get('step')):
            self._steps = int(last_row['step'])
        elif 'total_steps' in summary.columns and pd.notna(last_row.get('total_steps')):
            self._steps = int(last_row['total_steps'])
        else:
            logger.warning("Summary fallback at %s has no step/total_steps column", summary_path)
            return False

        if 'episode' in summary.columns and pd.notna(last_row.get('episode')):
            self._episodes = int(last_row['episode'])
        elif 'ep_count' in summary.columns and pd.notna(last_row.get('ep_count')):
            self._episodes = int(last_row['ep_count'])

        if 'ep_reward' in summary.columns:
            rewards = pd.to_numeric(summary['ep_reward'], errors='coerce')
            if rewards.notna().any():
                self.best_reward = float(rewards.max())

        if 'BestLap' in summary.columns:
            best_laps = pd.to_numeric(summary['BestLap'], errors='coerce')
            best_laps = best_laps[best_laps > 0]
            if not best_laps.empty:
                self.best_lap_time = float(best_laps.min())

        self._configure_replay_source_for_shift_phase()
        self._sync_env_training_counters()
        logger.info(
            "loaded training counters from summary fallback %s. steps=%s episodes=%s",
            summary_path,
            self._steps,
            self._episodes,
        )
        return True

    def _load_checkpoint_path_training_state(self, path):
        checkpoint_step = checkpoint_step_from_path(path)
        if checkpoint_step is None:
            return False

        self._steps = int(checkpoint_step)
        self._sync_env_training_counters()
        logger.info(
            "restored training step %s from checkpoint path %s",
            self._steps,
            path,
        )
        return True

    def save(self, path, save_buffer=True):
        try:
            self._writer.flush()
        except Exception:
            logger.exception("Failed to flush TensorBoard writer before saving")
        self._algo.save_models(path)
        self._save_training_state(path)
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
        self._loaded_model = True
        loaded_training_state = self._load_training_state(path)
        if not loaded_training_state:
            loaded_training_state = self._load_summary_training_state(path)
        if not loaded_training_state:
            loaded_training_state = self._load_checkpoint_path_training_state(path)

        if load_buffer:
            loaded_buffer = False
            replay_buffer_path = os.path.join(path, 'replay_buffer.pkl')
            replay_buffer_tmp_path = replay_buffer_path + ".tmp"
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
                    if loaded_training_state:
                        return
                    raise
                if getattr(loaded_replay_buffer, "_state_shape", None) != self._env.observation_space.shape:
                    raise ValueError(
                        "Replay buffer is incompatible with the current observation shape. "
                        "Use --load_weights_only for inference or to restart data collection."
                    )
                if getattr(loaded_replay_buffer, "_action_shape", None) != self._env.action_space.shape:
                    raise ValueError(
                        "Replay buffer is incompatible with the current action shape. "
                        "This project now stores 3 continuous actions plus separate shift labels; "
                        "start a fresh run or load weights without the old buffer."
                    )
                expected_shift_shape = (getattr(self._env, "shift_action_dim", 2),)
                if getattr(loaded_replay_buffer, "_shift_shape", expected_shift_shape) != expected_shift_shape:
                    raise ValueError(
                        "Replay buffer is incompatible with the current shift-label shape. "
                        "Use --load_weights_only for inference or to restart data collection."
                    )
                self._replay_buffer = loaded_replay_buffer
                self._configure_replay_source_for_shift_phase()
                loaded_buffer = True
                logger.info(f"loaded buffer from {path}. Number of steps: {len(self._replay_buffer)}")
            else:
                logger.warning("replay_buffer.pkl not found in %s; continuing with model weights only", path)
                if os.path.exists(replay_buffer_tmp_path):
                    logger.warning(
                        "Found %s, which usually means a previous replay-buffer save was interrupted.",
                        replay_buffer_tmp_path,
                    )
            if not loaded_training_state and loaded_buffer:
                self._steps = len(self._replay_buffer)
                self._sync_env_training_counters()
                logger.info(
                    "No training state found; restored steps from replay buffer occupancy: %s",
                    self._steps,
                )
            if not loaded_buffer:
                logger.warning(
                    "No replay buffer was restored. Training can resume collecting new samples, "
                    "but updates will wait until at least one batch is available."
                )

    def _sync_env_training_counters(self):
        sync_env_training_counters(
            (self._env, self._test_env),
            steps=self._steps,
            episodes=self._episodes,
        )

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
        if self._steps % self._update_interval == 0 and self.can_update_model():
            batch = self.sample_replay_batch()
            train_stats = self.update_model_from_batch(
                batch,
                train_shift_rl=self.shift_rl_training_enabled(),
            )
        return train_stats

    def sample_replay_batch(self):
        if isinstance(self._replay_buffer, EnsembleBuffer):
            return self._replay_buffer.sample(
                self._batch_size,
                self._device,
                offline_ratio=self.offline_sample_ratio(),
            )
        return self._replay_buffer.sample(self._batch_size, self._device)

    def update_model_from_batch(self, batch, train_shift_rl=True):
        train_stats = self._algo.update_online_networks(
            batch,
            self._writer,
            train_shift_rl=train_shift_rl,
        )
        self._algo.update_target_networks()
        return train_stats

    def has_min_experience(self):
        if self._loaded_model:
            return True
        if self._start_steps_count == "replay_buffer":
            return len(self._replay_buffer) >= self._start_steps
        return self._steps >= self._start_steps

    def can_update_model(self):
        return self.has_min_experience() and len(self._replay_buffer) >= self._batch_size

    def can_update_shift_model(self):
        return len(self._replay_buffer) >= self._batch_size

    def update_shift_model(self):
        if self._steps % self._update_interval != 0 or not self.can_update_shift_model():
            return None
        batch = self.sample_replay_batch()
        return self._algo.update_shift_model_from_batch(batch, self._writer)

    def update_shift_behavior_clone(self):
        if self._steps % self._update_interval != 0 or not self.can_update_shift_model():
            return None
        batch = self.sample_replay_batch()
        return self._algo.update_shift_behavior_clone_from_batch(batch, self._writer)

    def train_shift_behavior_clone_updates(self, num_updates, desc="Shift behavior clone"):
        num_updates = int(num_updates)
        if num_updates <= 0:
            return 0
        if not self.can_update_shift_model():
            logger.warning(
                "Skipping shift behavior cloning because replay buffer has %s samples, "
                "but batch_size is %s.",
                len(self._replay_buffer),
                self._batch_size,
            )
            return 0

        updates = 0
        for _ in tqdm(range(num_updates), desc=desc):
            batch = self.sample_replay_batch()
            self._algo.update_shift_behavior_clone_from_batch(batch, self._writer)
            updates += 1
        self._writer.flush()
        return updates

    def offline_sample_ratio(self):
        if not self._use_offline_buffer or not self._offline_sampling_enabled:
            return 0.0
        elapsed = max(self._steps - self._start_steps, 0)
        progress = min(elapsed / self._offline_sampling_transition_steps, 1.0)
        ratio = self._offline_sampling_initial_ratio + (
            (self._offline_sampling_final_ratio - self._offline_sampling_initial_ratio) * progress
        )
        return float(np.clip(ratio, 0.0, 1.0))

    def shift_ac_auto_phase_active(self, eval_mode=False):
        if not self._shift_curriculum_enabled:
            return False
        if eval_mode and self._shift_eval_use_manual:
            return False
        return self._steps < self._shift_auto_only_steps and not self._shift_handoff_completed

    def shift_rl_training_enabled(self):
        return not self.shift_ac_auto_phase_active(eval_mode=False)

    def shift_exploration_epsilon(self):
        if self.shift_ac_auto_phase_active(eval_mode=False):
            return 0.0
        manual_steps = max(self._steps - self._shift_auto_only_steps, 0)
        progress = min(manual_steps / self._shift_epsilon_decay_steps, 1.0)
        epsilon = self._shift_initial_epsilon + (
            (self._shift_final_epsilon - self._shift_initial_epsilon) * progress
        )
        return float(np.clip(epsilon, 0.0, 1.0))

    def shift_action_mask(self, env=None):
        active_env = env if env is not None else self._env
        shift_dim = getattr(active_env, "shift_action_dim", 2)
        mask = np.ones(shift_dim + 1, dtype=bool)
        state = getattr(active_env, "state", None) or {}
        gear = state.get("actualGear", None)
        if gear is None:
            return mask

        try:
            gear = int(gear)
        except (TypeError, ValueError):
            return mask

        if shift_dim >= 1 and self._shift_max_gear is not None and gear >= self._shift_max_gear:
            mask[1] = False
        if shift_dim >= 2 and self._shift_prevent_low_gear_downshift and gear <= self._shift_min_downshift_gear:
            mask[2] = False
        mask[0] = True
        return mask

    def _configure_replay_source_for_shift_phase(self):
        if isinstance(self._replay_buffer, EnsembleBuffer):
            self._replay_buffer.online(
                getattr(self, "_collect_online_experience", False)
                or self.shift_rl_training_enabled()
            )

    def start_online_replay_collection(self):
        self._collect_online_experience = True
        self._configure_replay_source_for_shift_phase()

    def _read_key(self):
        if os.name != "nt":
            return None
        try:
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch()
        except Exception:
            logger.exception("Unable to read keyboard input for pause handling")
        return None

    def _handle_keyboard_pause(self):
        key = self._read_key()
        if key == CTRL_P:
            self._paused = not self._paused
            if self._paused:
                logger.info("Training paused. Press Ctrl+P to resume.")
                print("\nTraining paused. Press Ctrl+P to resume.\n")
            else:
                logger.info("Training resumed with Ctrl+P")
                print("Training resumed.\n")

        if not self._paused:
            return

        while self._paused:
            time.sleep(0.1)
            key = self._read_key()
            if key == CTRL_P:
                self._paused = False
                logger.info("Training resumed with Ctrl+P")
                print("Training resumed.\n")

    def _wait_for_space(self, message):
        print("\n" + "=" * 72)
        print(message)
        print("Press SPACE to continue.")
        print("=" * 72 + "\n")
        logger.info(message)

        if os.name != "nt":
            input("Press Enter after turning off Assetto Corsa automatic shifting...")
            return

        while True:
            time.sleep(0.1)
            key = self._read_key()
            if key == SPACE and not self._paused:
                return
            if key == CTRL_P:
                self._paused = not self._paused
                if self._paused:
                    print("\nTraining paused. Press Ctrl+P to resume, then SPACE for the handoff.\n")
                else:
                    print("\nTraining resumed. Press SPACE for the handoff.\n")

    def maybe_pause_for_shift_handoff(self):
        if (
            not self._shift_curriculum_enabled
            or self._shift_handoff_completed
            or self._steps < self._shift_auto_only_steps
        ):
            return

        if self._shift_handoff_bc_updates > 0:
            logger.info(
                "Running %s shift behavior-clone updates before manual handoff.",
                self._shift_handoff_bc_updates,
            )
            self.train_shift_behavior_clone_updates(
                self._shift_handoff_bc_updates,
                desc="Shift handoff BC",
            )

        self._wait_for_space(self._shift_handoff_prompt)
        self._shift_handoff_completed = True
        self._configure_replay_source_for_shift_phase()
        logger.info("Shift curriculum handoff complete. Manual shift policy is now active.")

    def infer_shift_action_from_gear_delta(self, previous_state, current_state):
        action = np.zeros(getattr(self._env, "shift_action_dim", 2), dtype=np.float32)
        if not previous_state or not current_state:
            return action
        previous_gear = int(previous_state.get("actualGear", 0))
        current_gear = int(current_state.get("actualGear", previous_gear))
        first_gear = 1

        if previous_gear > first_gear and current_gear <= first_gear:
            self._assetto_auto_shift_source_gear = previous_gear
            return action

        source_gear = self._assetto_auto_shift_source_gear
        if source_gear is not None:
            if current_gear <= first_gear:
                return action
            if current_gear > source_gear:
                action[0] = 1.0
            elif current_gear < source_gear and action.shape[0] > 1:
                action[1] = 1.0
            self._assetto_auto_shift_source_gear = None
            return action

        if previous_gear <= first_gear or current_gear <= first_gear:
            return action
        if current_gear > previous_gear:
            action[0] = 1.0
        elif current_gear < previous_gear and action.shape[0] > 1:
            action[1] = 1.0
        return action

    def record_shift_teacher_for_latest_state(self, shift_action):
        if not getattr(self._env, "states", None):
            return
        shift_action = np.asarray(shift_action, dtype=np.float32).reshape(-1)
        latest_state = self._env.states[-1]
        for i in range(min(getattr(self._env, "shift_action_dim", 2), shift_action.shape[0])):
            latest_state[f"shift_teacher_{i:01d}"] = float(shift_action[i])

    def select_action_and_shift(self, state, eval_mode=False, env=None):
        active_env = env if env is not None else self._env
        shift_action_mask = self.shift_action_mask(active_env)
        if (not eval_mode) and (not self.has_min_experience()):
            action = active_env.action_space.sample()
            policy_info = {
                "entropies": None,
                "shift_action": np.zeros(getattr(active_env, "shift_action_dim", 2), dtype=np.float32),
                "shift_probs": np.zeros(getattr(active_env, "shift_action_dim", 2), dtype=np.float32),
            }
        else:
            if eval_mode:
                action, policy_info = self._algo.exploit(
                    state,
                    shift_action_mask=shift_action_mask,
                )
            else:
                action, policy_info = self._algo.explore(
                    state,
                    shift_deterministic=self._shift_execute_deterministic,
                    shift_epsilon=self.shift_exploration_epsilon(),
                    shift_action_mask=shift_action_mask,
                )

        zero_shift = np.zeros(getattr(active_env, "shift_action_dim", 2), dtype=np.float32)
        manual_shift = np.asarray(policy_info.get("shift_action", zero_shift), dtype=np.float32)
        if self.shift_ac_auto_phase_active(eval_mode=eval_mode):
            shift_action = zero_shift
            teacher_shift = zero_shift
            shift_source = "assetto_auto"
        else:
            shift_action = manual_shift
            teacher_shift = zero_shift
            shift_source = "manual"

        return action, shift_action, teacher_shift, shift_source, policy_info

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
            if hasattr(self._env, "set_training_step"):
                self._env.set_training_step(self._steps)
            state = self._env.reset()
            self._assetto_auto_shift_source_gear = None
            step_start_time = time.perf_counter()

            while (not done):
                self._handle_keyboard_pause()
                self.maybe_pause_for_shift_handoff()
                if hasattr(self._env, "set_training_step"):
                    self._env.set_training_step(self._steps)

                start_profile = time.perf_counter()
                previous_env_state = getattr(self._env, "state", {}).copy()
                action, shift_action, teacher_shift, shift_source, policy_info = self.select_action_and_shift(state, env=self._env)
                action_perf.append(time.perf_counter() - start_profile)

                # apply actions right away without blocking
                self._env.set_actions(
                    action,
                    shift_action=shift_action,
                    shift_teacher=teacher_shift,
                    shift_source=shift_source,
                )

                # update model
                start_profile = time.perf_counter()
                if self.can_update_model():
                    train_stats = self.update_model()
                    if self.shift_ac_auto_phase_active():
                        shift_stats = self.update_shift_behavior_clone()
                        if shift_stats:
                            if train_stats is None:
                                train_stats = {}
                            train_stats.update(shift_stats)
                else:
                    if self.shift_ac_auto_phase_active():
                        shift_stats = self.update_shift_behavior_clone()
                    elif self.shift_rl_training_enabled():
                        shift_stats = self.update_shift_model()
                    else:
                        shift_stats = None
                    if shift_stats:
                        train_stats = shift_stats
                update_model_perf.append(time.perf_counter() - start_profile)

                # get observations
                next_state, reward, done, info = self._env.step(action=None)  # action is already applied
                if shift_source == "assetto_auto":
                    replay_shift_action = self.infer_shift_action_from_gear_delta(
                        previous_env_state,
                        getattr(self._env, "state", {}),
                    )
                    self.record_shift_teacher_for_latest_state(replay_shift_action)
                    replay_shift_weight = 1.0
                else:
                    replay_shift_action = shift_action
                    replay_shift_weight = 0.0
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
                    episode_done=rb_done,
                    shift_label=replay_shift_action,
                    shift_weight=replay_shift_weight,
                    demo_weight=0.0)

                self._steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state
                self._writer.add_scalar(
                    'curriculum/shift_ac_auto_phase',
                    float(self.shift_ac_auto_phase_active()),
                    self._steps,
                )
                self._writer.add_scalar(
                    'curriculum/shift_rl_training_enabled',
                    float(self.shift_rl_training_enabled()),
                    self._steps,
                )
                self._writer.add_scalar(
                    'curriculum/shift_exploration_epsilon',
                    self.shift_exploration_epsilon(),
                    self._steps,
                )

                if self.checkpoint_freq and (self._steps % self.checkpoint_freq == 0):
                    logger.info(f"checkpointing model {self._steps} steps")
                    self.save(
                        os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"),
                        save_buffer=self.save_checkpoint_buffer,
                    )
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
        ep_stats = merge_env_episode_stats(ep_stats, env_ep_stats)

        best_lap = float(env_ep_stats.get("BestLap", np.inf)) if isinstance(env_ep_stats, dict) else np.inf
        if best_lap > 0.0 and best_lap < self.best_lap_time:
            logger.info(f"new best lap time {best_lap}")
            self.best_lap_time = best_lap
            self.save(os.path.join(self._model_dir, 'best_lap_time'), save_buffer=False)

        env_reward = float(env_ep_stats.get("ep_reward", episode_return)) if isinstance(env_ep_stats, dict) else episode_return
        if env_reward > self.best_reward:
            logger.info(f"new best reward {env_reward}")
            self.best_reward = env_reward
            self.save(os.path.join(self._model_dir, 'best_reward'), save_buffer=False)

        eval_metrics = self.common_metrics()
        eval_metrics.update(ep_stats)
        if train_stats:
            eval_metrics.update(train_stats)
        update_model_perf = np.asarray(update_model_perf, dtype=np.float32)
        step_perf = np.asarray(step_perf, dtype=np.float32)
        action_perf = np.asarray(action_perf, dtype=np.float32)
        eval_metrics["update_model_perf_mean"] = update_model_perf.mean().item() if update_model_perf.size else 0.0
        eval_metrics["update_model_perf_max"] = update_model_perf.max().item() if update_model_perf.size else 0.0
        eval_metrics["update_model_perf_std"] = update_model_perf.std().item() if update_model_perf.size else 0.0
        eval_metrics["step_perf_mean"] = step_perf.mean().item() if step_perf.size else 0.0
        eval_metrics["step_perf_max"] = step_perf.max().item() if step_perf.size else 0.0
        eval_metrics["step_perf_std"] = step_perf.std().item() if step_perf.size else 0.0
        eval_metrics["step_perf_q99"] = np.quantile(step_perf, 0.99).item() if step_perf.size else 0.0
        eval_metrics["step_perf_> thres"] = np.sum(step_perf > 0.041).item() if step_perf.size else 0
        eval_metrics["action_perf_mean"] = action_perf.mean().item() if action_perf.size else 0.0
        eval_metrics["action_perf_max"] = action_perf.max().item() if action_perf.size else 0.0
        eval_metrics["action_perf_std"] = action_perf.std().item() if action_perf.size else 0.0
        logger.info(f"Avr step time: {eval_metrics['step_perf_mean']:.3f}s, actions: {eval_metrics['action_perf_mean']:.4f}s, update: {eval_metrics['update_model_perf_mean']:.3f}s")
        logger.info(f"Max step time: {eval_metrics['step_perf_max']:.3f}s, actions: {eval_metrics['action_perf_max']:.4f}s, update: {eval_metrics['update_model_perf_max']:.3f}s")
        logger.info(f"std step time: {eval_metrics['step_perf_std']:.3f}s, actions: {eval_metrics['action_perf_std']:.4f}s, update: {eval_metrics['update_model_perf_std']:.3f}s")
        logger.info(f"step_perf_> thres: {eval_metrics['step_perf_> thres']} / {len(step_perf)}")
        if self.wandb_logger:
            self.wandb_logger.log(eval_metrics, 'episodes')
        self.write_tensorboard_metrics(eval_metrics)
        self.episodes_stats.append(eval_metrics)
        pd.DataFrame(self.episodes_stats).to_csv(os.path.join(self._log_dir, 'summary.csv'), index=None)
        logger.info(f'Episode done. Took {ep_time:.2f}s.  Steps per episode: {episode_steps}. Buffer size: {len(self._replay_buffer)} fps: {episode_steps/ep_time:.2f}')

    def evaluate(self):
        try:
            total_return = 0.0
            for _ in range(self._num_eval_episodes):
                if hasattr(self._test_env, "set_training_step"):
                    self._test_env.set_training_step(self._steps)
                state = self._test_env.reset()
                episode_return = 0.0
                done = False

                while (not done):
                    if hasattr(self._test_env, "set_training_step"):
                        self._test_env.set_training_step(self._steps)
                    action, shift_action, teacher_shift, shift_source, policy_info = self.select_action_and_shift(state, eval_mode=True, env=self._test_env)
                    self._test_env.set_actions(
                        action,
                        shift_action=shift_action,
                        shift_teacher=teacher_shift,
                        shift_source=shift_source,
                    )
                    next_state, reward, done, info = self._test_env.step(action=None)
                    entropies = policy_info.get("entropies")
                    if entropies is not None:
                        self._test_env.states[-1]["entropies"] = entropies.cpu().numpy().item()
                    episode_return += reward
                    state = next_state
                total_return += episode_return
        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            env_ep_stats = self._test_env.close()
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
        metrics = dict(
            step=self._steps,
            episode=self._episodes,
            buffer_size=len(self._replay_buffer),
            offline_sample_ratio=self.offline_sample_ratio(),
            shift_ac_auto_phase=float(self.shift_ac_auto_phase_active()),
            shift_rl_training_enabled=float(self.shift_rl_training_enabled()),
            shift_handoff_completed=float(self._shift_handoff_completed),
            total_time=time.time() - self._start_time,
        )
        if isinstance(self._replay_buffer, EnsembleBuffer):
            metrics["offline_buffer_size"] = self._replay_buffer.offline_size
            metrics["online_buffer_size"] = self._replay_buffer.online_size
        return metrics

    def write_tensorboard_metrics(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, (bool, int, float, np.integer, np.floating)) and np.isfinite(value):
                tag = str(key).replace(" ", "_").replace(">", "gt")
                self._writer.add_scalar(f'episodes/{tag}', float(value), self._steps)
        self._writer.flush()

    def load_pre_train_data(self, trajs_path, env, log_steer_ratios=False, demo_filter_config=None):
        total_added_episodes = 0
        total_added_transitions = 0
        total_skipped_transitions = 0
        load_start_time = time.time()

        buffer_size_before = len(self._replay_buffer)
        env_data = DataLoader(env, trajs_path, log_steer_ratios=log_steer_ratios)
        for ep in tqdm(range(env_data.trajectories_count)[:], desc="Loading demo trajectories"):
            state = env_data.reset()
            lap_counts = demo_lap_sample_counts(env_data.trajectory)
            filter_stats = {}
            track_length = float(
                env_data.static_info.get(
                    "TrackLength",
                    getattr(env, "track_length", 1.0),
                )
            )

            total_added_episodes += 1
            trajectory_steps = max(len(env_data.trajectory) - 1, 0)
            logger.info(
                "Loading demonstration trajectory %s/%s with %s transitions",
                ep + 1,
                env_data.trajectories_count,
                trajectory_steps,
            )
            step_progress = tqdm(
                range(trajectory_steps),
                desc=f"Trajectory {ep + 1}/{env_data.trajectories_count}",
                leave=False,
            )
            for i in step_progress:
                action = env_data.act()
                next_state, reward, done, info = env_data.step(action)
                current_index = max(env_data.current_step - 1, 0)
                previous_index = max(current_index - 1, 0)
                previous_raw_state = env_data.trajectory[previous_index] if previous_index < len(env_data.trajectory) else None
                current_raw_state = env_data.trajectory[current_index] if current_index < len(env_data.trajectory) else None
                keep_transition, filter_reason = should_keep_demo_transition(
                    previous_raw_state,
                    current_raw_state,
                    lap_counts,
                    demo_filter_config,
                    track_length,
                )
                filter_stats[filter_reason] = filter_stats.get(filter_reason, 0) + 1

                if info['terminated']:
                    terminated = True
                else:
                    terminated = False

                # end of trajectory
                if i >= len(env_data.trajectory) - 2:
                    episode_done = True # will add done as zero to the RB
                else:
                    episode_done = False # use the termination signal from the environment

                if not keep_transition:
                    total_skipped_transitions += 1
                    state = next_state
                    if episode_done:
                        break
                    continue

                shift_label = info.get("shift_label", getattr(env_data, "shift_label", None))
                self._replay_buffer.append(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated=terminated,
                    episode_done=episode_done,
                    shift_label=shift_label,
                    shift_weight=1.0 if shift_label is not None else 0.0,
                    demo_weight=1.0,
                )
                total_added_transitions += 1
                state = next_state
                if episode_done:
                    break
            step_progress.close()
            skipped = sum(count for reason, count in filter_stats.items() if reason != "kept")
            if skipped:
                logger.info(
                    "Demo trajectory filter skipped %s/%s transitions from %s: %s",
                    skipped,
                    trajectory_steps,
                    env_data.trajectories_paths[ep],
                    filter_stats,
                )
        added_transitions = len(self._replay_buffer) - buffer_size_before
        self._demo_transition_count += added_transitions
        load_elapsed = time.time() - load_start_time
        logger.info(
            "Loaded %s demonstration episodes from %s. Added %s transitions, "
            "skipped %s transitions. Buffer size: %s. Took %.1fs",
            total_added_episodes,
            trajs_path,
            added_transitions,
            total_skipped_transitions,
            len(self._replay_buffer),
            load_elapsed,
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
                    self.update_model_from_batch(batch, train_shift_rl=False)
                    self._algo.update_shift_behavior_clone_from_batch(batch, self._writer)
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

    def pre_train_shift_behavior_clone_epochs(self, num_epochs, num_samples=None):
        if len(self._replay_buffer) == 0:
            raise ValueError("Cannot pre-train shifter without demonstration data in the replay buffer")

        num_epochs = int(num_epochs)
        if num_epochs <= 0:
            logger.info("Skipping shifter demonstration pre-training because num_epochs=%s", num_epochs)
            return 0

        if num_samples is None:
            num_samples = self._demo_transition_count or len(self._replay_buffer)
        num_samples = min(int(num_samples), len(self._replay_buffer))
        if num_samples <= 0:
            raise ValueError("Cannot pre-train shifter without demonstration transitions")

        total_updates = 0
        logger.info(
            "Pre-training shifter behavior clone from demonstrations for %s epochs "
            "over %s samples (batch_size=%s)",
            num_epochs,
            num_samples,
            self._batch_size,
        )
        for epoch in range(num_epochs):
            epoch_updates = 0
            progress = tqdm(
                self._replay_buffer.iter_batches(
                    self._batch_size,
                    self._device,
                    num_samples=num_samples,
                    shuffle=True,
                ),
                desc=f"Shift demo epoch {epoch + 1}/{num_epochs}",
            )
            for batch in progress:
                self._algo.update_shift_behavior_clone_from_batch(batch, self._writer)
                epoch_updates += 1
                total_updates += 1
            logger.info(
                "Finished shifter demonstration epoch %s/%s with %s updates",
                epoch + 1,
                num_epochs,
                epoch_updates,
            )

        logger.info("Finished shifter demonstration pre-training with %s total updates", total_updates)
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
                batch = self.sample_replay_batch()
                self.update_model_from_batch(batch, train_shift_rl=False)
                self._algo.update_shift_behavior_clone_from_batch(batch, self._writer)
        finally:
            self._algo.update_entropy = True
