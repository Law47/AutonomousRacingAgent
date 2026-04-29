from collections import deque
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)


class NStepBuffer:
    def __init__(self, gamma=0.99, nstep=3):
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._discounts = [gamma ** i for i in range(nstep)]
        self._nstep = nstep
        self.reset()

    def append(self, state, action, reward, shift_label, shift_weight, demo_weight):
        self._states.append(state)
        self._actions.append(action)
        self._shift_labels.append(shift_label)
        self._shift_weights.append(shift_weight)
        self._demo_weights.append(demo_weight)
        self._rewards.append(reward)

    def get(self):
        assert len(self._rewards) > 0

        state = self._states.popleft()
        action = self._actions.popleft()
        shift_label = self._shift_labels.popleft()
        shift_weight = self._shift_weights.popleft()
        demo_weight = self._demo_weights.popleft()
        reward = self._nstep_reward()
        return state, action, shift_label, shift_weight, demo_weight, reward

    def _nstep_reward(self):
        reward = np.sum([
            r * d for r, d in zip(self._rewards, self._discounts)])
        self._rewards.popleft()
        return reward

    def reset(self):
        self._states = deque(maxlen=self._nstep)
        self._actions = deque(maxlen=self._nstep)
        self._shift_labels = deque(maxlen=self._nstep)
        self._shift_weights = deque(maxlen=self._nstep)
        self._demo_weights = deque(maxlen=self._nstep)
        self._rewards = deque(maxlen=self._nstep)

    def is_empty(self):
        return len(self._rewards) == 0

    def is_full(self):
        return len(self._rewards) == self._nstep

    def __len__(self):
        return len(self._rewards)


class ReplayBuffer:
    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99, nstep=1, shift_shape=(2,)):
        assert isinstance(memory_size, int) and memory_size > 0
        assert isinstance(state_shape, tuple)
        assert isinstance(action_shape, tuple)
        assert isinstance(shift_shape, tuple)
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._memory_size = memory_size
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._shift_shape = shift_shape
        self._gamma = gamma
        self._nstep = nstep
        self._reset()

    def _reset(self):
        self._n = 0
        self._p = 0

        self._states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._next_states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._actions = np.empty(
            (self._memory_size, ) + self._action_shape, dtype=np.float32)
        self._shift_labels = np.empty(
            (self._memory_size, ) + self._shift_shape, dtype=np.float32)
        self._shift_weights = np.empty((self._memory_size, 1), dtype=np.float32)
        self._demo_weights = np.empty((self._memory_size, 1), dtype=np.float32)

        self._rewards = np.empty((self._memory_size, 1), dtype=np.float32)
        self._dones = np.empty((self._memory_size, 1), dtype=np.float32)

        if self._nstep != 1:
            self._nstep_buffer = NStepBuffer(self._gamma, self._nstep)
        logger.info(f"Replay buffer initialized for {self._memory_size} samples")

    def __getstate__(self):
        state = self.__dict__.copy()
        serialized_n = int(self._n)
        for key in ("_states", "_next_states", "_actions", "_shift_labels", "_shift_weights", "_demo_weights", "_rewards", "_dones"):
            state[key] = state[key][:serialized_n].copy()
        state["_serialized_n"] = serialized_n
        return state

    def __setstate__(self, state):
        serialized_n = int(state.pop("_serialized_n", state.get("_n", 0)))
        array_keys = ("_states", "_next_states", "_actions", "_shift_labels", "_shift_weights", "_demo_weights", "_rewards", "_dones")
        saved_arrays = {key: state.pop(key) for key in array_keys if key in state}
        self.__dict__.update(state)
        if self._nstep != 1 and (
            not hasattr(self, "_nstep_buffer")
            or not hasattr(self._nstep_buffer, "_shift_weights")
            or not hasattr(self._nstep_buffer, "_demo_weights")
        ):
            self._nstep_buffer = NStepBuffer(self._gamma, self._nstep)

        self._states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._next_states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._actions = np.empty(
            (self._memory_size, ) + self._action_shape, dtype=np.float32)
        self._shift_labels = np.empty(
            (self._memory_size, ) + self._shift_shape, dtype=np.float32)
        self._shift_weights = np.empty((self._memory_size, 1), dtype=np.float32)
        self._demo_weights = np.empty((self._memory_size, 1), dtype=np.float32)
        self._rewards = np.empty((self._memory_size, 1), dtype=np.float32)
        self._dones = np.empty((self._memory_size, 1), dtype=np.float32)

        copy_n = min(serialized_n, self._memory_size)
        if copy_n:
            self._states[:copy_n] = saved_arrays["_states"][:copy_n]
            self._next_states[:copy_n] = saved_arrays["_next_states"][:copy_n]
            self._actions[:copy_n] = saved_arrays["_actions"][:copy_n]
            self._shift_labels[:copy_n] = saved_arrays["_shift_labels"][:copy_n]
            if "_shift_weights" in saved_arrays:
                self._shift_weights[:copy_n] = saved_arrays["_shift_weights"][:copy_n]
            else:
                self._shift_weights[:copy_n] = 1.0
            if "_demo_weights" in saved_arrays:
                self._demo_weights[:copy_n] = saved_arrays["_demo_weights"][:copy_n]
            else:
                self._demo_weights[:copy_n] = 0.0
            self._rewards[:copy_n] = saved_arrays["_rewards"][:copy_n]
            self._dones[:copy_n] = saved_arrays["_dones"][:copy_n]

        self._n = copy_n
        if self._n >= self._memory_size:
            self._p = int(state.get("_p", 0)) % self._memory_size
        else:
            self._p = self._n

    def append(self, state, action, reward, next_state, terminated, episode_done=None, shift_label=None, shift_weight=None, demo_weight=None):
        """
        done (masked_done): False if the agent reach time horizons. Else = done
        """
        if shift_label is None:
            shift_label = np.zeros(self._shift_shape, dtype=np.float32)
            if shift_weight is None:
                shift_weight = 0.0
        else:
            shift_label = np.asarray(shift_label, dtype=np.float32).reshape(self._shift_shape)
            if shift_weight is None:
                shift_weight = 1.0
        shift_weight = np.asarray(shift_weight, dtype=np.float32).reshape(1)
        if demo_weight is None:
            demo_weight = 0.0
        demo_weight = np.asarray(demo_weight, dtype=np.float32).reshape(1)

        if self._nstep != 1:
            self._nstep_buffer.append(state, action, reward, shift_label, shift_weight, demo_weight)

            if self._nstep_buffer.is_full():
                state, action, shift_label, shift_weight, demo_weight, reward = self._nstep_buffer.get()
                self._append(state, action, shift_label, shift_weight, demo_weight, reward, next_state, terminated)

            if terminated or episode_done:
                while not self._nstep_buffer.is_empty():
                    state, action, shift_label, shift_weight, demo_weight, reward = self._nstep_buffer.get()
                    self._append(state, action, shift_label, shift_weight, demo_weight, reward, next_state, terminated)

        else:
            self._append(state, action, shift_label, shift_weight, demo_weight, reward, next_state, terminated)

    def _append(self, state, action, shift_label, shift_weight, demo_weight, reward, next_state, done):
        self._states[self._p, ...] = state
        self._actions[self._p, ...] = action
        self._shift_labels[self._p, ...] = shift_label
        self._shift_weights[self._p, ...] = shift_weight
        self._demo_weights[self._p, ...] = demo_weight
        self._rewards[self._p, ...] = reward
        self._next_states[self._p, ...] = next_state
        self._dones[self._p, ...] = done

        self._n = min(self._n + 1, self._memory_size)
        self._p = (self._p + 1) % self._memory_size

    def sample(self, batch_size, device=torch.device('cpu')):
        assert isinstance(batch_size, int) and batch_size > 0

        idxes = self._sample_idxes(batch_size)
        return self._sample_batch(idxes, batch_size, device)

    def iter_batches(self, batch_size, device=torch.device('cpu'), num_samples=None, shuffle=True):
        assert isinstance(batch_size, int) and batch_size > 0

        sample_count = self._n if num_samples is None else min(int(num_samples), self._n)
        if sample_count <= 0:
            return

        idxes = np.arange(sample_count)
        if shuffle:
            np.random.shuffle(idxes)

        for start in range(0, sample_count, batch_size):
            batch_idxes = idxes[start:start + batch_size]
            yield self._sample_batch(batch_idxes, len(batch_idxes), device)

    def _sample_idxes(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample_batch(self, idxes, batch_size, device):
        states = torch.tensor(
            self._states[idxes], dtype=torch.float, device=device)
        actions = torch.tensor(
            self._actions[idxes], dtype=torch.float, device=device)
        shift_labels = torch.tensor(
            self._shift_labels[idxes], dtype=torch.float, device=device)
        shift_weights = torch.tensor(
            self._shift_weights[idxes], dtype=torch.float, device=device)
        demo_weights = torch.tensor(
            self._demo_weights[idxes], dtype=torch.float, device=device)
        rewards = torch.tensor(
            self._rewards[idxes], dtype=torch.float, device=device)
        dones = torch.tensor(
            self._dones[idxes], dtype=torch.float, device=device)
        next_states = torch.tensor(
            self._next_states[idxes], dtype=torch.float, device=device)

        return states, actions, shift_labels, shift_weights, demo_weights, rewards, next_states, dones

    def __len__(self):
        return self._n


class EnsembleBuffer(ReplayBuffer):
    """
    Ensemble of an offline dataloader and an online replay buffer.
    """
    def __init__(self, memory_size, state_shape, action_shape, offline_buffer_size, gamma=0.99, nstep=1, shift_shape=(2,)):
        # Initialize the offline buffer with the specified offline_buffer_size
        self._offline = ReplayBuffer(offline_buffer_size, state_shape, action_shape, gamma, nstep, shift_shape)

        # Initialize the online buffer using the parent class constructor
        super().__init__(memory_size, state_shape, action_shape, gamma, nstep, shift_shape)
        self._online = False

    def online(self, enable):
        """Enable or disable sampling from the online buffer."""
        self._online = enable
        if enable:
            logger.info("Switching to Online buffer.")

    def append(self, state, action, reward, next_state, terminated, episode_done=None, shift_label=None, shift_weight=None, demo_weight=None):
        if self._online:
            super().append(state, action, reward, next_state, terminated, episode_done, shift_label, shift_weight, demo_weight)
        else:
            self._offline.append(state, action, reward, next_state, terminated, episode_done, shift_label, shift_weight, demo_weight)

    def __len__(self):
        offline_len = len(self._offline)
        online_len = super().__len__()
        return offline_len + online_len

    @property
    def online_size(self):
        return super().__len__()

    @property
    def offline_size(self):
        return len(self._offline)

    def sample(self, batch_size, device=torch.device('cpu'), offline_ratio=0.5):
        """Sample a batch of data from the two buffers."""
        offline_ratio = float(np.clip(offline_ratio, 0.0, 1.0))

        if not self._online or self.online_size == 0:
            return self._offline.sample(batch_size, device)
        if self.offline_size == 0 or offline_ratio <= 0.0:
            return super().sample(batch_size, device)

        offline_count = int(round(batch_size * offline_ratio))
        offline_count = max(0, min(batch_size, offline_count))
        online_count = batch_size - offline_count

        if online_count == 0:
            return self._offline.sample(batch_size, device)
        if offline_count == 0:
            return super().sample(batch_size, device)

        offline_batch = self._offline.sample(offline_count, device)
        online_batch = super().sample(online_count, device)

        return tuple(torch.cat([offline_tensor, online_tensor], dim=0)
                     for offline_tensor, online_tensor in zip(offline_batch, online_batch))

    def iter_batches(self, batch_size, device=torch.device('cpu'), num_samples=None, shuffle=True):
        if len(self._offline) > 0:
            yield from self._offline.iter_batches(batch_size, device, num_samples, shuffle)
        else:
            yield from super().iter_batches(batch_size, device, num_samples, shuffle)
