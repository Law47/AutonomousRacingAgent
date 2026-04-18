import os
import torch
from torch.optim import Adam
from torch.nn import functional as F

from .base import Algorithm
from discor.network import TwinnedStateActionFunction, GaussianPolicy
from discor.utils import disable_gradients, soft_update, update_params, \
    assert_action


import logging
logger = logging.getLogger(__name__)

SAC_TRAINING_STATE_FILE = "sac_training_state.pth"

class SAC(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0):
        super().__init__(
            state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        # Build networks.
        self._policy_net = GaussianPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._online_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device)
        self._target_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        # Optimizers.
        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        # Target entropy is -|A|.
        self._target_entropy = -float(self._action_dim)

        # We optimize log(alpha), instead of alpha.
        self._log_alpha = torch.zeros(
            1, device=self._device, requires_grad=True)
        self._alpha = self._log_alpha.detach().exp()
        self._alpha_optim = Adam([self._log_alpha], lr=entropy_lr)

        self._target_update_coef = target_update_coef
        self.update_entropy = True

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action, entropies, _ = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, entropies

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            _, entropies, action = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, entropies

    def update_target_networks(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)

    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        stats = self.update_policy_and_entropy(batch, writer)
        self.update_q_functions(batch, writer)
        return stats

    def update_behavior_cloning(self, batch, writer, action_weights=None, loss_coef=1.0):
        states, demo_actions, _, _, _ = batch

        _, _, deterministic_actions = self._policy_net(states)
        bc_loss = F.mse_loss(deterministic_actions, demo_actions, reduction='none')
        if action_weights is not None:
            action_weights = torch.as_tensor(
                action_weights,
                dtype=bc_loss.dtype,
                device=bc_loss.device,
            ).view(1, -1)
            bc_loss = bc_loss * action_weights
        bc_loss = bc_loss.mean() * float(loss_coef)

        update_params(self._policy_optim, bc_loss)

        stats = {
            "behavior_clone_loss": bc_loss.detach().item(),
            "behavior_clone_shift_up_mean": deterministic_actions[:, 3].detach().mean().item()
            if self._action_dim >= 5
            else 0.0,
            "behavior_clone_shift_down_mean": deterministic_actions[:, 4].detach().mean().item()
            if self._action_dim >= 5
            else 0.0,
        }
        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar('loss/behavior_clone', stats["behavior_clone_loss"], self._learning_steps)
            if self._action_dim >= 5:
                writer.add_scalar(
                    'stats/behavior_clone_shift_up_mean',
                    stats["behavior_clone_shift_up_mean"],
                    self._learning_steps,
                )
                writer.add_scalar(
                    'stats/behavior_clone_shift_down_mean',
                    stats["behavior_clone_shift_down_mean"],
                    self._learning_steps,
                )
        return stats

    def update_policy_and_entropy(self, batch, writer):
        states, actions, rewards, next_states, dones = batch

        # Update policy.
        policy_loss, entropies = self.calc_policy_loss(states)
        update_params(self._policy_optim, policy_loss)

        # Update the entropy coefficient.
        entropy_loss = 0.
        if self.update_entropy:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self._alpha_optim, entropy_loss)
            entropy_loss = entropy_loss.detach().item()
        self._alpha = self._log_alpha.detach().exp()

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/entropy', entropy_loss,
                self._learning_steps)
            writer.add_scalar(
                'stats/alpha', self._alpha.item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self._learning_steps)

            return {"policy_loss": policy_loss.detach().item(),
                    "entropy_loss": entropy_loss,
                    "alpha": self._alpha.item(), "entropy": entropies.detach().mean().item()}

    def calc_policy_loss(self, states):
        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self._online_q_net(states, sampled_actions)
        qs = torch.min(qs1, qs2)

        # Policy objective is maximization of (Q + alpha * entropy).
        assert qs.shape == entropies.shape
        policy_loss = torch.mean((- qs - self._alpha * entropies))

        return policy_loss, entropies.detach_()

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self._log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None):
        states, actions, rewards, next_states, dones = batch

        # Calculate current and target Q values.
        curr_qs1, curr_qs2 = self.calc_current_qs(states, actions)
        target_qs = self.calc_target_qs(rewards, next_states, dones)

        # Update Q functions.
        q_loss, mean_q1, mean_q2 = \
            self.calc_q_loss(curr_qs1, curr_qs2, target_qs, imp_ws1, imp_ws2)
        update_params(self._q_optim, q_loss)

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q', q_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)

        # Return there values for DisCor algorithm.
        return curr_qs1.detach(), curr_qs2.detach(), target_qs

    def calc_current_qs(self, states, actions):
        curr_qs1, curr_qs2 = self._online_q_net(states, actions)
        return curr_qs1, curr_qs2

    def calc_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_states)
            next_qs1, next_qs2 = self._target_q_net(next_states, next_actions)
            next_qs = \
                torch.min(next_qs1, next_qs2) + self._alpha * next_entropies

        assert rewards.shape == next_qs.shape
        target_qs = rewards + (1.0 - dones) * self._discount * next_qs

        return target_qs

    def calc_q_loss(self, curr_qs1, curr_qs2, target_qs, imp_ws1=None,
                    imp_ws2=None):
        assert imp_ws1 is None or imp_ws1.shape == curr_qs1.shape
        assert imp_ws2 is None or imp_ws2.shape == curr_qs2.shape
        assert not target_qs.requires_grad
        assert curr_qs1.shape == target_qs.shape

        # Q loss is mean squared TD errors with importance weights.
        if imp_ws1 is None:
            q1_loss = torch.mean((curr_qs1 - target_qs).pow(2))
            q2_loss = torch.mean((curr_qs2 - target_qs).pow(2))

        else:
            q1_loss = torch.sum((curr_qs1 - target_qs).pow(2) * imp_ws1)
            q2_loss = torch.sum((curr_qs2 - target_qs).pow(2) * imp_ws2)

        # Mean Q values for logging.
        mean_q1 = curr_qs1.detach().mean().item()
        mean_q2 = curr_qs2.detach().mean().item()

        return q1_loss + q2_loss, mean_q1, mean_q2

    def _optimizer_to_device(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self._device)

    def save_training_state(self, save_dir):
        state = {
            "format_version": 1,
            "learning_steps": self._learning_steps,
            "log_alpha": self._log_alpha.detach().cpu(),
            "alpha": self._alpha.detach().cpu(),
            "policy_optimizer": self._policy_optim.state_dict(),
            "q_optimizer": self._q_optim.state_dict(),
            "alpha_optimizer": self._alpha_optim.state_dict(),
            "target_entropy": self._target_entropy,
            "target_update_coef": self._target_update_coef,
            "update_entropy": self.update_entropy,
        }
        torch.save(state, os.path.join(save_dir, SAC_TRAINING_STATE_FILE))

    def load_training_state(self, load_dir):
        state_path = os.path.join(load_dir, SAC_TRAINING_STATE_FILE)
        if not os.path.exists(state_path):
            logger.warning(
                "%s not found in %s; loaded model weights only. "
                "Optimizer state, entropy temperature, and SAC learning step will start fresh.",
                SAC_TRAINING_STATE_FILE,
                load_dir,
            )
            return False

        state = torch.load(state_path, map_location=self._device)
        self._learning_steps = int(state.get("learning_steps", self._learning_steps))

        loaded_log_alpha = state.get("log_alpha", None)
        if loaded_log_alpha is not None:
            loaded_log_alpha = loaded_log_alpha.to(self._device).view_as(self._log_alpha)
            self._log_alpha.data.copy_(loaded_log_alpha)
        self._alpha = self._log_alpha.detach().exp()

        if "policy_optimizer" in state:
            self._policy_optim.load_state_dict(state["policy_optimizer"])
            self._optimizer_to_device(self._policy_optim)
        if "q_optimizer" in state:
            self._q_optim.load_state_dict(state["q_optimizer"])
            self._optimizer_to_device(self._q_optim)
        if "alpha_optimizer" in state:
            self._alpha_optim.load_state_dict(state["alpha_optimizer"])
            self._optimizer_to_device(self._alpha_optim)

        self._target_entropy = float(state.get("target_entropy", self._target_entropy))
        self._target_update_coef = float(state.get("target_update_coef", self._target_update_coef))
        self.update_entropy = bool(state.get("update_entropy", self.update_entropy))
        logger.info("Loaded SAC training state from %s", state_path)
        return True

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
        self.save_training_state(save_dir)

    def load_models(self, load_dir, load_training_state=True):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))
        if load_training_state:
            self.load_training_state(load_dir)
        else:
            logger.info("Skipping SAC training-state load; loaded network weights only from %s", load_dir)
