import os
import torch
from torch.optim import Adam
from torch.nn import functional as F

from .base import Algorithm
from discor.network import (
    DiscreteShiftPolicy,
    GaussianPolicy,
    TwinnedDiscreteStateActionFunction,
    TwinnedStateActionFunction,
)
from discor.utils import disable_gradients, soft_update, update_params, \
    assert_action


import logging
logger = logging.getLogger(__name__)

SAC_TRAINING_STATE_FILENAME = 'sac_training_state.pth'

class SAC(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                  target_update_coef=0.005, log_interval=10, seed=0,
                  target_entropy=None, shift_enabled=True, shift_lr=0.0003,
                  shift_hidden_units=[256, 256], shift_dim=2,
                  shift_loss_weight=0.0, shift_pos_weight=None,
                  shift_bc_loss_weight=None, shift_bc_class_weights=None,
                  shift_bc_focal_gamma=0.0, shift_threshold=0.5,
                  shift_entropy_lr=None, shift_target_entropy=0.2,
                  shift_reward_scale=1.0, demo_actor_loss_weight=0.0,
                  demo_actor_min_loss_weight=0.0, demo_actor_decay_steps=0,
                  demo_actor_bc_mode="q_filter",
                  demo_actor_temperature=2.0,
                  demo_actor_max_weight=20.0,
                  demo_actor_q_filter_margin=0.0,
                  demo_actor_q_filter_warmup_steps=1000):
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
        self._target_entropy = -float(self._action_dim) if target_entropy is None else float(target_entropy)

        # We optimize log(alpha), instead of alpha.
        self._log_alpha = torch.zeros(
            1, device=self._device, requires_grad=True)
        self._alpha = self._log_alpha.detach().exp()
        self._alpha_optim = Adam([self._log_alpha], lr=entropy_lr)

        self._target_update_coef = target_update_coef
        self.update_entropy = True
        self._demo_actor_loss_weight = max(float(demo_actor_loss_weight or 0.0), 0.0)
        self._demo_actor_min_loss_weight = min(
            max(float(demo_actor_min_loss_weight or 0.0), 0.0),
            self._demo_actor_loss_weight,
        )
        self._demo_actor_decay_steps = max(int(demo_actor_decay_steps or 0), 0)
        self._demo_actor_bc_mode = str(demo_actor_bc_mode or "bc").lower()
        self._demo_actor_temperature = max(float(demo_actor_temperature or 1.0), 1e-6)
        self._demo_actor_max_weight = max(float(demo_actor_max_weight or 1.0), 1.0)
        self._demo_actor_q_filter_margin = float(demo_actor_q_filter_margin or 0.0)
        self._demo_actor_q_filter_warmup_steps = max(
            int(demo_actor_q_filter_warmup_steps or 0),
            0,
        )

        self._shift_enabled = bool(shift_enabled)
        self._shift_dim = int(shift_dim)
        self._shift_action_count = self._shift_dim + 1
        self._shift_threshold = float(shift_threshold)
        self._shift_learning_steps = 0
        self._shift_reward_scale = float(shift_reward_scale)
        self._shift_target_entropy = (
            float(shift_target_entropy)
            if shift_target_entropy is not None
            else 0.2
        )
        if shift_bc_loss_weight is None:
            shift_bc_loss_weight = shift_loss_weight
        self._shift_bc_loss_weight = max(float(shift_bc_loss_weight or 0.0), 0.0)
        self._shift_bc_focal_gamma = max(float(shift_bc_focal_gamma or 0.0), 0.0)
        if shift_bc_class_weights is None and shift_pos_weight is not None:
            shift_bc_class_weights = [1.0] + [float(shift_pos_weight)] * self._shift_dim

        self._shift_bc_class_weights = None
        if shift_bc_class_weights is not None:
            class_weights = torch.as_tensor(
                shift_bc_class_weights,
                dtype=torch.float,
                device=self._device,
            ).view(-1)
            if class_weights.numel() == self._shift_dim:
                class_weights = torch.cat([
                    torch.ones(1, dtype=torch.float, device=self._device),
                    class_weights,
                ])
            if class_weights.numel() != self._shift_action_count:
                raise ValueError(
                    "shift_bc_class_weights must contain either "
                    f"{self._shift_dim} shift weights or "
                    f"{self._shift_action_count} action weights"
                )
            self._shift_bc_class_weights = class_weights
        if self._shift_enabled:
            self._shift_policy_net = DiscreteShiftPolicy(
                state_dim=self._state_dim,
                hidden_units=shift_hidden_units,
                shift_dim=self._shift_dim,
            ).to(self._device)
            self._shift_online_q_net = TwinnedDiscreteStateActionFunction(
                state_dim=self._state_dim,
                action_count=self._shift_action_count,
                hidden_units=shift_hidden_units,
            ).to(self._device)
            self._shift_target_q_net = TwinnedDiscreteStateActionFunction(
                state_dim=self._state_dim,
                action_count=self._shift_action_count,
                hidden_units=shift_hidden_units,
            ).to(self._device).eval()
            self._shift_target_q_net.load_state_dict(self._shift_online_q_net.state_dict())
            disable_gradients(self._shift_target_q_net)

            self._shift_policy_optim = Adam(self._shift_policy_net.parameters(), lr=shift_lr)
            self._shift_q_optim = Adam(self._shift_online_q_net.parameters(), lr=shift_lr)
            self._shift_log_alpha = torch.zeros(
                1, device=self._device, requires_grad=True)
            self._shift_alpha = self._shift_log_alpha.detach().exp()
            self._shift_alpha_optim = Adam(
                [self._shift_log_alpha],
                lr=shift_entropy_lr if shift_entropy_lr is not None else shift_lr,
            )
            self._shift_net = self._shift_policy_net
        else:
            self._shift_net = None
            self._shift_policy_net = None
            self._shift_online_q_net = None
            self._shift_target_q_net = None
            self._shift_policy_optim = None
            self._shift_q_optim = None
            self._shift_log_alpha = None
            self._shift_alpha = None
            self._shift_alpha_optim = None

    def unpack_batch(self, batch):
        states, actions, shift_actions, _shift_weights, _demo_weights, rewards, next_states, dones = (
            self.unpack_batch_with_metadata(batch)
        )
        return states, actions, shift_actions, rewards, next_states, dones

    def unpack_batch_with_shift_weights(self, batch):
        states, actions, shift_actions, shift_weights, _demo_weights, rewards, next_states, dones = (
            self.unpack_batch_with_metadata(batch)
        )
        return states, actions, shift_actions, shift_weights, rewards, next_states, dones

    def unpack_batch_with_metadata(self, batch):
        if len(batch) == 8:
            return batch
        if len(batch) == 7:
            states, actions, shift_actions, shift_weights, rewards, next_states, dones = batch
            demo_weights = torch.zeros(
                (states.shape[0], 1),
                dtype=states.dtype,
                device=states.device,
            )
            return states, actions, shift_actions, shift_weights, demo_weights, rewards, next_states, dones
        if len(batch) == 6:
            states, actions, shift_actions, rewards, next_states, dones = batch
            shift_weights = torch.ones(
                (states.shape[0], 1),
                dtype=states.dtype,
                device=states.device,
            )
            demo_weights = torch.zeros(
                (states.shape[0], 1),
                dtype=states.dtype,
                device=states.device,
            )
            return states, actions, shift_actions, shift_weights, demo_weights, rewards, next_states, dones

        states, actions, rewards, next_states, dones = batch
        shift_actions = torch.zeros(
            (states.shape[0], self._shift_dim),
            dtype=states.dtype,
            device=states.device,
        )
        shift_weights = torch.zeros(
            (states.shape[0], 1),
            dtype=states.dtype,
            device=states.device,
        )
        demo_weights = torch.zeros(
            (states.shape[0], 1),
            dtype=states.dtype,
            device=states.device,
        )
        return states, actions, shift_actions, shift_weights, demo_weights, rewards, next_states, dones

    def explore(self, state, shift_deterministic=False, shift_epsilon=0.0, shift_action_mask=None):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action, entropies, _ = self._policy_net(state)
            shift_action, shift_probs = self._sample_shift_from_tensor(
                state,
                deterministic=shift_deterministic,
                action_mask=shift_action_mask,
                epsilon=shift_epsilon,
            )
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, {
            "entropies": entropies,
            "shift_action": shift_action,
            "shift_probs": shift_probs,
        }

    def exploit(self, state, shift_action_mask=None):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            _, entropies, action = self._policy_net(state)
            shift_action, shift_probs = self._sample_shift_from_tensor(
                state,
                deterministic=True,
                action_mask=shift_action_mask,
            )
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, {
            "entropies": entropies,
            "shift_action": shift_action,
            "shift_probs": shift_probs,
        }

    def _sample_shift_from_tensor(self, states, deterministic=False, action_mask=None, epsilon=0.0):
        if not self._shift_enabled:
            zeros = torch.zeros((states.shape[0], self._shift_dim), device=states.device)
            return zeros.cpu().numpy()[0], zeros.cpu().numpy()[0]

        _, actions, probs, _ = self._shift_policy_net.sample(
            states,
            deterministic=deterministic,
            threshold=self._shift_threshold,
            action_mask=action_mask,
            epsilon=epsilon,
        )
        shift_probs = probs[:, 1:]
        return actions.cpu().numpy()[0], shift_probs.cpu().numpy()[0]

    def update_target_networks(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)
        if self._shift_enabled:
            soft_update(
                self._shift_target_q_net,
                self._shift_online_q_net,
                self._target_update_coef,
            )

    def update_online_networks(self, batch, writer, train_shift_rl=True):
        self._learning_steps += 1
        stats = self.update_policy_and_entropy(batch, writer)
        self.update_q_functions(batch, writer)
        shift_stats = self.update_shift_model_from_batch(batch, writer) if train_shift_rl else None
        if shift_stats:
            if stats is None:
                stats = {}
            stats.update(shift_stats)
        return stats

    def update_policy_and_entropy(self, batch, writer):
        states, actions, _shift_actions, _shift_weights, demo_weights, _rewards, _next_states, _dones = (
            self.unpack_batch_with_metadata(batch)
        )

        # Update policy.
        policy_loss, entropies, demo_actor_stats = self.calc_policy_loss(
            states,
            demo_actions=actions,
            demo_weights=demo_weights,
        )
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
            writer.add_scalar(
                'loss/demo_actor_bc',
                demo_actor_stats["loss"],
                self._learning_steps)
            writer.add_scalar(
                'stats/demo_actor_loss_weight',
                demo_actor_stats["loss_weight"],
                self._learning_steps)
            writer.add_scalar(
                'stats/demo_actor_sample_fraction',
                demo_actor_stats["sample_fraction"],
                self._learning_steps)
            writer.add_scalar(
                'stats/demo_actor_effective_fraction',
                demo_actor_stats["effective_fraction"],
                self._learning_steps)
            writer.add_scalar(
                'stats/demo_actor_mean_advantage',
                demo_actor_stats["mean_advantage"],
                self._learning_steps)

            return {"policy_loss": policy_loss.detach().item(),
                    "entropy_loss": entropy_loss,
                    "alpha": self._alpha.item(),
                    "entropy": entropies.detach().mean().item(),
                    "demo_actor_bc_loss": demo_actor_stats["loss"],
                    "demo_actor_loss_weight": demo_actor_stats["loss_weight"],
                    "demo_actor_sample_fraction": demo_actor_stats["sample_fraction"],
                    "demo_actor_effective_fraction": demo_actor_stats["effective_fraction"],
                    "demo_actor_mean_advantage": demo_actor_stats["mean_advantage"]}

    def calc_policy_loss(self, states, demo_actions=None, demo_weights=None):
        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, deterministic_actions = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self._online_q_net(states, sampled_actions)
        qs = torch.min(qs1, qs2)

        # Policy objective is maximization of (Q + alpha * entropy).
        assert qs.shape == entropies.shape
        policy_loss = torch.mean((- qs - self._alpha * entropies))

        demo_actor_stats = self.empty_demo_actor_stats()
        demo_loss_weight = self.demo_actor_loss_weight()
        if demo_loss_weight > 0.0 and demo_actions is not None and demo_weights is not None:
            demo_bc_loss, demo_actor_stats = self.calc_demo_actor_bc_loss(
                states,
                demo_actions,
                demo_weights,
                policy_actions=deterministic_actions,
            )
            demo_actor_stats["loss_weight"] = demo_loss_weight
            if demo_bc_loss is not None:
                policy_loss = policy_loss + demo_loss_weight * demo_bc_loss

        return policy_loss, entropies.detach(), demo_actor_stats

    def demo_actor_loss_weight(self):
        if self._demo_actor_loss_weight <= 0.0:
            return 0.0
        if self._demo_actor_decay_steps <= 0:
            return self._demo_actor_loss_weight
        progress = min(self._learning_steps / self._demo_actor_decay_steps, 1.0)
        return float(
            self._demo_actor_loss_weight
            + (self._demo_actor_min_loss_weight - self._demo_actor_loss_weight) * progress
        )

    def empty_demo_actor_stats(self):
        return {
            "loss": 0.0,
            "loss_weight": self.demo_actor_loss_weight(),
            "sample_fraction": 0.0,
            "effective_fraction": 0.0,
            "mean_advantage": 0.0,
            "mean_weight": 0.0,
        }

    def calc_demo_actor_bc_loss(self, states, demo_actions, demo_weights, policy_actions=None):
        if self._demo_actor_bc_mode in ("off", "none"):
            return None, self.empty_demo_actor_stats()

        demo_weights = demo_weights.reshape(-1, 1).clamp_min(0.0)
        demo_mask = demo_weights > 0.0
        if not torch.any(demo_mask):
            return None, self.empty_demo_actor_stats()

        if policy_actions is None:
            _sampled_actions, _entropies, policy_actions = self._policy_net(states)

        per_sample_loss = (policy_actions - demo_actions).pow(2).mean(dim=1, keepdim=True)
        value_weights = demo_weights
        mean_advantage = 0.0

        mode = self._demo_actor_bc_mode
        use_value_weights = (
            mode in ("q_filter", "awac", "awac_q_filter")
            and self._learning_steps >= self._demo_actor_q_filter_warmup_steps
        )
        if use_value_weights:
            with torch.no_grad():
                demo_qs1, demo_qs2 = self._online_q_net(states, demo_actions)
                policy_qs1, policy_qs2 = self._online_q_net(states, policy_actions.detach())
                demo_qs = torch.min(demo_qs1, demo_qs2)
                policy_qs = torch.min(policy_qs1, policy_qs2)
                advantages = demo_qs - policy_qs
                mean_advantage = advantages[demo_mask].mean().item()

                if mode == "q_filter":
                    value_weights = demo_weights * (
                        advantages > self._demo_actor_q_filter_margin
                    ).float()
                elif mode == "awac":
                    value_weights = demo_weights * torch.exp(
                        advantages / self._demo_actor_temperature
                    ).clamp(max=self._demo_actor_max_weight)
                elif mode == "awac_q_filter":
                    awac_weights = torch.exp(
                        advantages / self._demo_actor_temperature
                    ).clamp(max=self._demo_actor_max_weight)
                    q_filter = (advantages > self._demo_actor_q_filter_margin).float()
                    value_weights = demo_weights * awac_weights * q_filter

        weight_sum = value_weights.sum()
        if weight_sum.item() <= 0.0:
            stats = self.empty_demo_actor_stats()
            stats["sample_fraction"] = demo_mask.float().mean().item()
            stats["mean_advantage"] = mean_advantage
            return None, stats

        bc_loss = (per_sample_loss * value_weights).sum() / weight_sum.clamp_min(1e-6)
        with torch.no_grad():
            effective_mask = value_weights > 0.0
            stats = {
                "loss": bc_loss.detach().item(),
                "loss_weight": self.demo_actor_loss_weight(),
                "sample_fraction": demo_mask.float().mean().item(),
                "effective_fraction": effective_mask.float().mean().item(),
                "mean_advantage": mean_advantage,
                "mean_weight": value_weights[effective_mask].mean().item()
                if torch.any(effective_mask)
                else 0.0,
            }
        return bc_loss, stats

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self._log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None):
        states, actions, _shift_actions, rewards, next_states, dones = self.unpack_batch(batch)

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

    def update_shift_model_from_batch(self, batch, writer):
        if not self._shift_enabled:
            return None

        states, actions, shift_actions, shift_weights, rewards, next_states, dones = (
            self.unpack_batch_with_shift_weights(batch)
        )
        self._shift_learning_steps += 1
        shift_q_loss, shift_mean_q1, shift_mean_q2, shift_td_abs = self.update_shift_q_functions(
            states,
            shift_actions,
            rewards,
            next_states,
            dones,
        )
        (
            shift_policy_loss,
            shift_entropy_loss,
            shift_entropy,
            shift_action_rates,
            shift_policy_rates,
            shift_aux_bc_loss,
        ) = self.update_shift_policy_and_entropy(states, shift_actions, shift_weights)

        if self._shift_learning_steps % self._log_interval == 0:
            q_loss_value = shift_q_loss.detach().item()
            policy_loss_value = shift_policy_loss.detach().item()
            entropy_loss_value = shift_entropy_loss.detach().item()
            writer.add_scalar('loss/shift_Q', q_loss_value, self._shift_learning_steps)
            writer.add_scalar('loss/shift_policy', policy_loss_value, self._shift_learning_steps)
            writer.add_scalar('loss/shift_entropy', entropy_loss_value, self._shift_learning_steps)
            writer.add_scalar('loss/shift_aux_bc', shift_aux_bc_loss, self._shift_learning_steps)
            writer.add_scalar('stats/shift_alpha', self._shift_alpha.item(), self._shift_learning_steps)
            writer.add_scalar('stats/shift_entropy', shift_entropy, self._shift_learning_steps)
            writer.add_scalar('stats/shift_mean_Q1', shift_mean_q1, self._shift_learning_steps)
            writer.add_scalar('stats/shift_mean_Q2', shift_mean_q2, self._shift_learning_steps)
            writer.add_scalar('stats/shift_td_abs', shift_td_abs, self._shift_learning_steps)
            writer.add_scalar('stats/shift_noop_action_rate', shift_action_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_up_action_rate', shift_action_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_down_action_rate', shift_action_rates[2], self._shift_learning_steps)
            writer.add_scalar('stats/shift_noop_policy_prob', shift_policy_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_up_policy_prob', shift_policy_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_down_policy_prob', shift_policy_rates[2], self._shift_learning_steps)
            return {
                "shift_q_loss": q_loss_value,
                "shift_policy_loss": policy_loss_value,
                "shift_entropy_loss": entropy_loss_value,
                "shift_aux_bc_loss": shift_aux_bc_loss,
                "shift_alpha": self._shift_alpha.item(),
                "shift_entropy": shift_entropy,
                "shift_mean_q1": shift_mean_q1,
                "shift_mean_q2": shift_mean_q2,
                "shift_td_abs": shift_td_abs,
                "shift_noop_action_rate": shift_action_rates[0],
                "shift_up_action_rate": shift_action_rates[1],
                "shift_down_action_rate": shift_action_rates[2],
                "shift_noop_policy_prob": shift_policy_rates[0],
                "shift_up_policy_prob": shift_policy_rates[1],
                "shift_down_policy_prob": shift_policy_rates[2],
            }
        return None

    def update_shift_behavior_clone_from_batch(self, batch, writer):
        if not self._shift_enabled:
            return None

        states, _actions, shift_actions, shift_weights, _rewards, _next_states, _dones = (
            self.unpack_batch_with_shift_weights(batch)
        )
        action_indices = self.shift_action_indices(shift_actions)
        logits, probs = self._shift_policy_net(states)
        bc_loss, bc_stats = self.calc_shift_bc_loss(logits, action_indices, shift_weights)
        if bc_loss is None:
            return None

        self._shift_learning_steps += 1
        update_params(self._shift_policy_optim, bc_loss)

        if self._shift_learning_steps % self._log_interval == 0:
            loss_value = bc_loss.detach().item()
            accuracy = bc_stats["accuracy"]
            action_rates = bc_stats["action_rates"]
            policy_rates = bc_stats["policy_rates"]
            writer.add_scalar('loss/shift_bc', loss_value, self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_accuracy', accuracy, self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_teacher_fraction', bc_stats["teacher_fraction"], self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_noop_action_rate', action_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_up_action_rate', action_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_down_action_rate', action_rates[2], self._shift_learning_steps)
            writer.add_scalar('stats/shift_noop_policy_prob', policy_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_up_policy_prob', policy_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_down_policy_prob', policy_rates[2], self._shift_learning_steps)
            return {
                "shift_bc_loss": loss_value,
                "shift_bc_accuracy": accuracy,
                "shift_bc_teacher_fraction": bc_stats["teacher_fraction"],
                "shift_bc_noop_action_rate": action_rates[0],
                "shift_bc_up_action_rate": action_rates[1],
                "shift_bc_down_action_rate": action_rates[2],
                "shift_noop_policy_prob": policy_rates[0],
                "shift_up_policy_prob": policy_rates[1],
                "shift_down_policy_prob": policy_rates[2],
            }
        return None

    def shift_action_indices(self, shift_actions):
        shift_actions = shift_actions.clamp(0.0, 1.0)
        shift_up = shift_actions[:, 0] > 0.5
        shift_down = shift_actions[:, 1] > 0.5 if self._shift_dim > 1 else torch.zeros_like(shift_up)
        valid_up = shift_up & ~shift_down
        valid_down = shift_down & ~shift_up
        action_indices = torch.zeros(
            shift_actions.shape[0],
            dtype=torch.long,
            device=shift_actions.device,
        )
        action_indices = torch.where(valid_up, torch.ones_like(action_indices), action_indices)
        if self._shift_dim > 1:
            action_indices = torch.where(
                valid_down,
                torch.full_like(action_indices, 2),
                action_indices,
            )
        return action_indices

    def calc_shift_bc_loss(self, logits, action_indices, shift_weights):
        shift_weights = shift_weights.reshape(-1).clamp_min(0.0)
        teacher_mask = shift_weights > 0.0
        if not torch.any(teacher_mask):
            return None, {}

        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, action_indices.view(-1, 1)).squeeze(1)
        sample_losses = -chosen_log_probs

        if self._shift_bc_class_weights is not None:
            sample_losses = sample_losses * self._shift_bc_class_weights[action_indices]
        if self._shift_bc_focal_gamma > 0.0:
            chosen_probs = chosen_log_probs.exp().clamp(0.0, 1.0)
            focal_weights = (1.0 - chosen_probs).pow(self._shift_bc_focal_gamma)
            sample_losses = sample_losses * focal_weights

        weighted_losses = sample_losses * shift_weights
        loss = weighted_losses.sum() / shift_weights.sum().clamp_min(1e-6)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            teacher_indices = action_indices[teacher_mask]
            accuracy = (
                (predictions[teacher_mask] == teacher_indices).float().mean().item()
                if teacher_indices.numel() > 0
                else 0.0
            )
            action_rates = [
                (teacher_indices == index).float().mean().item()
                if teacher_indices.numel() > 0
                else 0.0
                for index in range(self._shift_action_count)
            ]
            policy_rates = (
                probs[teacher_mask].mean(dim=0).detach().cpu().tolist()
                if teacher_indices.numel() > 0
                else probs.mean(dim=0).detach().cpu().tolist()
            )
            teacher_fraction = teacher_mask.float().mean().item()

        return loss, {
            "accuracy": accuracy,
            "action_rates": action_rates,
            "policy_rates": policy_rates,
            "teacher_fraction": teacher_fraction,
        }

    def update_shift_q_functions(self, states, shift_actions, rewards, next_states, dones):
        curr_qs1, curr_qs2 = self.calc_shift_current_qs(states, shift_actions)
        target_qs = self.calc_shift_target_qs(rewards, next_states, dones)
        q1_loss = F.mse_loss(curr_qs1, target_qs)
        q2_loss = F.mse_loss(curr_qs2, target_qs)
        q_loss = q1_loss + q2_loss
        update_params(self._shift_q_optim, q_loss)

        with torch.no_grad():
            td_abs = torch.mean((torch.min(curr_qs1, curr_qs2) - target_qs).abs()).item()
            mean_q1 = curr_qs1.mean().item()
            mean_q2 = curr_qs2.mean().item()
        return q_loss, mean_q1, mean_q2, td_abs

    def calc_shift_current_qs(self, states, shift_actions):
        action_indices = self.shift_action_indices(shift_actions)
        qs1, qs2 = self._shift_online_q_net(states)
        action_indices = action_indices.view(-1, 1)
        return qs1.gather(1, action_indices), qs2.gather(1, action_indices)

    def calc_shift_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            _, next_probs = self._shift_policy_net(next_states)
            next_log_probs = torch.log(next_probs.clamp_min(1e-8))
            next_qs1, next_qs2 = self._shift_target_q_net(next_states)
            next_qs = torch.min(next_qs1, next_qs2)
            next_values = (
                next_probs * (next_qs - self._shift_alpha * next_log_probs)
            ).sum(dim=1, keepdim=True)
            target_qs = self._shift_reward_scale * rewards + (1.0 - dones) * self._discount * next_values
        return target_qs

    def update_shift_policy_and_entropy(self, states, shift_actions, shift_weights=None):
        policy_loss, entropy, probs, aux_bc_loss = self.calc_shift_policy_loss(
            states,
            shift_actions=shift_actions,
            shift_weights=shift_weights,
        )
        update_params(self._shift_policy_optim, policy_loss)

        entropy_loss = self.calc_shift_entropy_loss(entropy.detach())
        update_params(self._shift_alpha_optim, entropy_loss)
        self._shift_alpha = self._shift_log_alpha.detach().exp()

        with torch.no_grad():
            action_indices = self.shift_action_indices(shift_actions)
            action_rates = [
                (action_indices == index).float().mean().item()
                for index in range(self._shift_action_count)
            ]
            policy_rates = probs.mean(dim=0).detach().cpu().tolist()
            entropy_value = entropy.mean().item()

        return policy_loss, entropy_loss, entropy_value, action_rates, policy_rates, aux_bc_loss

    def calc_shift_policy_loss(self, states, shift_actions=None, shift_weights=None):
        logits, probs = self._shift_policy_net(states)
        log_probs = torch.log(probs.clamp_min(1e-8))
        qs1, qs2 = self._shift_online_q_net(states)
        qs = torch.min(qs1, qs2)
        policy_loss = (probs * (self._shift_alpha * log_probs - qs)).sum(dim=1).mean()
        aux_bc_loss_value = 0.0
        if (
            self._shift_bc_loss_weight > 0.0
            and shift_actions is not None
            and shift_weights is not None
        ):
            action_indices = self.shift_action_indices(shift_actions)
            bc_loss, _bc_stats = self.calc_shift_bc_loss(logits, action_indices, shift_weights)
            if bc_loss is not None:
                policy_loss = policy_loss + self._shift_bc_loss_weight * bc_loss
                aux_bc_loss_value = bc_loss.detach().item()
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        return policy_loss, entropy, probs.detach(), aux_bc_loss_value

    def calc_shift_entropy_loss(self, entropy):
        assert not entropy.requires_grad
        return torch.mean(self._shift_log_alpha * (entropy - self._shift_target_entropy))

    def training_state_dict(self):
        state = {
            "version": 1,
            "learning_steps": int(self._learning_steps),
            "log_alpha": self._log_alpha.detach().cpu(),
            "alpha_optim": self._alpha_optim.state_dict(),
            "policy_optim": self._policy_optim.state_dict(),
            "q_optim": self._q_optim.state_dict(),
            "update_entropy": bool(self.update_entropy),
        }
        if self._shift_enabled:
            state.update({
                "shift_learning_steps": int(self._shift_learning_steps),
                "shift_log_alpha": self._shift_log_alpha.detach().cpu(),
                "shift_alpha_optim": self._shift_alpha_optim.state_dict(),
                "shift_policy_optim": self._shift_policy_optim.state_dict(),
                "shift_q_optim": self._shift_q_optim.state_dict(),
            })
        return state

    def load_training_state_dict(self, state):
        if not isinstance(state, dict):
            logger.warning("SAC training state was %s, expected dict; optimizer state starts fresh", type(state).__name__)
            return False

        self._learning_steps = int(state.get("learning_steps", self._learning_steps))
        self.update_entropy = bool(state.get("update_entropy", self.update_entropy))

        log_alpha = state.get("log_alpha", None)
        if log_alpha is not None:
            with torch.no_grad():
                self._log_alpha.copy_(
                    torch.as_tensor(log_alpha, device=self._device).reshape_as(self._log_alpha)
                )
            self._alpha = self._log_alpha.detach().exp()

        for name, optimizer in (
            ("policy_optim", self._policy_optim),
            ("q_optim", self._q_optim),
            ("alpha_optim", self._alpha_optim),
        ):
            optim_state = state.get(name, None)
            if optim_state is not None:
                try:
                    optimizer.load_state_dict(optim_state)
                except ValueError:
                    logger.warning("Ignoring incompatible SAC optimizer state: %s", name)

        if self._shift_enabled:
            self._shift_learning_steps = int(
                state.get("shift_learning_steps", self._shift_learning_steps)
            )
            shift_log_alpha = state.get("shift_log_alpha", None)
            if shift_log_alpha is not None:
                with torch.no_grad():
                    self._shift_log_alpha.copy_(
                        torch.as_tensor(shift_log_alpha, device=self._device).reshape_as(self._shift_log_alpha)
                    )
                self._shift_alpha = self._shift_log_alpha.detach().exp()

            for name, optimizer in (
                ("shift_policy_optim", self._shift_policy_optim),
                ("shift_q_optim", self._shift_q_optim),
                ("shift_alpha_optim", self._shift_alpha_optim),
            ):
                optim_state = state.get(name, None)
                if optim_state is not None:
                    try:
                        optimizer.load_state_dict(optim_state)
                    except ValueError:
                        logger.warning("Ignoring incompatible SAC optimizer state: %s", name)

        logger.info(
            "loaded SAC training state. learning_steps=%s shift_learning_steps=%s",
            self._learning_steps,
            self._shift_learning_steps if self._shift_enabled else 0,
        )
        return True

    def save_training_state(self, save_dir):
        torch.save(
            self.training_state_dict(),
            os.path.join(save_dir, SAC_TRAINING_STATE_FILENAME),
        )

    def load_training_state(self, load_dir):
        state_path = os.path.join(load_dir, SAC_TRAINING_STATE_FILENAME)
        if not os.path.exists(state_path):
            logger.warning(
                "SAC training state not found at %s; optimizer and entropy states start fresh",
                state_path,
            )
            return False
        try:
            state = torch.load(state_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=self._device)
        return self.load_training_state_dict(state)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
        if self._shift_enabled:
            self._shift_policy_net.save(os.path.join(save_dir, 'shift_net.pth'))
            self._shift_online_q_net.save(os.path.join(save_dir, 'shift_online_q_net.pth'))
            self._shift_target_q_net.save(os.path.join(save_dir, 'shift_target_q_net.pth'))
        self.save_training_state(save_dir)

    def load_models(self, load_dir):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))
        if self._shift_enabled:
            loaded_shift_policy = self._load_optional_shift_model(
                self._shift_policy_net,
                os.path.join(load_dir, 'shift_net.pth'),
                'shift policy',
            )
            loaded_shift_q = self._load_optional_shift_model(
                self._shift_online_q_net,
                os.path.join(load_dir, 'shift_online_q_net.pth'),
                'shift online Q',
            )
            loaded_shift_target_q = self._load_optional_shift_model(
                self._shift_target_q_net,
                os.path.join(load_dir, 'shift_target_q_net.pth'),
                'shift target Q',
            )
            if loaded_shift_policy and not loaded_shift_q:
                logger.warning(
                    "Loaded shift policy without shift Q critics; shift RL critics start fresh."
                )
            if loaded_shift_q and not loaded_shift_target_q:
                self._shift_target_q_net.load_state_dict(self._shift_online_q_net.state_dict())
        self.load_training_state(load_dir)

    def _load_optional_shift_model(self, model, path, description):
        if not os.path.exists(path):
            logger.warning("%s not found at %s; %s starts fresh", description, path, description)
            return False
        try:
            model.load(path)
        except ValueError:
            logger.warning(
                "Ignoring incompatible %s checkpoint at %s; %s starts fresh",
                description,
                path,
                description,
            )
            return False
        return True
