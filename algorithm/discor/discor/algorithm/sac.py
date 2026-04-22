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

class SAC(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                  target_update_coef=0.005, log_interval=10, seed=0,
                  target_entropy=None, shift_enabled=True, shift_lr=0.0003,
                  shift_hidden_units=[256, 256], shift_dim=2,
                  shift_loss_weight=1.0, shift_pos_weight=None,
                  shift_threshold=0.5, shift_entropy_lr=None,
                  shift_target_entropy=0.2, shift_reward_scale=1.0):
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
        if self._shift_enabled:
            if shift_loss_weight != 1.0 or shift_pos_weight is not None:
                logger.warning(
                    "ShiftModel loss_weight/pos_weight are ignored because the shifter "
                    "now trains with discrete SAC instead of supervised BCE."
                )
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
        if len(batch) == 6:
            return batch
        states, actions, rewards, next_states, dones = batch
        shift_actions = torch.zeros(
            (states.shape[0], self._shift_dim),
            dtype=states.dtype,
            device=states.device,
        )
        return states, actions, shift_actions, rewards, next_states, dones

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action, entropies, _ = self._policy_net(state)
            shift_action, shift_probs = self._sample_shift_from_tensor(state, deterministic=False)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, {
            "entropies": entropies,
            "shift_action": shift_action,
            "shift_probs": shift_probs,
        }

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            _, entropies, action = self._policy_net(state)
            shift_action, shift_probs = self._sample_shift_from_tensor(state, deterministic=True)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, {
            "entropies": entropies,
            "shift_action": shift_action,
            "shift_probs": shift_probs,
        }

    def _sample_shift_from_tensor(self, states, deterministic=False):
        if not self._shift_enabled:
            zeros = torch.zeros((states.shape[0], self._shift_dim), device=states.device)
            return zeros.cpu().numpy()[0], zeros.cpu().numpy()[0]

        _, actions, probs, _ = self._shift_policy_net.sample(
            states,
            deterministic=deterministic,
            threshold=self._shift_threshold,
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
        states, actions, _shift_actions, rewards, next_states, dones = self.unpack_batch(batch)

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

        states, actions, shift_actions, rewards, next_states, dones = self.unpack_batch(batch)
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
        ) = self.update_shift_policy_and_entropy(states, shift_actions)

        if self._shift_learning_steps % self._log_interval == 0:
            q_loss_value = shift_q_loss.detach().item()
            policy_loss_value = shift_policy_loss.detach().item()
            entropy_loss_value = shift_entropy_loss.detach().item()
            writer.add_scalar('loss/shift_Q', q_loss_value, self._shift_learning_steps)
            writer.add_scalar('loss/shift_policy', policy_loss_value, self._shift_learning_steps)
            writer.add_scalar('loss/shift_entropy', entropy_loss_value, self._shift_learning_steps)
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

        states, _actions, shift_actions, _rewards, _next_states, _dones = self.unpack_batch(batch)
        self._shift_learning_steps += 1
        action_indices = self.shift_action_indices(shift_actions)
        logits, probs = self._shift_policy_net(states)
        bc_loss = F.cross_entropy(logits, action_indices)
        update_params(self._shift_policy_optim, bc_loss)

        if self._shift_learning_steps % self._log_interval == 0:
            with torch.no_grad():
                predictions = torch.argmax(probs, dim=-1)
                accuracy = (predictions == action_indices).float().mean().item()
                action_rates = [
                    (action_indices == index).float().mean().item()
                    for index in range(self._shift_action_count)
                ]
                policy_rates = probs.mean(dim=0).detach().cpu().tolist()

            loss_value = bc_loss.detach().item()
            writer.add_scalar('loss/shift_bc', loss_value, self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_accuracy', accuracy, self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_noop_action_rate', action_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_up_action_rate', action_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_bc_down_action_rate', action_rates[2], self._shift_learning_steps)
            writer.add_scalar('stats/shift_noop_policy_prob', policy_rates[0], self._shift_learning_steps)
            writer.add_scalar('stats/shift_up_policy_prob', policy_rates[1], self._shift_learning_steps)
            writer.add_scalar('stats/shift_down_policy_prob', policy_rates[2], self._shift_learning_steps)
            return {
                "shift_bc_loss": loss_value,
                "shift_bc_accuracy": accuracy,
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

    def update_shift_policy_and_entropy(self, states, shift_actions):
        policy_loss, entropy, probs = self.calc_shift_policy_loss(states)
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

        return policy_loss, entropy_loss, entropy_value, action_rates, policy_rates

    def calc_shift_policy_loss(self, states):
        _, probs = self._shift_policy_net(states)
        log_probs = torch.log(probs.clamp_min(1e-8))
        qs1, qs2 = self._shift_online_q_net(states)
        qs = torch.min(qs1, qs2)
        policy_loss = (probs * (self._shift_alpha * log_probs - qs)).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        return policy_loss, entropy, probs.detach()

    def calc_shift_entropy_loss(self, entropy):
        assert not entropy.requires_grad
        return torch.mean(self._shift_log_alpha * (entropy - self._shift_target_entropy))

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
        if self._shift_enabled:
            self._shift_policy_net.save(os.path.join(save_dir, 'shift_net.pth'))
            self._shift_online_q_net.save(os.path.join(save_dir, 'shift_online_q_net.pth'))
            self._shift_target_q_net.save(os.path.join(save_dir, 'shift_target_q_net.pth'))

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
