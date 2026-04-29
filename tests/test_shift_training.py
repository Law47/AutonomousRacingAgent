import os
import pickle
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISCOR_ROOT = os.path.join(ROOT, "algorithm", "discor")
ASSETTO_ROOT = os.path.join(ROOT, "assetto_corsa_gym")
if ASSETTO_ROOT not in sys.path:
    sys.path.insert(0, ASSETTO_ROOT)
if DISCOR_ROOT not in sys.path:
    sys.path.insert(0, DISCOR_ROOT)

from discor.agent import demo_lap_sample_counts, should_keep_demo_transition
from discor.algorithm.sac import SAC
from discor.network import DiscreteShiftPolicy
from discor.replay_buffer import ReplayBuffer
from train import maybe_load_demonstrations


def set_shift_bias(policy, bias):
    linear_layers = [module for module in policy.net if isinstance(module, nn.Linear)]
    output_layer = linear_layers[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(torch.tensor(bias, dtype=torch.float))


class ShiftTrainingTests(unittest.TestCase):
    def test_discrete_shift_policy_masks_invalid_deterministic_action(self):
        policy = DiscreteShiftPolicy(state_dim=2, hidden_units=[], shift_dim=2)
        set_shift_bias(policy, [0.0, -4.0, 8.0])

        states = torch.zeros((3, 2), dtype=torch.float)
        action_mask = torch.tensor([True, True, False])
        indices, actions, probs, deterministic = policy.sample(
            states,
            deterministic=True,
            threshold=0.5,
            action_mask=action_mask,
        )

        self.assertTrue(torch.equal(indices, deterministic))
        self.assertTrue(torch.equal(indices, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.allclose(actions, torch.zeros((3, 2))))
        self.assertLess(torch.max(probs[:, 2]).item(), 1e-6)

    def test_replay_buffer_preserves_shift_weights_through_pickle(self):
        buffer = ReplayBuffer(
            memory_size=4,
            state_shape=(3,),
            action_shape=(2,),
            gamma=0.99,
            nstep=1,
            shift_shape=(2,),
        )
        buffer.append(
            np.ones(3, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            1.0,
            np.ones(3, dtype=np.float32) * 2.0,
            False,
            shift_label=np.array([1.0, 0.0], dtype=np.float32),
            shift_weight=1.0,
            demo_weight=1.0,
        )
        buffer.append(
            np.ones(3, dtype=np.float32) * 3.0,
            np.ones(2, dtype=np.float32),
            0.5,
            np.ones(3, dtype=np.float32) * 4.0,
            True,
            shift_label=np.array([0.0, 1.0], dtype=np.float32),
            shift_weight=0.0,
            demo_weight=0.0,
        )

        loaded = pickle.loads(pickle.dumps(buffer))
        batch = loaded._sample_batch(np.array([0, 1]), 2, torch.device("cpu"))

        self.assertEqual(len(batch), 8)
        shift_labels = batch[2]
        shift_weights = batch[3]
        demo_weights = batch[4]
        self.assertTrue(torch.equal(shift_labels, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertTrue(torch.equal(shift_weights, torch.tensor([[1.0], [0.0]])))
        self.assertTrue(torch.equal(demo_weights, torch.tensor([[1.0], [0.0]])))

    def test_shift_bc_loss_ignores_zero_weight_manual_labels(self):
        sac = SAC(
            state_dim=4,
            action_dim=2,
            device=torch.device("cpu"),
            policy_hidden_units=[8],
            q_hidden_units=[8],
            shift_hidden_units=[8],
            shift_bc_loss_weight=0.5,
        )
        logits = torch.tensor(
            [[-10.0, 10.0, -10.0], [10.0, 0.0, 0.0]],
            requires_grad=True,
        )
        action_indices = torch.tensor([1, 2], dtype=torch.long)
        shift_weights = torch.tensor([[0.0], [1.0]], dtype=torch.float)

        loss, stats = sac.calc_shift_bc_loss(logits, action_indices, shift_weights)
        self.assertIsNotNone(loss)
        loss.backward()

        self.assertTrue(torch.allclose(logits.grad[0], torch.zeros(3)))
        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(stats["teacher_fraction"], 0.5)

        zero_loss, zero_stats = sac.calc_shift_bc_loss(
            logits.detach(),
            action_indices,
            torch.zeros((2, 1), dtype=torch.float),
        )
        self.assertIsNone(zero_loss)
        self.assertEqual(zero_stats, {})

    def test_sac_explore_can_execute_shifter_deterministically_with_mask(self):
        sac = SAC(
            state_dim=4,
            action_dim=2,
            device=torch.device("cpu"),
            policy_hidden_units=[8],
            q_hidden_units=[8],
            shift_hidden_units=[],
            shift_threshold=0.5,
        )
        set_shift_bias(sac._shift_policy_net, [0.0, -4.0, 8.0])

        _action, info = sac.explore(
            np.zeros(4, dtype=np.float32),
            shift_deterministic=True,
            shift_action_mask=np.array([True, True, False]),
        )

        self.assertTrue(np.array_equal(info["shift_action"], np.zeros(2, dtype=np.float32)))
        self.assertLess(float(info["shift_probs"][1]), 1e-6)

    def test_demo_actor_bc_loss_ignores_online_samples(self):
        sac = SAC(
            state_dim=4,
            action_dim=2,
            device=torch.device("cpu"),
            policy_hidden_units=[8],
            q_hidden_units=[8],
            shift_hidden_units=[8],
            demo_actor_loss_weight=0.2,
            demo_actor_bc_mode="bc",
        )
        states = torch.zeros((2, 4), dtype=torch.float)
        policy_actions = torch.zeros((2, 2), dtype=torch.float, requires_grad=True)
        demo_actions = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float)
        demo_weights = torch.tensor([[0.0], [1.0]], dtype=torch.float)

        loss, stats = sac.calc_demo_actor_bc_loss(
            states,
            demo_actions,
            demo_weights,
            policy_actions=policy_actions,
        )
        self.assertIsNotNone(loss)
        loss.backward()

        self.assertTrue(torch.allclose(policy_actions.grad[0], torch.zeros(2)))
        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(stats["sample_fraction"], 0.5)
        self.assertAlmostEqual(stats["effective_fraction"], 0.5)

    def test_demo_loader_runs_shift_only_pretrain_when_configured(self):
        class FakeAgent:
            def __init__(self):
                self.loaded_paths = []
                self.pretrain_calls = []
                self.shift_pretrain_calls = []

            def load_pre_train_data(self, path, env, log_steer_ratios=False, demo_filter_config=None):
                self.loaded_paths.append((path, log_steer_ratios))
                return 8

            def pre_train_epochs(self, num_epochs, num_samples=None):
                self.pretrain_calls.append((num_epochs, num_samples))

            def pre_train_shift_behavior_clone_epochs(self, num_epochs, num_samples=None):
                self.shift_pretrain_calls.append((num_epochs, num_samples))

        agent = FakeAgent()
        config = SimpleNamespace(
            Demonstrations=SimpleNamespace(
                enabled=True,
                data_path="demo_session",
                data_paths=[],
                pretrain_epochs=0,
                shift_pretrain_epochs=2,
                log_steer_ratios=True,
            )
        )

        result = maybe_load_demonstrations(agent, env=None, config=config)

        self.assertEqual(result["transitions"], 8)
        self.assertFalse(result["pretrained"])
        self.assertTrue(result["shift_pretrained"])
        self.assertEqual(agent.pretrain_calls, [])
        self.assertEqual(agent.shift_pretrain_calls, [(2, 8)])

    def test_demo_loader_runs_extra_shift_pretrain_with_joint_pretrain(self):
        class FakeAgent:
            def __init__(self):
                self.pretrain_calls = []
                self.shift_pretrain_calls = []

            def load_pre_train_data(self, path, env, log_steer_ratios=False, demo_filter_config=None):
                return 12

            def pre_train_epochs(self, num_epochs, num_samples=None):
                self.pretrain_calls.append((num_epochs, num_samples))

            def pre_train_shift_behavior_clone_epochs(self, num_epochs, num_samples=None):
                self.shift_pretrain_calls.append((num_epochs, num_samples))

        agent = FakeAgent()
        config = SimpleNamespace(
            Demonstrations=SimpleNamespace(
                enabled=True,
                data_path=None,
                data_paths=["session_a", "session_b"],
                pretrain_epochs=3,
                shift_pretrain_epochs=1,
                log_steer_ratios=False,
            )
        )

        result = maybe_load_demonstrations(agent, env=None, config=config)

        self.assertTrue(result["pretrained"])
        self.assertTrue(result["shift_pretrained"])
        self.assertEqual(result["transitions"], 24)
        self.assertEqual(agent.pretrain_calls, [(3, 24)])
        self.assertEqual(agent.shift_pretrain_calls, [(1, 24)])

    def test_demo_transition_filter_rejects_long_lap_and_progress_jump(self):
        trajectory = (
            [{"LapCount": 1, "LapDist": float(i), "speed": 20.0} for i in range(5)]
            + [{"LapCount": 2, "LapDist": float(i), "speed": 20.0} for i in range(5001)]
        )
        lap_counts = demo_lap_sample_counts(trajectory)
        config = {
            "enabled": True,
            "max_lap_samples": 4000,
            "max_abs_progress_delta_m": 20.0,
            "min_speed_ms": 1.0,
        }

        keep, reason = should_keep_demo_transition(
            trajectory[0],
            trajectory[1],
            lap_counts,
            config,
            track_length=6000.0,
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "kept")

        keep, reason = should_keep_demo_transition(
            trajectory[5],
            trajectory[6],
            lap_counts,
            config,
            track_length=6000.0,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "long_lap")

        keep, reason = should_keep_demo_transition(
            {"LapCount": 3, "LapDist": 100.0, "speed": 20.0},
            {"LapCount": 3, "LapDist": 70.0, "speed": 20.0},
            {3: 2},
            config,
            track_length=6000.0,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "progress_jump")


if __name__ == "__main__":
    unittest.main()
