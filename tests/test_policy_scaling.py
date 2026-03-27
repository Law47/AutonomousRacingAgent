import numpy as np
import torch

from Model.discor.network import GaussianPolicy


def test_gaussian_policy_respects_action_bounds():
    policy = GaussianPolicy(
        state_dim=5,
        action_dim=2,
        hidden_units=[8],
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    states = torch.zeros((4, 5), dtype=torch.float32)
    actions, _, deterministic_actions = policy(states)

    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.all(deterministic_actions >= -1.0)
    assert torch.all(deterministic_actions <= 1.0)
