import torch
from torch import nn
from torch.distributions import Categorical, Normal


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    assert isinstance(input_dim, int) and isinstance(output_dim, int)
    assert isinstance(hidden_units, list) or isinstance(hidden_units, list)

    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        first_param = next(self.parameters(), None)
        map_location = first_param.device if first_param is not None else "cpu"
        try:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=map_location)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(
                "Model checkpoint is incompatible with the current network shape. "
                "This project now uses 3 continuous control actions and a separate "
                "discrete shift policy, so old checkpoints from the 5-action "
                "policy shape must not be resumed; start a fresh run instead."
            ) from exc


class StateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_units=hidden_units)

    def forward(self, x):
        return self.net(x)


class TwinnedStateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net1 = StateActionFunction(state_dim, action_dim, hidden_units)
        self.net2 = StateActionFunction(state_dim, action_dim, hidden_units)

    def forward(self, states, actions):
        assert states.dim() == 2 and actions.dim() == 2

        x = torch.cat([states, actions], dim=1)
        value1 = self.net1(x)
        value2 = self.net2(x)
        return value1, value2


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=2*action_dim,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2

        # Calculate means and stds of actions.
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        # Gaussian distributions.
        normals = Normal(means, stds)

        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)

        # Calculate entropies.
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)


class DiscreteStateActionFunction(BaseNetwork):
    def __init__(self, state_dim, action_count, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=action_count,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2
        return self.net(states)


class TwinnedDiscreteStateActionFunction(BaseNetwork):
    def __init__(self, state_dim, action_count, hidden_units=[256, 256]):
        super().__init__()

        self.net1 = DiscreteStateActionFunction(state_dim, action_count, hidden_units)
        self.net2 = DiscreteStateActionFunction(state_dim, action_count, hidden_units)

    def forward(self, states):
        assert states.dim() == 2

        value1 = self.net1(states)
        value2 = self.net2(states)
        return value1, value2


class DiscreteShiftPolicy(BaseNetwork):
    def __init__(self, state_dim, hidden_units=[256, 256], shift_dim=2):
        super().__init__()

        self.shift_dim = int(shift_dim)
        self.action_count = self.shift_dim + 1
        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=self.action_count,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2

        logits = self.net(states)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def action_vectors(self, action_indices):
        actions = torch.zeros(
            (action_indices.shape[0], self.shift_dim),
            dtype=torch.float,
            device=action_indices.device,
        )
        if self.shift_dim >= 1:
            actions[:, 0] = (action_indices == 1).float()
        if self.shift_dim >= 2:
            actions[:, 1] = (action_indices == 2).float()
        return actions

    def deterministic_indices(self, probs, threshold=0.5):
        if threshold is None:
            return torch.argmax(probs, dim=-1)

        shift_probs = probs[:, 1:]
        best_shift_probs, best_shift_offsets = torch.max(shift_probs, dim=-1)
        best_shift_indices = best_shift_offsets + 1
        return torch.where(
            best_shift_probs >= threshold,
            best_shift_indices,
            torch.zeros_like(best_shift_indices),
        )

    def sample(self, states, deterministic=False, threshold=0.5):
        logits, probs = self.forward(states)
        distribution = Categorical(probs=probs)
        sampled_indices = distribution.sample()
        deterministic_indices = self.deterministic_indices(probs, threshold=threshold)
        action_indices = deterministic_indices if deterministic else sampled_indices
        return action_indices, self.action_vectors(action_indices), probs, deterministic_indices
