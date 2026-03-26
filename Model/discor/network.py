import torch
from torch import nn
from torch.distributions import Normal


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
        self.load_state_dict(torch.load(path))


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

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256],
                 action_low=None, action_high=None):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=2*action_dim,
            hidden_units=hidden_units)

        if action_low is None:
            action_low = -torch.ones(action_dim, dtype=torch.float32)
        else:
            action_low = torch.as_tensor(action_low, dtype=torch.float32)

        if action_high is None:
            action_high = torch.ones(action_dim, dtype=torch.float32)
        else:
            action_high = torch.as_tensor(action_high, dtype=torch.float32)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer('_action_scale', action_scale)
        self.register_buffer('_action_bias', action_bias)

    def forward(self, states):
        assert states.dim() == 2
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        normals = Normal(means, stds)
        xs = normals.rsample()
        raw_actions = torch.tanh(xs)
        actions = raw_actions * self._action_scale + self._action_bias

        log_probs = normals.log_prob(xs) - torch.log(1 - raw_actions.pow(2) + 1e-6)
        log_probs = log_probs - torch.log(self._action_scale + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        deterministic_actions = torch.tanh(means) * self._action_scale + self._action_bias
        return actions, entropies, deterministic_actions
