import torch
import torch.nn as nn


class QuantileQNetwork(nn.Module):
    """Feed-forward quantile network for QR-DQN."""

    def __init__(self, obs_dim: int, num_actions: int, num_quantiles: int = 51, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile values with shape [batch, num_actions, num_quantiles]."""
        out = self.net(x)
        return out.view(-1, self.num_actions, self.num_quantiles)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values by averaging quantiles over the atom dimension."""
        return self.forward(x).mean(dim=2)
