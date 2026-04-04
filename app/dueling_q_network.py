import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        shared = self.shared_net(x)
        value = self.value_head(shared)
        advantage = self.advantage_head(shared)
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values