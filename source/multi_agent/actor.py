import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x = self.out(x)
        x = torch.clamp(x, min=-10, max=10)  # Prevent extreme negative values
        x = x - x.max(dim=-1, keepdim=True)[0]
        action_probs = nn.functional.softmax(x, dim=-1)
        action_probs = action_probs + 1e-8  # Add small epsilon to prevent zeros
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        if not torch.isfinite(action_probs).all():
            print("Actor forward pass has NaN or Inf in action_probs")
            print("action_probs:", action_probs)
        return action_probs
