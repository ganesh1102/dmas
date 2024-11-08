import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs_all_agents, actions_all_agents):
        x = torch.cat([obs_all_agents, actions_all_agents], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        q_value = self.out(x)
        return q_value
