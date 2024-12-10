# maddpg.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MADDPGAgent:
    def __init__(self, index, obs_dim, action_dim, total_obs_dim, total_action_dim, device,
                 initial_noise=1.0, final_noise=0.1):
        self.index = index
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        self.device = device

        # Initialize networks
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(total_obs_dim, total_action_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Noise parameters
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.noise = initial_noise

    def act(self, obs, explore=True):
        """
        Select an action for the agent given the observation.
        If `explore` is True, adds Gaussian noise for exploration.
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy().flatten()
        self.actor.train()
        if explore:
            # Add exploration noise
            action += np.random.normal(0, self.noise, size=self.action_dim)
            action = np.clip(action, -1.0, 1.0)
        return action

    def update_noise_adaptive(self, reward):
        """
        Adjust noise level based on the agent's reward.
        Increases noise for negative rewards and decreases for positive rewards.
        """
        if reward < 0:
            self.noise = min(self.noise + 0.05, self.initial_noise)
        else:
            self.noise = max(self.noise * 0.95, self.final_noise)

    def update(self, replay_buffer, batch_size, agents):
        """
        Update the actor and critic networks using sampled experiences from the replay buffer.
        """
        if len(replay_buffer) < batch_size:
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Target Q-value calculation
        target_actions = []
        for i, agent in enumerate(agents):
            target_action = agent.actor(next_states[:, i * agent.obs_dim : (i + 1) * agent.obs_dim])
            target_actions.append(target_action)
        target_actions = torch.cat(target_actions, dim=1).to(self.device)
        target_q = rewards[:, self.index].unsqueeze(1) + \
                   0.95 * self.critic(next_states, target_actions) * (1 - dones[:, self.index].unsqueeze(1))

        # Current Q-value
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        current_actions = []
        for i, agent in enumerate(agents):
            if i == self.index:
                current_action = agent.actor(states[:, i * agent.obs_dim : (i + 1) * agent.obs_dim])
            else:
                current_action = agent.actor(states[:, i * agent.obs_dim : (i + 1) * agent.obs_dim]).detach()
            current_actions.append(current_action)
        current_actions = torch.cat(current_actions, dim=1).to(self.device)
        actor_loss = -self.critic(states, current_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Actions in range [-1, 1]
        return x


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        return q
