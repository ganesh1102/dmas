# maddpg.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        # Adjusted layer sizes due to smaller input dimension
        self.fc1 = nn.Linear(obs_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, action_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = obs.view(-1, self.obs_dim)
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        logits = self.fc3(x)
        # Subtract max for numerical stability
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        action_probs = F.softmax(logits, dim=-1)
        # Add epsilon to prevent zeros and normalize
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        return action_probs

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim):
        super(Critic, self).__init__()
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs_all_agents, actions_all_agents):
        x = torch.cat([obs_all_agents, actions_all_agents], dim=1)
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        q_value = self.fc3(x)
        return q_value

class MADDPGAgent:
    def __init__(self, index, obs_dim, action_dim, total_obs_dim, total_action_dim, device, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.index = index
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(total_obs_dim, total_action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.target_critic = Critic(total_obs_dim, total_action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

        self.update_targets(tau=1.0)

    def act(self, obs, explore=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # Shape: (1, obs_dim)
        action_probs = self.actor(obs)
        if torch.isnan(action_probs).any():
            print(f"NaN in action_probs for agent {self.index}")
            print(f"obs: {obs}")
            action_probs = torch.nan_to_num(action_probs, nan=1e-8)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        action_probs_np = action_probs.cpu().detach().numpy().flatten()
        if np.isnan(action_probs_np).any() or np.sum(action_probs_np) == 0:
            print(f"Invalid action_probs_np for agent {self.index}")
            action_probs_np = np.ones(self.action_dim) / self.action_dim
        if explore:
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_dim)
            else:
                action = np.random.choice(self.action_dim, p=action_probs_np)
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            action = action_probs_np.argmax()
        return action

    def update(self, replay_buffer, batch_size, agents):
        if len(replay_buffer) < batch_size:
            return
        samples = replay_buffer.sample(batch_size)
        obs_n, actions_n, rewards_n, next_obs_n, dones_n = samples

        # Convert lists to numpy arrays
        obs_n = np.array(obs_n)  # Shape: (batch_size, num_agents, obs_dim)
        actions_n = np.array(actions_n)  # Shape: (batch_size, num_agents)
        rewards_n = np.array(rewards_n)  # Shape: (batch_size, num_agents)
        next_obs_n = np.array(next_obs_n)  # Shape: (batch_size, num_agents, obs_dim)
        dones_n = np.array(dones_n)  # Shape: (batch_size, num_agents)

        # Current observations and actions for this agent
        obs = torch.FloatTensor(obs_n[:, self.index]).to(self.device)  # Shape: (batch_size, obs_dim)
        actions = torch.LongTensor(actions_n[:, self.index]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)
        rewards = torch.FloatTensor(rewards_n[:, self.index]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)
        next_obs = torch.FloatTensor(next_obs_n[:, self.index]).to(self.device)  # Shape: (batch_size, obs_dim)
        dones = torch.FloatTensor(dones_n[:, self.index]).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)

        obs_all_agents = []
        actions_all_agents = []
        next_obs_all_agents = []
        next_actions_all_agents = []

        for agent in agents:
            agent_obs = torch.FloatTensor(obs_n[:, agent.index]).to(self.device)  # Shape: (batch_size, obs_dim)
            obs_all_agents.append(agent_obs)

            action_one_hot = torch.zeros(batch_size, agent.action_dim, device=self.device)
            action_indices = torch.LongTensor(actions_n[:, agent.index]).unsqueeze(1).to(self.device)
            action_one_hot.scatter_(1, action_indices, 1)
            actions_all_agents.append(action_one_hot)

            agent_next_obs = torch.FloatTensor(next_obs_n[:, agent.index]).to(self.device)  # Shape: (batch_size, obs_dim)
            next_obs_all_agents.append(agent_next_obs)

            with torch.no_grad():
                next_action_probs = agent.target_actor(agent_next_obs)
                next_action = next_action_probs.multinomial(num_samples=1)
                next_action_one_hot = torch.zeros(batch_size, agent.action_dim, device=self.device)
                next_action_one_hot.scatter_(1, next_action, 1)
                next_actions_all_agents.append(next_action_one_hot)

        obs_all_agents = torch.cat(obs_all_agents, dim=1)  # Shape: (batch_size, num_agents * obs_dim)
        actions_all_agents = torch.cat(actions_all_agents, dim=1)  # Shape: (batch_size, num_agents * action_dim)
        next_obs_all_agents = torch.cat(next_obs_all_agents, dim=1)  # Shape: (batch_size, num_agents * obs_dim)
        next_actions_all_agents = torch.cat(next_actions_all_agents, dim=1)  # Shape: (batch_size, num_agents * action_dim)

        with torch.no_grad():
            target_q_values = self.target_critic(next_obs_all_agents, next_actions_all_agents)  # Shape: (batch_size, 1)
            target_q = rewards + self.gamma * target_q_values * (1 - dones)  # Shape: (batch_size, 1)

        q_values = self.critic(obs_all_agents, actions_all_agents)  # Shape: (batch_size, 1)

        critic_loss = nn.MSELoss()(q_values, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Current action probabilities for this agent
        curr_action_probs = self.actor(obs)  # Shape: (batch_size, action_dim)

        # Convert to one-hot encoding
        gumbel_actions = F.gumbel_softmax(curr_action_probs.log(), hard=True)  # Shape: (batch_size, action_dim)

        # Replace the current agent's actions in the joint actions
        start = self.index * self.action_dim
        end = start + self.action_dim
        actions_all_agents[:, start:end] = gumbel_actions

        # Actor loss: maximize expected Q value
        actor_loss = -self.critic(obs_all_agents, actions_all_agents).mean()

        # Entropy bonus to encourage exploration
        entropy = - (curr_action_probs * torch.log(curr_action_probs + 1e-8)).sum(dim=-1).mean()
        actor_loss = actor_loss - 0.01 * entropy  # Adjust the entropy coefficient as needed

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.update_targets()

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update actor target network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update critic target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=int(capacity))

    def push(self, *args):
        self.buffer.append(tuple(map(np.array, args)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)
