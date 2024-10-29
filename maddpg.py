import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=int(capacity))
        
    def push(self, *args):
        self.buffer.append(tuple(map(np.array, args)))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, index, obs_dim, action_dim, total_obs_dim, total_action_dim, device, lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01):
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

 
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

        self.obs_max_value = 4 

        self.update_targets(tau=1.0)
        
    def act(self, obs, explore=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device) / self.obs_max_value
        if not torch.isfinite(obs).all():
            print(f"Agent {self.index}: Invalid obs detected: {obs}")
        action_probs = self.actor(obs)
        if not torch.isfinite(action_probs).all():
            print(f"Agent {self.index}: Invalid action_probs detected: {action_probs}")
            print(f"Observations: {obs}")
        if explore:
            action_probs_np = action_probs.cpu().detach().numpy().flatten()
            if not np.isfinite(action_probs_np).all():
                print(f"Agent {self.index}: action_probs_np contains NaN or Inf: {action_probs_np}")
                action_probs_np = np.nan_to_num(action_probs_np, nan=1e-8)
            sum_probs = np.sum(action_probs_np)
            if sum_probs == 0:
                print(f"Agent {self.index}: Sum of action_probs_np is zero")
                action_probs_np = np.ones_like(action_probs_np) / self.action_dim
            else:
                action_probs_np = action_probs_np / sum_probs
            action = np.random.choice(self.action_dim, p=action_probs_np)
        else:
            action = action_probs.argmax(dim=-1).item()
        return action
        
    def update(self, replay_buffer, batch_size, agents):
        if len(replay_buffer) < batch_size:
            return
        
        samples = replay_buffer.sample(batch_size)
        obs_n, actions_n, rewards_n, next_obs_n, dones_n = samples
        
        
        obs = torch.FloatTensor(obs_n[:, self.index]).to(self.device) / self.obs_max_value
        actions = torch.LongTensor(actions_n[:, self.index]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards_n[:, self.index]).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs_n[:, self.index]).to(self.device) / self.obs_max_value
        dones = torch.FloatTensor(dones_n[:, self.index]).unsqueeze(1).to(self.device)
        

        obs_all_agents = []
        actions_all_agents = []
        next_obs_all_agents = []
        next_actions_all_agents = []
        
        for agent in agents:

            agent_obs = torch.FloatTensor(obs_n[:, agent.index]).to(self.device) / self.obs_max_value
            obs_all_agents.append(agent_obs)
            

            action_one_hot = torch.zeros(batch_size, agent.action_dim, device=self.device)
            action_indices = torch.LongTensor(actions_n[:, agent.index]).unsqueeze(1).to(self.device)
            action_one_hot.scatter_(1, action_indices, 1)
            actions_all_agents.append(action_one_hot)
            

            agent_next_obs = torch.FloatTensor(next_obs_n[:, agent.index]).to(self.device) / self.obs_max_value
            next_obs_all_agents.append(agent_next_obs)
            

            with torch.no_grad():
                next_action_probs = agent.target_actor(agent_next_obs)
                next_action = next_action_probs.multinomial(num_samples=1)
                next_action_one_hot = torch.zeros(batch_size, agent.action_dim, device=self.device)
                next_action_one_hot.scatter_(1, next_action, 1)
                next_actions_all_agents.append(next_action_one_hot)
        
        obs_all_agents = torch.cat(obs_all_agents, dim=1)
        actions_all_agents = torch.cat(actions_all_agents, dim=1)
        next_obs_all_agents = torch.cat(next_obs_all_agents, dim=1)
        next_actions_all_agents = torch.cat(next_actions_all_agents, dim=1)
        

        with torch.no_grad():
            target_q_values = self.target_critic(next_obs_all_agents, next_actions_all_agents)
            target_q = rewards + self.gamma * target_q_values * (1 - dones)
        

        q_values = self.critic(obs_all_agents, actions_all_agents)
        
        critic_loss = nn.MSELoss()(q_values, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        

        curr_action_probs = self.actor(obs)

        gumbel_actions = nn.functional.gumbel_softmax(curr_action_probs.log(), hard=True)
        

        start = self.index * self.action_dim
        end = start + self.action_dim
        actions_all_agents[:, start:end] = gumbel_actions

        actor_loss = -self.critic(obs_all_agents, actions_all_agents).mean()
        
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
