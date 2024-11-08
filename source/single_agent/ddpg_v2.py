import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm  # For progress bar

# Check for MPS support
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

class HideAndSeekEnv(gym.Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.searcher_position = np.random.uniform(0, 1, size=(2,))
        self.hider_position = np.random.uniform(0, 1, size=(2,))
        self.done = False
        return np.concatenate([self.searcher_position, self.hider_position])

    def step(self, action):
        self.searcher_position = np.clip(self.searcher_position + action, 0, 1)
        distance = np.linalg.norm(self.searcher_position - self.hider_position)
        reward = -distance / np.sqrt(2)  # Normalized reward
        reward = np.clip(reward, -10, 0)  # Clipping reward
        self.done = distance < 0.05
        return np.concatenate([self.searcher_position, self.hider_position]), reward, self.done, {}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = []
        self.max_size = 100000
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.losses = []

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            state = state.clone().detach().unsqueeze(0).to(device)

        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train_agent(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, target_actions)
            target_q_values = rewards + self.gamma * (1 - dones) * target_q_values

        current_q_values = self.critic(states, actions)
        target_q_values = target_q_values.view(current_q_values.size())
        
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        self.losses.append(critic_loss.item())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_targets()

    def update_targets(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, experience):
        if len(self.replay_buffer) > self.max_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

def compute_shannon_entropy(action):
    action_magnitude = np.abs(action)
    total = np.sum(action_magnitude)
    if total == 0:
        return 0.0
    probs = action_magnitude / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def update_searcher_confidence(entropy, max_entropy=np.log(2)):
    confidence = 1 - (entropy / max_entropy)
    return confidence

def main():
    env = HideAndSeekEnv()
    state_dim = 4
    action_dim = 2
    max_action = 1

    agent = DDPGAgent(state_dim, action_dim, max_action)

    num_episodes = 1000
    max_steps_per_episode = 100

    episode_rewards = []
    episode_losses = []
    episode_entropies = []
    episode_confidences = []

    with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            total_entropy = 0
            total_confidence = 0
            steps = 0

            # To store positions for visualization after each episode
            searcher_positions = [state[:2]]
            hider_position = state[2:]  # Hider remains static

            while not done and steps < max_steps_per_episode:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                entropy = compute_shannon_entropy(action)
                confidence = update_searcher_confidence(entropy)

                agent.add_experience((state, action, reward, next_state, done))
                agent.train_agent()

                state = next_state
                total_reward += reward
                total_entropy += entropy
                total_confidence += confidence
                steps += 1

                searcher_positions.append(state[:2])  # Track searcher position

            episode_rewards.append(total_reward)

            # Only compute average loss if losses exist for this episode
            if agent.losses:
                episode_losses.append(np.mean(agent.losses[-steps:]))
            else:
                episode_losses.append(0)

            episode_entropies.append(total_entropy / steps)
            episode_confidences.append(total_confidence / steps)

            # Update the progress bar with the reward and average loss
            pbar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Avg Loss': f'{episode_losses[-1]:.4f}'
            })
            pbar.update(1)

            # Plot the agent's behavior after each episode
            if episode % 10 == 0:  # Plot every 10 episodes
                plt.figure(figsize=(6, 6))
                searcher_positions = np.array(searcher_positions)

                # Plot hider's static position
                plt.scatter(hider_position[0], hider_position[1], color='red', label='Hider Position', s=100)
                # Plot searcher's path
                plt.plot(searcher_positions[:, 0], searcher_positions[:, 1], '-o', color='blue', label='Searcher Path')
                plt.scatter(searcher_positions[0, 0], searcher_positions[0, 1], color='green', label='Start Position', s=100)
                plt.scatter(searcher_positions[-1, 0], searcher_positions[-1, 1], color='purple', label='End Position', s=100)

                plt.title(f'Agent Behavior - Episode {episode}')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.legend()
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(True)
                plt.savefig(f'behavior_episode_{episode}.png')
                plt.close()

    # Plotting metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(episode_rewards)
    axs[0, 0].set_title('Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')

    axs[0, 1].plot(episode_losses)
    axs[0, 1].set_title('Critic Loss')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Loss')

    axs[1, 0].plot(episode_entropies)
    axs[1, 0].set_title('Shannon Entropy')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Average Entropy')

    axs[1, 1].plot(episode_confidences)
    axs[1, 1].set_title('Searcher Confidence')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Average Confidence')

    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()

if __name__ == "__main__":
    main()
