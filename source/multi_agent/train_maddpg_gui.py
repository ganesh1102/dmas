# train_maddpg_gui.py
import numpy as np
import torch
from search_hider_env import SearchHiderEnv
from maddpg import MADDPGAgent, ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_positions(positions_s0, positions_s1, hider_positions, episode):
    """Plot the positions of both agents and the hider."""
    plt.figure(figsize=(6, 6))
    if positions_s0:
        plt.plot(*zip(*positions_s0), label="Searcher 0", marker="o", color="blue", linestyle="--")
    if positions_s1:
        plt.plot(*zip(*positions_s1), label="Searcher 1", marker="o", color="green", linestyle="--")
    if hider_positions:
        plt.plot(*zip(*hider_positions), label="Hider", marker="x", color="red", linestyle="-")

    if positions_s0:
        plt.scatter(*positions_s0[0], color="blue", marker="o", s=100, label="S0 Start")
        plt.scatter(*positions_s0[-1], color="darkblue", marker="o", s=100, label="S0 End")
    if positions_s1:
        plt.scatter(*positions_s1[0], color="green", marker="o", s=100, label="S1 Start")
        plt.scatter(*positions_s1[-1], color="darkgreen", marker="o", s=100, label="S1 End")
    if hider_positions:
        plt.scatter(*hider_positions[0], color="red", marker="x", s=100, label="Hider Start")
        plt.scatter(*hider_positions[-1], color="darkred", marker="x", s=100, label="Hider End")

    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Agent and Hider Positions - Episode {episode}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'positions_episode_{episode}.png')
    plt.close()

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    env = SearchHiderEnv(grid_size=10)  # Changed grid_size to 10
    num_agents = env.num_agents
    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # 10 * 10 = 100
    action_dim = env.action_space.n
    total_obs_dim = obs_dim * num_agents
    total_action_dim = action_dim * num_agents
    agents = []
    for i in range(num_agents):
        agent = MADDPGAgent(
            index=i, 
            obs_dim=obs_dim, 
            action_dim=action_dim,
            total_obs_dim=total_obs_dim, 
            total_action_dim=total_action_dim,
            device=device
        )
        agents.append(agent)

    replay_buffer = ReplayBuffer(capacity=1e6)
    max_episodes = 200
    max_steps = 25
    batch_size = 1024

    episode_rewards_history = []
    distances_history = []
    distances_s0 = []
    distances_s1 = []
    hider_found_history = []

    print('Starting training...')

    with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(max_episodes):
            obs_n = env.reset()  # List of flattened observations per agent
            episode_rewards = np.zeros(num_agents)
            episode_distances_s0 = []
            episode_distances_s1 = []
            hider_found = False

            # Track positions for plotting
            positions_s0, positions_s1, hider_positions = [], [], []

            for step in range(max_steps):
                actions = []
                for i, agent in enumerate(agents):
                    action = agent.act(obs_n[i], explore=True)
                    actions.append(action)
                next_obs_n, reward_n, done, info_n = env.step(actions)
                replay_buffer.push(obs_n, actions, reward_n, next_obs_n, [done]*num_agents)
                episode_rewards += reward_n

                # Get positions for each searcher and the hider
                pos_s0 = info_n[0]['searcher_position']
                pos_s1 = info_n[1]['searcher_position']
                hider_pos = info_n[0]['hider_position']

                # Track positions for plotting
                positions_s0.append(pos_s0)
                positions_s1.append(pos_s1)
                hider_positions.append(hider_pos)

                # Calculate distance
                distance_s0 = np.linalg.norm(np.array(pos_s0) - np.array(hider_pos))
                distance_s1 = np.linalg.norm(np.array(pos_s1) - np.array(hider_pos))

                episode_distances_s0.append(distance_s0)
                episode_distances_s1.append(distance_s1)

                if info_n[0]['found_hider'] or info_n[1]['found_hider']:
                    hider_found = True

                obs_n = next_obs_n  # Update observations for next step
                if done:
                    break
                for agent in agents:
                    agent.update(replay_buffer, batch_size, agents)

            total_episode_reward = np.sum(episode_rewards)
            episode_rewards_history.append(total_episode_reward)

            avg_distance_s0 = np.mean(episode_distances_s0) if episode_distances_s0 else 0
            avg_distance_s1 = np.mean(episode_distances_s1) if episode_distances_s1 else 0
            avg_distance = (avg_distance_s0 + avg_distance_s1) / 2
            distances_history.append(avg_distance)
            distances_s0.append(avg_distance_s0)
            distances_s1.append(avg_distance_s1)

            hider_found_history.append(1 if hider_found else 0)

            N = 10
            average_reward = np.mean(episode_rewards_history[-N:]) if len(episode_rewards_history) >= N else np.mean(episode_rewards_history)

            pbar.set_postfix({
                'Total Reward': f'{total_episode_reward:.2f}',
                f'Avg Reward ({N})': f'{average_reward:.2f}',
                'Hider Found': hider_found
            })
            pbar.update(1)

            # Plot positions after each episode
            if episode % 10 == 0:
                plot_positions(positions_s0, positions_s1, hider_positions, episode)

    print('Training complete and plots saved.')
    env.close()

if __name__ == '__main__':
    main()
