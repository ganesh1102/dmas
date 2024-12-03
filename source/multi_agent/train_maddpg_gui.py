# train_maddpg.py

import numpy as np
import torch
from search_hider_env import SearchHiderEnv
from maddpg import MADDPGAgent
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def maddpg_train():
    """Train agents using the MADDPG algorithm in the search and hider environment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_episodes = 500
    max_steps = 500  # Increased to allow agents more steps to find hiders
    env = SearchHiderEnv(
        grid_size=10,
        num_agents=4,
        num_hiders=2,
        visibility_radius=1.0,
        max_action=1.0,
        max_steps=max_steps,
        central_square_size=4.0  # Define the size of the central square for hider initialization
    )
    num_agents = env.num_agents
    num_hiders = env.num_hiders
    obs_dim = env.observation_space.shape[0]  # Observation dimension per agent
    action_dim = env.action_space.shape[0]    # Action dimension per agent (e.g., 2 for x and y movement)
    total_obs_dim = obs_dim * num_agents
    total_action_dim = action_dim * num_agents

    # Initialize agents with exploration noise parameters
    agents = []
    initial_noise = 1.0
    final_noise = 0.1  # Minimum noise level
    for i in range(num_agents):
        agent = MADDPGAgent(
            index=i,
            obs_dim=obs_dim,
            action_dim=action_dim,
            total_obs_dim=total_obs_dim,
            total_action_dim=total_action_dim,
            device=device,
            initial_noise=initial_noise,
            final_noise=final_noise
        )
        agents.append(agent)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=int(1e6))

    batch_size = 1024

    hiders_found_per_episode = []  # To store cumulative hiders found per step per episode
    belief_maps = []  # To store belief maps at the end of each episode

    if not os.path.exists('plots'):
        os.makedirs('plots')

    print('Starting training...')

    with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(max_episodes):
            obs_n = env.reset()
            cumulative_hiders_found = []  # To track cumulative hiders found at each step
            agent_rewards = [0.0 for _ in agents]  # Track individual agent rewards
            hiders_found_episode = 0
            termination_step = max_steps  # Default termination step

            # Initialize tracking for when each hider was found
            hider_found_steps = [None for _ in range(num_hiders)]  # Initialize as None

            # Initialize positions for plotting
            agent_positions = [[] for _ in range(num_agents)]  # List to store positions per agent
            hider_positions = [[] for _ in range(num_hiders)]  # List to store positions per hider

            for step in range(max_steps):
                actions = []
                for i, agent in enumerate(agents):
                    action = agent.act(obs_n[i], explore=True)
                    actions.append(action)
                next_obs_n, reward_n, dones, info_n = env.step(actions)

                # Record agent and hider positions
                for i in range(num_agents):
                    agent_positions[i].append(env.agent_positions[i].copy())
                for j in range(num_hiders):
                    if not env.hiders_found[j]:
                        hider_positions[j].append(env.hider_positions[j].copy())
                    else:
                        # If hider is found, append None to indicate removal
                        hider_positions[j].append(None)
                        # Record the step when the hider was found
                        if hider_found_steps[j] is None:
                            hider_found_steps[j] = step + 1  # +1 for 1-based indexing

                # Concatenate observations and actions for replay buffer
                state = np.concatenate(obs_n)
                action_vec = np.concatenate(actions)
                next_state = np.concatenate(next_obs_n)

                # Store experience in replay buffer with per-agent rewards and dones
                replay_buffer.push(state, action_vec, np.array(reward_n), next_state, np.array(dones))

                # Update individual agent rewards
                agent_rewards = [ar + r for ar, r in zip(agent_rewards, reward_n)]

                # Record cumulative hiders found
                hiders_found_episode = sum(env.hiders_found)
                cumulative_hiders_found.append(hiders_found_episode)

                obs_n = next_obs_n

                # Update agents
                if len(replay_buffer) > batch_size:
                    for agent in agents:
                        agent.update(replay_buffer, batch_size, agents)

                # Check if all hiders have been found
                if all(env.hiders_found):
                    termination_step = step + 1  # +1 to make it 1-based indexing
                    break  # End episode early if all hiders are found

                if any(dones):
                    termination_step = step + 1
                    break

            # After episode ends
            hiders_found_per_episode.append(cumulative_hiders_found)
            # Save the belief map at the end of the episode
            belief_maps.append(env.get_belief_map())

            # Plot agent and hider trajectories for the episode, indicating termination step
            plot_positions(agent_positions, hider_positions, episode, env.grid_size, hider_found_steps)

            # Update exploration noise for all agents based on performance
            for idx, agent in enumerate(agents):
                agent.update_noise_adaptive(agent_rewards[idx])

            # Update progress bar
            pbar.set_postfix({
                'Episode': episode + 1,
                'Hiders Found': hiders_found_episode,
                'Avg Reward': np.mean(agent_rewards)
            })
            pbar.update(1)

    # After training, plot mean number of hiders found over steps
    plot_mean_hiders_found(hiders_found_per_episode)

    # Plot the heatmap of final average uncertainty
    plot_average_uncertainty_heatmap(belief_maps, env.grid_size)

    print('Training complete and plots saved.')
    env.close()

def plot_positions(agent_positions, hider_positions, episode, grid_size, hider_found_steps):
    """
    Plot the trajectories of agents and hiders for a given episode, indicating when each hider was found.

    Args:
        agent_positions (list of lists): Each sublist contains positions (x, y) of an agent over time.
        hider_positions (list of lists): Each sublist contains positions (x, y) of a hider over time.
        episode (int): The episode number.
        grid_size (int): Size of the grid.
        hider_found_steps (list of int or None): Step at which each hider was found.
    """
    plt.figure(figsize=(8, 8))
    agent_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    hider_colors = ['black', 'gray', 'brown', 'olive']

    # Plot agent trajectories
    for idx, positions in enumerate(agent_positions):
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], color=agent_colors[idx % len(agent_colors)],
                 linewidth=1.0, label=f'Agent {idx}')
        # Mark start and end positions
        plt.scatter(positions[0, 0], positions[0, 1], color=agent_colors[idx % len(agent_colors)],
                    marker='o', s=50, label=f'Agent {idx} Start')
        plt.scatter(positions[-1, 0], positions[-1, 1], color=agent_colors[idx % len(agent_colors)],
                    marker='X', s=50, label=f'Agent {idx} End')

    # Plot hider trajectories
    for idx, positions in enumerate(hider_positions):
        # Remove None entries indicating hiders that were found
        valid_positions = [pos for pos in positions if pos is not None]
        if not valid_positions:
            continue  # Skip if hider was found in the first step
        positions = np.array(valid_positions)
        plt.plot(positions[:, 0], positions[:, 1], color=hider_colors[idx % len(hider_colors)],
                 linestyle='--', linewidth=1.0, label=f'Hider {idx}')
        # Mark start and end positions
        plt.scatter(positions[0, 0], positions[0, 1], color=hider_colors[idx % len(hider_colors)],
                    marker='s', s=50, label=f'Hider {idx} Start')
        plt.scatter(positions[-1, 0], positions[-1, 1], color=hider_colors[idx % len(hider_colors)],
                    marker='D', s=50, label=f'Hider {idx} End')

    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)

    # Construct title with hider found steps
    hider_info = []
    for idx, step in enumerate(hider_found_steps):
        if step is not None:
            hider_info.append(f'Hider {idx} found at Step {step}')
        else:
            hider_info.append(f'Hider {idx} not found')
    hider_info_str = '; '.join(hider_info)
    plt.title(f'Episode {episode + 1}: ' + hider_info_str)

    # To avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/positions_episode_{episode + 1}.png')
    plt.close()

def plot_mean_hiders_found(hiders_found_per_episode):
    """
    Plot the mean number of hiders found over steps across all episodes.

    Args:
        hiders_found_per_episode (list of lists): Each sublist contains cumulative hiders found per step in an episode.
    """

    num_episodes = len(hiders_found_per_episode)
    max_length = max(len(cumulative_hiders_found) for cumulative_hiders_found in hiders_found_per_episode)

    # Create an array to store hiders found per step per episode
    hiders_found_array = np.full((num_episodes, max_length), np.nan)

    for i, cumulative_hiders_found in enumerate(hiders_found_per_episode):
        length = len(cumulative_hiders_found)
        hiders_found_array[i, :length] = cumulative_hiders_found

    # Compute mean number of hiders found at each step over episodes
    mean_hiders_found = np.nanmean(hiders_found_array, axis=0)

    steps = np.arange(1, len(mean_hiders_found) + 1)

    plt.figure()
    plt.plot(steps, mean_hiders_found, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Mean Number of Hiders Found')
    plt.title('Mean Number of Hiders Found vs Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_hiders_found_vs_steps.png')
    plt.close()

def plot_average_uncertainty_heatmap(belief_maps, grid_size):
    """
    Plot a heatmap of average uncertainty across cells over all episodes,
    using a custom colormap where darker red indicates more confidence (lower uncertainty),
    and yellow indicates less confidence (higher uncertainty).

    Args:
        belief_maps (list of np.ndarray): List containing belief maps for each episode.
        grid_size (int): Size of the grid.
    """
    import matplotlib.colors as mcolors  # Import for custom colormap

    # Compute average belief over all episodes
    belief_maps_array = np.array(belief_maps)
    average_belief = np.mean(belief_maps_array, axis=0)

    # Compute per-cell entropy
    p = average_belief.flatten()
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    per_cell_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    average_uncertainty = per_cell_entropy.reshape(grid_size, grid_size)

    # Create custom colormap from dark red to yellow
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['darkred', 'red', 'orange', 'yellow'])

    # Plot the heatmap with the custom colormap
    plt.figure(figsize=(6, 5))
    plt.imshow(average_uncertainty, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Average Entropy')
    plt.title('Heatmap of Final Average Uncertainty')
    plt.xlabel('Y Position')
    plt.ylabel('X Position')
    plt.tight_layout()
    plt.savefig('plots/average_uncertainty_heatmap.png')
    plt.close()

if __name__ == '__main__':
    maddpg_train()
