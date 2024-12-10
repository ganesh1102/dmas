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
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    max_episodes = 500
    max_steps = 500
    env = SearchHiderEnv(
        grid_size=10,
        num_agents=4,
        num_hiders=2,
        visibility_radius=1.0,
        max_action=1.0,
        max_steps=max_steps,
        central_square_size=4.0
        # coalitions default: [[0,1],[2,3]]
    )
    num_agents = env.num_agents
    num_hiders = env.num_hiders
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    total_obs_dim = obs_dim * num_agents
    total_action_dim = action_dim * num_agents

    # Initialize agents
    agents = []
    initial_noise = 1.0
    final_noise = 0.1
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

    replay_buffer = ReplayBuffer(capacity=int(1e6))
    batch_size = 1024

    # Initialize tracking lists
    hiders_found_per_episode = []        # Cumulative hiders found at each step per episode
    rewards_per_episode = []             # Total reward (summed over agents) at each step per episode
    belief_maps = []                     # Belief maps at the end of each episode
    coalitions_found_per_episode = []    # Hiders found by each coalition at each step per episode
    entropy_per_episode = []             # Entropy of the belief map at each step per episode
    steps_to_find_all_hiders = []        # Steps taken to find all hiders per episode

    if not os.path.exists('plots'):
        os.makedirs('plots')

    print('Starting training...')

    with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(max_episodes):
            obs_n = env.reset()
            cumulative_hiders_found = []
            agent_rewards = [0.0 for _ in agents]
            hiders_found_episode = 0
            termination_step = max_steps

            # Track positions (optional)
            agent_positions = [[] for _ in range(num_agents)]
            hider_positions = [[] for _ in range(num_hiders)]
            hider_found_steps = [None for _ in range(num_hiders)]

            # Initialize coalition step counts
            coalition_step_counts = [[] for _ in range(len(env.coalitions))]

            # Initialize entropy list for this episode
            entropy_steps = []

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
                        hider_positions[j].append(None)
                        if hider_found_steps[j] is None:
                            hider_found_steps[j] = step + 1

                # Extract coalition performance info from info_n
                # Assuming all agents share the same coalition_hider_found
                coalition_hider_found = info_n[0]['coalition_hider_found']

                # Append to coalition arrays
                for c_idx, c_val in coalition_hider_found.items():
                    coalition_step_counts[c_idx].append(c_val)

                # Concatenate observations and actions for replay buffer
                state = np.concatenate(obs_n)
                action_vec = np.concatenate(actions)
                next_state = np.concatenate(next_obs_n)

                replay_buffer.push(state, action_vec, np.array(reward_n), next_state, np.array(dones))

                # Update individual agent rewards
                agent_rewards = [ar + r for ar, r in zip(agent_rewards, reward_n)]
                hiders_found_episode = sum(env.hiders_found)
                cumulative_hiders_found.append(hiders_found_episode)

                # Store total reward this step (sum across all agents)
                total_reward_this_step = np.sum(reward_n)
                if len(rewards_per_episode) < (episode + 1):
                    rewards_per_episode.append([])
                rewards_per_episode[episode].append(total_reward_this_step)

                # Calculate entropy of the belief map
                entropy = calculate_entropy(env.get_belief_map())
                entropy_steps.append(entropy)

                obs_n = next_obs_n

                # Update agents
                if len(replay_buffer) > batch_size:
                    for agent in agents:
                        agent.update(replay_buffer, batch_size, agents)

                # Check termination conditions
                if all(env.hiders_found):
                    termination_step = step + 1
                    break

                if any(dones):
                    termination_step = step + 1
                    break

            # After episode ends
            hiders_found_per_episode.append(cumulative_hiders_found)
            belief_maps.append(env.get_belief_map())
            coalitions_found_per_episode.append(coalition_step_counts)
            entropy_per_episode.append(entropy_steps)
            steps_to_find_all_hiders.append(termination_step if all(env.hiders_found) else max_steps)

            # Plot agent and hider trajectories
            plot_positions(agent_positions, hider_positions, episode, env.grid_size, hider_found_steps)

            # Update exploration noise based on performance
            for idx, agent in enumerate(agents):
                agent.update_noise_adaptive(agent_rewards[idx])

            # Update progress bar
            pbar.set_postfix({
                'Episode': episode + 1,
                'Hiders Found': hiders_found_episode,
                'Avg Reward': np.mean(agent_rewards)
            })
            pbar.update(1)

    # Plotting after training
    plot_mean_hiders_found(hiders_found_per_episode)
    plot_mean_rewards_per_step(rewards_per_episode)
    plot_coalition_performance(coalitions_found_per_episode)
    plot_total_hiders_found_over_steps(hiders_found_per_episode)
    plot_system_entropy_over_steps(entropy_per_episode)
    plot_mean_steps_to_find_all_hiders(steps_to_find_all_hiders)
    plot_average_uncertainty_heatmap(belief_maps, env.grid_size)

    print('Training complete and plots saved.')
    env.close()


def smooth_data(data, window_size=10):
    """
    Apply a simple moving average to smooth the data.
    data: 1D array of values
    window_size: number of points to average over
    """
    if len(data) < window_size:
        return data  # Not enough data to smooth
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return smoothed


def calculate_entropy(belief_map):
    """
    Calculate the entropy of the belief map.
    belief_map: 2D numpy array representing the belief probabilities of each cell.
    Returns:
        entropy (float): The entropy of the system.
    """
    p = belief_map.flatten()
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    entropy = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
    return entropy


def plot_positions(agent_positions, hider_positions, episode, grid_size, hider_found_steps):
    """
    Plot the trajectories of agents and hiders for a given episode, with smoothing applied.
    """
    plt.figure(figsize=(8, 8))
    agent_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    hider_colors = ['black', 'gray', 'brown', 'olive']
    window_size = 10  # Adjust smoothing window as desired

    for idx, positions in enumerate(agent_positions):
        positions = np.array(positions)
        if len(positions) < window_size:
            smoothed_x, smoothed_y = positions[:, 0], positions[:, 1]
        else:
            smoothed_x = smooth_data(positions[:, 0], window_size=window_size)
            smoothed_y = smooth_data(positions[:, 1], window_size=window_size)

        plt.plot(smoothed_x, smoothed_y, color=agent_colors[idx % len(agent_colors)],
                 linewidth=2.0, label=f'Agent {idx}')
        # Mark start and end positions
        plt.scatter(smoothed_x[0], smoothed_y[0], color=agent_colors[idx % len(agent_colors)],
                    marker='o', s=100, label=f'Agent {idx} Start')
        plt.scatter(smoothed_x[-1], smoothed_y[-1], color=agent_colors[idx % len(agent_colors)],
                    marker='X', s=100, label=f'Agent {idx} End')

    for idx, positions in enumerate(hider_positions):
        valid_positions = [pos for pos in positions if pos is not None]
        if not valid_positions:
            continue
        positions = np.array(valid_positions)
        if len(positions) < window_size:
            smoothed_x, smoothed_y = positions[:, 0], positions[:, 1]
        else:
            smoothed_x = smooth_data(positions[:, 0], window_size=window_size)
            smoothed_y = smooth_data(positions[:, 1], window_size=window_size)

        plt.plot(smoothed_x, smoothed_y, color=hider_colors[idx % len(hider_colors)],
                 linestyle='--', linewidth=2.0, label=f'Hider {idx}')
        # Mark start and end positions
        plt.scatter(smoothed_x[0], smoothed_y[0], color=hider_colors[idx % len(hider_colors)],
                    marker='s', s=100, label=f'Hider {idx} Start')
        plt.scatter(smoothed_x[-1], smoothed_y[-1], color=hider_colors[idx % len(hider_colors)],
                    marker='D', s=100, label=f'Hider {idx} End')

    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)

    hider_info = []
    for idx, step in enumerate(hider_found_steps):
        if step is not None:
            hider_info.append(f'Hider {idx} found at Step {step}')
        else:
            hider_info.append(f'Hider {idx} not found')
    hider_info_str = '; '.join(hider_info)
    plt.title(f'Episode {episode + 1}: ' + hider_info_str)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/positions_episode_{episode + 1}_smoothed.png')
    plt.close()


def plot_mean_hiders_found(hiders_found_per_episode):
    """
    Plot the average number of hiders found at each step across all episodes, with smoothing.
    """
    num_episodes = len(hiders_found_per_episode)
    max_length = max(len(cumulative_hiders_found) for cumulative_hiders_found in hiders_found_per_episode)
    hiders_found_array = np.full((num_episodes, max_length), np.nan)
    for i, cumulative_hiders_found in enumerate(hiders_found_per_episode):
        length = len(cumulative_hiders_found)
        hiders_found_array[i, :length] = cumulative_hiders_found

    # Convert cumulative counts to per-step increments
    hiders_found_per_step_array = np.diff(hiders_found_array, axis=1, prepend=0)
    mean_hiders_found_at_step = np.nanmean(hiders_found_per_step_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_hiders_found_at_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_hiders_found_at_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='blue')
    plt.xlabel('Step')
    plt.ylabel('Mean Hiders Found at that Step')
    plt.title('Mean Hiders Found at Each Step Across Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_hiders_found_at_each_step_smoothed.png')
    plt.close()


def plot_mean_rewards_per_step(rewards_per_episode):
    """
    Plot the mean total reward at each step across all episodes, with smoothing.
    """
    num_episodes = len(rewards_per_episode)
    max_length = max(len(rews) for rews in rewards_per_episode)
    rewards_array = np.full((num_episodes, max_length), np.nan)
    for i, rews in enumerate(rewards_per_episode):
        length = len(rews)
        rewards_array[i, :length] = rews

    mean_rewards_per_step = np.nanmean(rewards_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_rewards_per_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_rewards_per_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='green')
    plt.xlabel('Step')
    plt.ylabel('Mean Total Reward')
    plt.title('Mean Total Reward at Each Step Across Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_rewards_per_step_smoothed.png')
    plt.close()


def plot_mean_steps_to_find_all_hiders(steps_to_find_all_hiders):
    """
    Plot the mean number of steps taken to find all hiders over episodes, with smoothing.
    """
    num_episodes = len(steps_to_find_all_hiders)
    mean_steps = np.cumsum(steps_to_find_all_hiders) / np.arange(1, num_episodes + 1)

    # Smooth the data
    smoothed = smooth_data(mean_steps, window_size=10)
    smoothed_steps = np.arange(10, len(mean_steps) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='teal')
    plt.xlabel('Episode')
    plt.ylabel('Mean Steps to Find All Hiders')
    plt.title('Mean Number of Steps to Find All Hiders Over Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_steps_to_find_all_hiders_over_episodes_smoothed.png')
    plt.close()


def plot_coalition_performance(coalitions_found_per_episode):
    """
    Plot the mean number of hiders found by each coalition at each step across all episodes, with smoothing.
    """
    num_episodes = len(coalitions_found_per_episode)
    if num_episodes == 0:
        return

    num_coalitions = len(coalitions_found_per_episode[0])
    max_length = 0
    for ep_data in coalitions_found_per_episode:
        coalition_lengths = [len(c_data) for c_data in ep_data]
        if len(coalition_lengths) > 0:
            max_length = max(max_length, max(coalition_lengths))

    # Initialize arrays for each coalition
    coalition_arrays = [np.full((num_episodes, max_length), np.nan) for _ in range(num_coalitions)]

    for i, ep_data in enumerate(coalitions_found_per_episode):
        for c_idx, c_data in enumerate(ep_data):
            length = len(c_data)
            coalition_arrays[c_idx][i, :length] = c_data

    # Compute mean hiders found per step for each coalition
    mean_coalition_per_step = [np.nanmean(c_arr, axis=0) for c_arr in coalition_arrays]

    # Plotting
    plt.figure()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    for c_idx, mean_data in enumerate(mean_coalition_per_step):
        if len(mean_data) < 10:
            smoothed = mean_data
            smoothed_steps = np.arange(1, len(mean_data) + 1)
        else:
            smoothed = smooth_data(mean_data, window_size=10)
            smoothed_steps = np.arange(10, len(mean_data) + 1)
        plt.plot(smoothed_steps, smoothed, linewidth=2.0, color=colors[c_idx % len(colors)],
                 label=f'Coalition {c_idx}')

    plt.xlabel('Step')
    plt.ylabel('Mean Hiders Found by Coalition at that Step')
    plt.title('Mean Hiders Found by Each Coalition at Each Step (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/coalition_mean_hiders_found_per_step_smoothed.png')
    plt.close()


def plot_system_entropy_over_steps(entropy_per_episode):
    """
    Plot the overall entropy of the system at each step across all episodes.
    Entropy measures the uncertainty in the belief map.
    """
    num_episodes = len(entropy_per_episode)
    if num_episodes == 0:
        return
    max_length = max(len(entropy_steps) for entropy_steps in entropy_per_episode)
    entropy_array = np.full((num_episodes, max_length), np.nan)
    for i, entropy_steps in enumerate(entropy_per_episode):
        length = len(entropy_steps)
        entropy_array[i, :length] = entropy_steps

    # Compute mean entropy at each step
    mean_entropy_at_step = np.nanmean(entropy_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_entropy_at_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_entropy_at_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='brown')
    plt.xlabel('Step')
    plt.ylabel('Mean System Entropy')
    plt.title('Mean System Entropy Over Steps (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_system_entropy_over_steps_smoothed.png')
    plt.close()


def plot_mean_steps_to_find_all_hiders(steps_to_find_all_hiders):
    """
    Plot the mean number of steps taken to find all hiders over episodes, with smoothing.
    """
    num_episodes = len(steps_to_find_all_hiders)
    mean_steps = np.cumsum(steps_to_find_all_hiders) / np.arange(1, num_episodes + 1)

    # Smooth the data
    smoothed = smooth_data(mean_steps, window_size=10)
    smoothed_steps = np.arange(10, len(mean_steps) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='teal')
    plt.xlabel('Episode')
    plt.ylabel('Mean Steps to Find All Hiders')
    plt.title('Mean Number of Steps to Find All Hiders Over Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_steps_to_find_all_hiders_over_episodes_smoothed.png')
    plt.close()


def plot_average_uncertainty_heatmap(belief_maps, grid_size):
    """
    Plot a heatmap of average uncertainty across cells over all episodes.
    """
    import matplotlib.colors as mcolors
    belief_maps_array = np.array(belief_maps)
    average_belief = np.mean(belief_maps_array, axis=0)

    # Compute per-cell entropy
    p = average_belief.flatten()
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    per_cell_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    average_uncertainty = per_cell_entropy.reshape(grid_size, grid_size)

    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['darkred', 'red', 'orange', 'yellow'])

    plt.figure(figsize=(6, 5))
    plt.imshow(average_uncertainty, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Average Entropy')
    plt.title('Heatmap of Final Average Uncertainty')
    plt.xlabel('Y Position')
    plt.ylabel('X Position')
    plt.tight_layout()
    plt.savefig('plots/average_uncertainty_heatmap.png')
    plt.close()


def plot_total_hiders_found_over_steps(hiders_found_per_episode):
    """
    Plot the total number of hiders found over each step across all episodes.
    This plot shows the cumulative hiders found up to each step.
    """
    num_episodes = len(hiders_found_per_episode)
    max_length = max(len(cumulative_hiders_found) for cumulative_hiders_found in hiders_found_per_episode)
    hiders_found_array = np.full((num_episodes, max_length), np.nan)
    for i, cumulative_hiders_found in enumerate(hiders_found_per_episode):
        length = len(cumulative_hiders_found)
        hiders_found_array[i, :length] = cumulative_hiders_found

    # Calculate mean cumulative hiders found at each step
    mean_cumulative_hiders_found = np.nanmean(hiders_found_array, axis=0)
    
    # Smooth the data
    smoothed = smooth_data(mean_cumulative_hiders_found, window_size=10)
    smoothed_steps = np.arange(10, len(mean_cumulative_hiders_found) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='magenta')
    plt.xlabel('Step')
    plt.ylabel('Mean Cumulative Hiders Found')
    plt.title('Mean Cumulative Hiders Found Over Steps (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_cumulative_hiders_found_over_steps_smoothed.png')
    plt.close()


def plot_system_entropy_over_steps(entropy_per_episode):
    """
    Plot the overall entropy of the system at each step across all episodes.
    Entropy measures the uncertainty in the belief map.
    """
    num_episodes = len(entropy_per_episode)
    if num_episodes == 0:
        return
    max_length = max(len(entropy_steps) for entropy_steps in entropy_per_episode)
    entropy_array = np.full((num_episodes, max_length), np.nan)
    for i, entropy_steps in enumerate(entropy_per_episode):
        length = len(entropy_steps)
        entropy_array[i, :length] = entropy_steps

    # Compute mean entropy at each step
    mean_entropy_at_step = np.nanmean(entropy_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_entropy_at_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_entropy_at_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='brown')
    plt.xlabel('Step')
    plt.ylabel('Mean System Entropy')
    plt.title('Mean System Entropy Over Steps (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_system_entropy_over_steps_smoothed.png')
    plt.close()


def plot_mean_steps_to_find_all_hiders(steps_to_find_all_hiders):
    """
    Plot the mean number of steps taken to find all hiders over episodes, with smoothing.
    """
    num_episodes = len(steps_to_find_all_hiders)
    mean_steps = np.cumsum(steps_to_find_all_hiders) / np.arange(1, num_episodes + 1)

    # Smooth the data
    smoothed = smooth_data(mean_steps, window_size=10)
    smoothed_steps = np.arange(10, len(mean_steps) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='teal')
    plt.xlabel('Episode')
    plt.ylabel('Mean Steps to Find All Hiders')
    plt.title('Mean Number of Steps to Find All Hiders Over Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_steps_to_find_all_hiders_over_episodes_smoothed.png')
    plt.close()


def plot_average_uncertainty_heatmap(belief_maps, grid_size):
    """
    Plot a heatmap of average uncertainty across cells over all episodes.
    """
    import matplotlib.colors as mcolors
    belief_maps_array = np.array(belief_maps)
    average_belief = np.mean(belief_maps_array, axis=0)

    # Compute per-cell entropy
    p = average_belief.flatten()
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    per_cell_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    average_uncertainty = per_cell_entropy.reshape(grid_size, grid_size)

    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['darkred', 'red', 'orange', 'yellow'])

    plt.figure(figsize=(6, 5))
    plt.imshow(average_uncertainty, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Average Entropy')
    plt.title('Heatmap of Final Average Uncertainty')
    plt.xlabel('Y Position')
    plt.ylabel('X Position')
    plt.tight_layout()
    plt.savefig('plots/average_uncertainty_heatmap.png')
    plt.close()


def plot_total_hiders_found_over_steps(hiders_found_per_episode):
    """
    Plot the total number of hiders found over each step across all episodes.
    This plot shows the cumulative hiders found up to each step.
    """
    num_episodes = len(hiders_found_per_episode)
    max_length = max(len(cumulative_hiders_found) for cumulative_hiders_found in hiders_found_per_episode)
    hiders_found_array = np.full((num_episodes, max_length), np.nan)
    for i, cumulative_hiders_found in enumerate(hiders_found_per_episode):
        length = len(cumulative_hiders_found)
        hiders_found_array[i, :length] = cumulative_hiders_found

    # Calculate mean cumulative hiders found at each step
    mean_cumulative_hiders_found = np.nanmean(hiders_found_array, axis=0)
    
    # Smooth the data
    smoothed = smooth_data(mean_cumulative_hiders_found, window_size=10)
    smoothed_steps = np.arange(10, len(mean_cumulative_hiders_found) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='magenta')
    plt.xlabel('Step')
    plt.ylabel('Mean Cumulative Hiders Found')
    plt.title('Mean Cumulative Hiders Found Over Steps (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_cumulative_hiders_found_over_steps_smoothed.png')
    plt.close()


def plot_mean_steps_to_find_all_hiders(steps_to_find_all_hiders):
    """
    Plot the mean number of steps taken to find all hiders over episodes, with smoothing.
    """
    num_episodes = len(steps_to_find_all_hiders)
    mean_steps = np.cumsum(steps_to_find_all_hiders) / np.arange(1, num_episodes + 1)

    # Smooth the data
    smoothed = smooth_data(mean_steps, window_size=10)
    smoothed_steps = np.arange(10, len(mean_steps) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='teal')
    plt.xlabel('Episode')
    plt.ylabel('Mean Steps to Find All Hiders')
    plt.title('Mean Number of Steps to Find All Hiders Over Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_steps_to_find_all_hiders_over_episodes_smoothed.png')
    plt.close()


def plot_mean_hiders_found(hiders_found_per_episode):
    """
    Plot the average number of hiders found at each step across all episodes, with smoothing.
    """
    num_episodes = len(hiders_found_per_episode)
    max_length = max(len(cumulative_hiders_found) for cumulative_hiders_found in hiders_found_per_episode)
    hiders_found_array = np.full((num_episodes, max_length), np.nan)
    for i, cumulative_hiders_found in enumerate(hiders_found_per_episode):
        length = len(cumulative_hiders_found)
        hiders_found_array[i, :length] = cumulative_hiders_found

    # Convert cumulative counts to per-step increments
    hiders_found_per_step_array = np.diff(hiders_found_array, axis=1, prepend=0)
    mean_hiders_found_at_step = np.nanmean(hiders_found_per_step_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_hiders_found_at_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_hiders_found_at_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='blue')
    plt.xlabel('Step')
    plt.ylabel('Mean Hiders Found at that Step')
    plt.title('Mean Hiders Found at Each Step Across Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_hiders_found_at_each_step_smoothed.png')
    plt.close()


def plot_mean_rewards_per_step(rewards_per_episode):
    """
    Plot the mean total reward at each step across all episodes, with smoothing.
    """
    num_episodes = len(rewards_per_episode)
    max_length = max(len(rews) for rews in rewards_per_episode)
    rewards_array = np.full((num_episodes, max_length), np.nan)
    for i, rews in enumerate(rewards_per_episode):
        length = len(rews)
        rewards_array[i, :length] = rews

    mean_rewards_per_step = np.nanmean(rewards_array, axis=0)

    # Smooth the data
    smoothed = smooth_data(mean_rewards_per_step, window_size=10)
    smoothed_steps = np.arange(10, len(mean_rewards_per_step) + 1)

    plt.figure()
    plt.plot(smoothed_steps, smoothed, linewidth=2.0, color='green')
    plt.xlabel('Step')
    plt.ylabel('Mean Total Reward')
    plt.title('Mean Total Reward at Each Step Across Episodes (Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_rewards_per_step_smoothed.png')
    plt.close()


if __name__ == '__main__':
    maddpg_train()
