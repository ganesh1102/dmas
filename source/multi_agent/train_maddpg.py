import numpy as np
import torch
from source.multi_agent.search_hider_env import SearchHiderEnv
from source.multi_agent.maddpg import MADDPGAgent, ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm


def moving_average(data, window_size):
    """Compute the moving average of the data using the specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    env = SearchHiderEnv(grid_size=5)
    num_agents = env.num_agents
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    total_obs_dim = obs_dim * num_agents
    total_action_dim = action_dim * num_agents
    agents = []
    for i in range(num_agents):
        agent = MADDPGAgent(
            index=i, obs_dim=obs_dim, action_dim=action_dim,
            total_obs_dim=total_obs_dim, total_action_dim=total_action_dim,
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

    with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(max_episodes):
            obs_n = env.reset()
            episode_rewards = np.zeros(num_agents)
            episode_distances_s0 = []
            episode_distances_s1 = []
            hider_found = False  
            for step in range(max_steps):
                actions = []
                for i, agent in enumerate(agents):
                    action = agent.act(obs_n[i], explore=True)
                    actions.append(action)
                next_obs_n, reward_n, done_n, done, info_n = env.step(actions)
                replay_buffer.push(obs_n, actions, reward_n, next_obs_n, done_n)
                episode_rewards += reward_n


                pos_s0 = info_n[0]['searcher_position']
                pos_s1 = info_n[1]['searcher_position']
                hider_pos = info_n[0]['hider_position']  

                distance_s0 = np.linalg.norm(pos_s0 - hider_pos)
                distance_s1 = np.linalg.norm(pos_s1 - hider_pos)

                episode_distances_s0.append(distance_s0)
                episode_distances_s1.append(distance_s1)

                
                if 'found_hider' in info_n[0] and info_n[0]['found_hider']:
                    hider_found = True
                if 'found_hider' in info_n[1] and info_n[1]['found_hider']:
                    hider_found = True

                obs_n = next_obs_n
                if done:
                    break
                for agent in agents:
                    agent.update(replay_buffer, batch_size, agents)
            total_episode_reward = np.sum(episode_rewards)  # Sum of rewards from all agents
            episode_rewards_history.append(total_episode_reward)

            avg_distance_s0 = np.mean(episode_distances_s0)
            avg_distance_s1 = np.mean(episode_distances_s1)
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
 
    print('Plotting rewards')

    plt.figure(figsize=(12, 6))
   
    if len(episode_rewards_history) >= 10:
        smoothed_rewards = moving_average(episode_rewards_history, window_size=10)
        plt.plot(range(9, len(episode_rewards_history)), smoothed_rewards, label='Smoothed Reward (Window=10)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('rewards.png')
    plt.close()

    
    
    print('Plotting distances')
    
    plt.figure(figsize=(12, 6))
    
    if len(distances_history) >= 10:
        smoothed_distances = moving_average(distances_history, window_size=10)
        plt.plot(range(9, len(distances_history)), smoothed_distances, label='Smoothed Distance (Window=10)')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.title('Average Distance to Hider per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_distance.png')
    plt.close()

    

    print('Plotting searcher distances')
    plt.figure(figsize=(12, 6))
    
    if len(distances_s0) >= 10:
        smoothed_s0 = moving_average(distances_s0, window_size=10)
        smoothed_s1 = moving_average(distances_s1, window_size=10)
        plt.plot(range(9, len(distances_s0)), smoothed_s0, label='Smoothed S0 Distance (Window=10)')
        plt.plot(range(9, len(distances_s1)), smoothed_s1, label='Smoothed S1 Distance (Window=10)')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.title('Distance of Each Searcher to Hider per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('searcher_distance.png')
    plt.close()

    cumulative_hider_found = np.cumsum(hider_found_history)
    print('Plotting hider found')
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_hider_found, label='Cumulative Hider Found')
    plt.xlabel('Episode')
    plt.ylabel('Number of Times Hider Found')
    plt.title('Cumulative Number of Times Hider Was Found')
    plt.legend()
    plt.grid(True)
    plt.savefig('hider_found.png')
    plt.close()

    
    print('Plotting hider found binary')
    plt.figure(figsize=(12, 6))
    if len(hider_found_history) >= 10:
        smoothed_hider_found = moving_average(hider_found_history, window_size=10)
        plt.plot(range(9, len(hider_found_history)), smoothed_hider_found, label='Smoothed Hider Found Rate (Window=10)')
    plt.xlabel('Episode')
    plt.ylabel('Hider Found (1=True, 0=False)')
    plt.title('Hider Found per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('hider_found_binary.png')
    plt.close()

    env.close()


if __name__ == '__main__':
    main()
