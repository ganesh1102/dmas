import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SearchHiderEnv(gym.Env):
    """
    Custom Environment for the search mission.
    Two searchers try to find one hider in a grid world.
    The searchers do not know the position of the hider.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=5):
        super(SearchHiderEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(5)  # 0: stay, 1: up, 2: down, 3: left, 4: right
        
        # Observation space for each agent is their own position
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        # For two searchers and one hider
        self.num_agents = 2  # Number of searchers
        self.positions = None  # Positions of agents and hider
        
    def reset(self):
        # Initialize positions randomly
        self.positions = {}
        self.positions['searcher_0'] = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        self.positions['searcher_1'] = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        self.positions['hider'] = np.array([np.random.randint(self.grid_size), np.random.randint(self.grid_size)])
        
        # Observations are positions of searchers (they don't know the hider's position)
        obs_0 = self.positions['searcher_0']
        obs_1 = self.positions['searcher_1']
        
        return [obs_0, obs_1]
    
    def step(self, actions):
        rewards = [0.0, 0.0]
        dones = [False, False]
        infos = [{}, {}]

        hider_found = False 
        for i in range(self.num_agents):
            agent_key = f'searcher_{i}'
            action = actions[i]
            self.positions[agent_key] = self.move(self.positions[agent_key], action)

  
            if np.array_equal(self.positions[agent_key], self.positions['hider']):
                rewards[i] = 10.0  # Reward for finding the hider
                dones[i] = True 
                infos[i]['found_hider'] = True
                hider_found = True  
            else:
                rewards[i] = -0.1  # Small negative reward to encourage efficiency

            infos[i]['searcher_position'] = self.positions[agent_key]
            infos[i]['hider_position'] = self.positions['hider']
            
        # If hider is found by any agent, the episode ends
        done = hider_found

        obs_0 = self.positions['searcher_0']
        obs_1 = self.positions['searcher_1']
        
        return [obs_0, obs_1], rewards, dones, done, infos
    
    def move(self, position, action):
        new_position = position.copy()
        if action == 1:  # up
            new_position[1] += 1
        elif action == 2:  # down
            new_position[1] -= 1
        elif action == 3:  # left
            new_position[0] -= 1
        elif action == 4:  # right
            new_position[0] += 1
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        return new_position
    
    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        s0_pos = self.positions['searcher_0']
        s1_pos = self.positions['searcher_1']
        h_pos = self.positions['hider']
        grid[s0_pos[1], s0_pos[0]] = 'S0'
        grid[s1_pos[1], s1_pos[0]] = 'S1'
        grid[h_pos[1], h_pos[0]] = 'H'
        print('\n'.join([' '.join(row) for row in grid]))
        print()
        
    def close(self):
        pass
