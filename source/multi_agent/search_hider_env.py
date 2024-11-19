# search_hider_env.py
import numpy as np
from itertools import product
from gymnasium import Env, spaces

class HiderAgent():
    '''
    Class that defines the hider agent and its actions.
    '''

    def __init__(self, grid_size, move_prob=1/8):
        self.grid_size = grid_size
        self.move_prob = move_prob
        self.position = np.random.randint(grid_size, size=2)

    def move(self):
        '''
        Move the hider agent in one of the 8 possible directions.
        '''
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

        move = directions[np.random.choice(len(directions))]
        new_x = np.clip(self.position[0] + move[0], 0, self.grid_size - 1)
        new_y = np.clip(self.position[1] + move[1], 0, self.grid_size - 1)
        self.position = (new_x, new_y)

class SearchHiderEnv(Env):
    '''
    Class that defines the overall environment for the target search game.
    '''

    def __init__(self, grid_size=10, num_agents=2, visibility_radius=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.visibility_radius = visibility_radius
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.reset()

    def reset(self):
        self.agents_positions = [np.random.randint(self.grid_size, size=2) for _ in range(self.num_agents)]
        self.hider = HiderAgent(self.grid_size)
        self.belief_map = np.full((self.grid_size, self.grid_size), 1 / (self.grid_size ** 2))
        # Initialize observed grids for each agent
        self.agent_observed_grids = [np.zeros((self.grid_size, self.grid_size), dtype=bool) for _ in range(self.num_agents)]
        return self.flatten_obs()

    def _get_obs(self):
        '''
        Get the partial observation of the current state of the environment for each agent.

        Returns:
            obs_n: List of observations per agent. Each observation is a numpy array of shape [grid_size, grid_size]
        '''
        obs_n = []

        for idx, agent_pos in enumerate(self.agents_positions):
            # Initialize observations to -1 (unobserved)
            obs = np.full((self.grid_size, self.grid_size), -1, dtype=np.float32)

            x_min = max(0, agent_pos[0] - self.visibility_radius)
            x_max = min(self.grid_size - 1, agent_pos[0] + self.visibility_radius)
            y_min = max(0, agent_pos[1] - self.visibility_radius)
            y_max = min(self.grid_size - 1, agent_pos[1] + self.visibility_radius)

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if np.array_equal(self.hider.position, [x, y]):
                        obs[x, y] = 1  # Hider is present
                    else:
                        obs[x, y] = 0  # Observed but hider not present
                    # Mark the cell as observed
                    self.agent_observed_grids[idx][x, y] = True
            obs_n.append(obs)
        return obs_n

    def flatten_obs(self):
        obs_n = self._get_obs()
        return [obs.flatten() for obs in obs_n]

    def update_belief(self):
        '''
        Update belief map based on the observations at the current state.
        '''
        # For simplicity, we skip belief map updates in this example
        pass

    def step(self, actions):
        '''
        Take a step in the environment based on the actions taken by the agents.

        Args:
          actions: List of integers representing the actions taken by each agent

        Returns:
          obs_n: List of observations per agent
          rewards: List of rewards per agent
          done: Boolean indicating if the episode is over
          infos: List of dictionaries containing additional information per agent
        '''

        for i, action in enumerate(actions):
            directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),          (0, 1),
                          (1, -1),  (1, 0),  (1, 1)]
            move = directions[action]
            new_x = np.clip(self.agents_positions[i][0] + move[0], 0, self.grid_size - 1)
            new_y = np.clip(self.agents_positions[i][1] + move[1], 0, self.grid_size - 1)
            self.agents_positions[i] = (new_x, new_y)

        self.hider.move()
        self.update_belief()

        obs_n = self.flatten_obs()

        rewards = []
        dones = []
        infos = []

        done = False  # Episode ends when any agent finds the hider

        for i, agent_pos in enumerate(self.agents_positions):
            found_hider = np.array_equal(self.hider.position, agent_pos)
            # Exploration bonus: number of newly observed cells
            x_min = max(0, agent_pos[0] - self.visibility_radius)
            x_max = min(self.grid_size - 1, agent_pos[0] + self.visibility_radius)
            y_min = max(0, agent_pos[1] - self.visibility_radius)
            y_max = min(self.grid_size - 1, agent_pos[1] + self.visibility_radius)
            new_observations = 0
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if not self.agent_observed_grids[i][x, y]:
                        new_observations += 1
            total_possible_observations = (x_max - x_min + 1) * (y_max - y_min + 1)
            exploration_bonus = new_observations / total_possible_observations
            # Adjusted reward with increased exploration bonus
            reward = 1.0 if found_hider else -0.1 + exploration_bonus * 0.5  # Adjust the scaling factor as needed
            info = {
                'searcher_position': agent_pos,
                'hider_position': self.hider.position,
                'found_hider': found_hider
            }
            rewards.append(reward)
            dones.append(found_hider)
            infos.append(info)
            if found_hider:
                done = True

        return obs_n, rewards, done, infos
