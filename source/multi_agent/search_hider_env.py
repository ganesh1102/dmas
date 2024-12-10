# search_hider_env.py

import numpy as np
import gym
from gym import spaces

class SearchHiderEnv(gym.Env):
    def __init__(self, grid_size, num_agents, num_hiders, visibility_radius, max_action, max_steps,
                 central_square_size=4.0, coalitions=None):
        super(SearchHiderEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_hiders = num_hiders
        self.visibility_radius = visibility_radius
        self.max_action = max_action
        self.max_steps = max_steps
        self.central_square_size = central_square_size

        # If coalitions not provided, default to two coalitions with half the agents each
        if coalitions is None:
            half = self.num_agents // 2
            self.coalitions = [list(range(0, half)), list(range(half, self.num_agents))]
        else:
            self.coalitions = coalitions

        # Observation space: each agent sees 2 values per hider (relative x,y) plus its own position (2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * num_hiders + 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.agent_positions = self.initialize_agents()
        self.hider_positions = [self.random_position_within_central_square() for _ in range(self.num_hiders)]
        self.hiders_found = [False] * self.num_hiders
        self.steps = 0
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float64) / (self.grid_size ** 2)
        return self.get_observations()

    def initialize_agents(self):
        """
        Initialize agents at the four corners of the grid.
        If there are more agents than corners, assign additional agents to predefined positions or random positions.
        """
        corners = [
            np.array([0, 0], dtype=np.float64),
            np.array([0, self.grid_size - 1], dtype=np.float64),
            np.array([self.grid_size - 1, 0], dtype=np.float64),
            np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float64)
        ]

        positions = []
        num_corners = len(corners)
        for i in range(self.num_agents):
            if i < num_corners:
                pos = corners[i].copy()
            else:
                # Assign additional agents to the center if available, else random positions
                if self.num_agents <= 8:
                    # Define additional fixed positions to avoid randomness
                    additional_positions = [
                        np.array([self.grid_size // 2, 0], dtype=np.float64),
                        np.array([0, self.grid_size // 2], dtype=np.float64),
                        np.array([self.grid_size - 1, self.grid_size // 2], dtype=np.float64),
                        np.array([self.grid_size // 2, self.grid_size - 1], dtype=np.float64)
                    ]
                    pos = additional_positions[i - num_corners].copy()
                else:
                    pos = self.random_position()
            positions.append(pos)
        
        return positions

    def random_position(self):
        """
        Generate a random position within the grid.
        """
        return np.random.uniform(0, self.grid_size, size=2).astype(np.float64)

    def random_position_within_central_square(self):
        """
        Generate a random position within the central square area.
        """
        half_size = self.central_square_size / 2.0
        center = self.grid_size / 2.0
        x = np.random.uniform(center - half_size, center + half_size)
        y = np.random.uniform(center - half_size, center + half_size)
        return np.array([x, y], dtype=np.float64)

    def get_observations(self):
        observations = []
        for agent_pos in self.agent_positions:
            obs = []
            for hider_pos, found in zip(self.hider_positions, self.hiders_found):
                if not found:
                    rel_pos = hider_pos - agent_pos
                else:
                    rel_pos = np.array([0.0, 0.0], dtype=np.float64)
                obs.extend(rel_pos)
            obs.extend(agent_pos)
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def step(self, actions):
        self.steps += 1
        rewards = []
        dones = []
        info = [{} for _ in range(self.num_agents)]

        # Update agent positions
        for i, action in enumerate(actions):
            self.agent_positions[i] += action * self.max_action
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0.0, self.grid_size - 1.0)

        # Update hider positions with increased velocity
        self.update_hider_positions()

        # Check for hiders found and track coalition performance
        coalition_hider_found = {c_idx: 0 for c_idx in range(len(self.coalitions))}
        for i, agent_pos in enumerate(self.agent_positions):
            reward = 0.0
            for j, (hider_pos, found) in enumerate(zip(self.hider_positions, self.hiders_found)):
                if not found:
                    distance = np.linalg.norm(agent_pos - hider_pos)
                    if distance <= self.visibility_radius:
                        self.hiders_found[j] = True
                        reward += 10.0
                        # Determine which coalition agent i belongs to
                        for c_idx, coalition_agents in enumerate(self.coalitions):
                            if i in coalition_agents:
                                coalition_hider_found[c_idx] += 1
                                break
            rewards.append(reward)
            dones.append(False)
            info[i]['agent_position'] = self.agent_positions[i].copy()

        # Update belief map
        self.update_belief_map()

        # Small penalty to encourage active searching
        rewards = [r - 0.1 for r in rewards]

        # Episode ends if all hiders are found or max steps reached
        done = all(self.hiders_found) or self.steps >= self.max_steps
        dones = [done for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            info[i]['coalition_hider_found'] = coalition_hider_found
            info[i]['coalitions'] = self.coalitions

        next_obs = self.get_observations()
        return next_obs, rewards, dones, info

    def update_hider_positions(self):
        """
        Update hider positions to simulate movement with increased velocity.
        """
        center = np.array([self.grid_size / 2.0, self.grid_size / 2.0], dtype=np.float64)
        max_radius = self.grid_size / 2.0 - 1.0
        for idx, (pos, found) in enumerate(zip(self.hider_positions, self.hiders_found)):
            if not found:
                angle = np.arctan2(pos[1] - center[1], pos[0] - center[0])
                radius = np.linalg.norm(pos - center)
                # Increase angle and radius increments for higher velocity
                angle += 0.2  # Increased angle step
                radius = min(radius + 0.1, max_radius)  # Increased radius step
                new_x = center[0] + radius * np.cos(angle)
                new_y = center[1] + radius * np.sin(angle)
                new_pos = np.array([new_x, new_y], dtype=np.float64)
                new_pos = np.clip(new_pos, 0.0, self.grid_size - 1.0)
                self.hider_positions[idx] = new_pos

    def update_belief_map(self):
        """
        Update the belief map based on agents' visibility.
        """
        for agent_pos in self.agent_positions:
            x_min = max(int(agent_pos[0] - self.visibility_radius), 0)
            x_max = min(int(agent_pos[0] + self.visibility_radius) + 1, self.grid_size)
            y_min = max(int(agent_pos[1] - self.visibility_radius), 0)
            y_max = min(int(agent_pos[1] + self.visibility_radius) + 1, self.grid_size)
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    cell_center = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                    distance = np.linalg.norm(agent_pos - cell_center)
                    if distance <= self.visibility_radius:
                        self.belief_map[x, y] *= 0.5

        total_belief = np.sum(self.belief_map)
        if total_belief > 0:
            self.belief_map /= total_belief
        else:
            self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float64) / (self.grid_size ** 2)

    def get_belief_map(self):
        return self.belief_map.copy()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
