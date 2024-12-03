# search_hider_env.py

import numpy as np
import gym
from gym import spaces

class SearchHiderEnv(gym.Env):
    def __init__(self, grid_size, num_agents, num_hiders, visibility_radius, max_action, max_steps, central_square_size=4.0):
        super(SearchHiderEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_hiders = num_hiders
        self.visibility_radius = visibility_radius
        self.max_action = max_action
        self.max_steps = max_steps
        self.central_square_size = central_square_size  # Size of the central square for hider initialization

        # Observation space: relative positions of hiders + agent's own position
        # For each hider: 2 values (x, y). If hider is found, relative position is [0,0]
        # Plus agent's own position: 2 values
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * num_hiders + 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self):
        # Initialize agents at the four corners of the grid
        self.agent_positions = self.initialize_agents()
        # Initialize hiders within the central square
        self.hider_positions = [self.random_position_within_central_square() for _ in range(self.num_hiders)]
        self.hiders_found = [False] * self.num_hiders
        self.steps = 0
        # Initialize belief map with uniform probability
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float64) / (self.grid_size ** 2)
        return self.get_observations()

    def initialize_agents(self):
        """
        Initialize agents at the corners of the grid.
        Assumes grid is square and has at least as many agents as corners.
        """
        positions = []
        corners = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, self.grid_size - 1], dtype=np.float64),
            np.array([self.grid_size - 1, 0.0], dtype=np.float64),
            np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float64)
        ]
        for i in range(min(self.num_agents, len(corners))):
            positions.append(corners[i].copy())
        while len(positions) < self.num_agents:
            positions.append(self.random_position())
        return positions

    def random_position(self):
        return np.random.uniform(0, self.grid_size, size=2).astype(np.float64)

    def random_position_within_central_square(self):
        """
        Initialize hiders uniformly within a central square around the grid center.

        Returns:
            np.ndarray: Position of the hider as a 2D coordinate.
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
                    rel_pos = np.array([0.0, 0.0], dtype=np.float64)  # Hider is no longer considered
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

        # Update hider positions based on spiral movement
        self.update_hider_positions()

        # Check for hiders found
        for i, agent_pos in enumerate(self.agent_positions):
            reward = 0.0
            for j, (hider_pos, found) in enumerate(zip(self.hider_positions, self.hiders_found)):
                if not found:
                    distance = np.linalg.norm(agent_pos - hider_pos)
                    if distance <= self.visibility_radius:
                        self.hiders_found[j] = True
                        reward += 10.0  # Reward for finding a hider
            rewards.append(reward)
            dones.append(False)
            info[i]['agent_position'] = self.agent_positions[i].copy()

        # Update belief map based on agent observations
        self.update_belief_map()

        # Small penalty to encourage active searching
        rewards = [r - 0.1 for r in rewards]

        # Episode ends if all hiders are found or max steps reached
        done = all(self.hiders_found) or self.steps >= self.max_steps
        dones = [done for _ in range(self.num_agents)]

        next_obs = self.get_observations()
        return next_obs, rewards, dones, info

    def update_hider_positions(self):
        """
        Update hider positions based on spiral movement.
        Each hider spirals outward from the center and stops moving once found.
        Hiders stay closer to the center by limiting their spiral movement.
        """
        center = np.array([self.grid_size / 2.0, self.grid_size / 2.0], dtype=np.float64)
        max_radius = self.grid_size / 2.0 - 1.0  # Ensure hiders stay within the grid
        for idx, (pos, found) in enumerate(zip(self.hider_positions, self.hiders_found)):
            if not found:
                # Calculate angle and radius for spiral
                angle = np.arctan2(pos[1] - center[1], pos[0] - center[0])
                radius = np.linalg.norm(pos - center)
                # Increment angle to create spiral effect
                angle += 0.1  # Adjust for spiral tightness
                # Increment radius to move outward, but cap it to max_radius
                radius = min(radius + 0.05, max_radius)
                # Update position
                new_x = center[0] + radius * np.cos(angle)
                new_y = center[1] + radius * np.sin(angle)
                new_pos = np.array([new_x, new_y], dtype=np.float64)
                # Ensure hider stays within grid boundaries
                new_pos = np.clip(new_pos, 0.0, self.grid_size - 1.0)
                self.hider_positions[idx] = new_pos

    def update_belief_map(self):
        """
        Update the belief map based on agents' observations.
        Agents reduce uncertainty in cells within their visibility radius.
        """
        for agent_pos in self.agent_positions:
            x_min = max(int(agent_pos[0] - self.visibility_radius), 0)
            x_max = min(int(agent_pos[0] + self.visibility_radius) + 1, self.grid_size)
            y_min = max(int(agent_pos[1] - self.visibility_radius), 0)
            y_max = min(int(agent_pos[1] + self.visibility_radius) + 1, self.grid_size)
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    # Compute distance from agent to cell center
                    cell_center = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                    distance = np.linalg.norm(agent_pos - cell_center)
                    if distance <= self.visibility_radius:
                        # Reduce belief in this cell due to observation
                        self.belief_map[x, y] *= 0.5 

        # Normalize the belief map
        total_belief = np.sum(self.belief_map)
        if total_belief > 0:
            self.belief_map /= total_belief
        else:
            # If all beliefs are zero (unlikely), reset to uniform
            self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float64) / (self.grid_size ** 2)

    def get_belief_map(self):
        """
        Return the current belief map.
        """
        return self.belief_map.copy()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
