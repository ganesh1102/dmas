import numpy as np
import matplotlib.pyplot as plt

# Environment Parameters
grid_size = (5, 5)  # 5x5 grid world
action_space = ['up', 'down', 'left', 'right']
n_actions = len(action_space)
n_states = grid_size[0] * grid_size[1]

# State Initialization
state_space = np.arange(n_states).reshape(grid_size)
rewards = np.zeros((grid_size))  # Set rewards, e.g., rewards[4, 4] = 10
rewards[4, 4] = 10  # Example goal with high reward

# Action transitions (e.g., 0: up, 1: down, 2: left, 3: right)
def transition(state, action):
    x, y = state // grid_size[1], state % grid_size[1]
    if action == 0 and x > 0:  # up
        x -= 1
    elif action == 1 and x < grid_size[0] - 1:  # down
        x += 1
    elif action == 2 and y > 0:  # left
        y -= 1
    elif action == 3 and y < grid_size[1] - 1:  # right
        y += 1
    return x * grid_size[1] + y  # New state

# Q-Table Initialization
q_table = np.zeros((n_states, n_actions))

# RL Parameters
alpha = 0.1   # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1 # Exploration rate
n_episodes = 1000
max_steps = 50  # Max steps per episode

# Training Loop
for episode in range(n_episodes):
    state = np.random.choice(n_states)  # Start from a random state
    total_reward = 0

    for step in range(max_steps):
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(q_table[state])

        # Transition to next state
        next_state = transition(state, action)
        reward = rewards[next_state // grid_size[1], next_state % grid_size[1]]

        # Q-Table Update
        q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        # Visualize the step
        if episode % 100 == 0:
            plt.imshow(rewards, cmap='hot', interpolation='nearest')
            plt.scatter([state % grid_size[1]], [state // grid_size[1]], c='blue', marker='o')  # Agent position
            plt.pause(1)  # Pause to update plot
            plt.clf()

        state = next_state
        total_reward += reward

        # End episode if reached goal
        if reward == 10:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

plt.show()
