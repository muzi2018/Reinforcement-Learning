import numpy as np
import random

# Define the environment
class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start  # Initialize agent's position

    def reset(self):
        """Reset the environment to the starting state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """Take an action and return the new state, reward, and if the episode is done."""
        # Define possible actions
        actions = ['up', 'down', 'left', 'right']
        if action not in actions:
            raise ValueError("Invalid action")

        # Calculate new state based on action
        new_state = list(self.state)
        if action == 'up':
            new_state[0] = max(0, new_state[0] - 1)  # Move up
        elif action == 'down':
            new_state[0] = min(self.grid_size[0] - 1, new_state[0] + 1)  # Move down
        elif action == 'left':
            new_state[1] = max(0, new_state[1] - 1)  # Move left
        elif action == 'right':
            new_state[1] = min(self.grid_size[1] - 1, new_state[1] + 1)  # Move right

        new_state = tuple(new_state)

        # Check if the new state is an obstacle
        if new_state in self.obstacles:
            new_state = self.state  # Stay in the same state

        # Define the reward structure
        if new_state == self.goal:
            reward = 1  # Reward for reaching the goal
        else:
            reward = -0.01  # Small penalty for each step

        self.state = new_state  # Update the current state
        done = (self.state == self.goal)  # Check if the episode is done

        return new_state, reward, done

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    """
    SARSA algorithm for finding the optimal policy in a GridWorld environment.
    
    Args:
        env: GridWorld environment object
        num_episodes: Number of episodes for training
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    
    Returns:
        Q: Q-table containing state-action values
    """
    # Initialize Q-table
    Q = np.zeros((env.grid_size[0], env.grid_size[1], 4))  # 4 possible actions (up, down, left, right)
    actions = ['up', 'down', 'left', 'right']

    def choose_action(state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)  # Explore: choose a random action
        else:
            return np.argmax(Q[state[0], state[1]])  # Exploit: choose the best action

    for episode in range(num_episodes):
        state = env.reset()
        action = choose_action(state)
        done = False

        while not done:
            next_state, reward, done = env.step(actions[action])
            next_action = choose_action(next_state)

            # Update Q-value using the SARSA update rule
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action

    return Q

def derive_policy(Q):
    """Derive the optimal policy from the Q-table."""
    policy = np.argmax(Q, axis=2)  # Choose the action with the highest Q-value
    return policy

if __name__ == "__main__":
    # Define the grid environment
    grid_size = (5, 5)
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (1, 2), (2, 1)]

    # Create the environment
    env = GridWorld(grid_size, start, goal, obstacles)

    # Hyperparameters
    num_episodes = 1000
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate

    # Train using SARSA
    Q = sarsa(env, num_episodes, alpha, gamma, epsilon)

    # Display the learned Q-values
    print("Learned Q-values:")
    print(Q)

    # Derive the optimal policy
    optimal_policy = derive_policy(Q)
    print("Optimal Policy (0: up, 1: down, 2: left, 3: right):")
    print(optimal_policy)

