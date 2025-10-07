import numpy as np

# Step 1: Define the GridWorld environment
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)  # Start at the top-left corner
        self.rewards = np.zeros((size, size))
        self.rewards[size - 1, size - 1] = 1  # Reward at the bottom-right corner

    def reset(self):
        self.state = (0, 0)  # Reset to the start
        return self.state

    def step(self, action):
        if action == 0:  # Up
            self.state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # Down
            self.state = (min(self.state[0] + 1, self.size - 1), self.state[1])
        elif action == 2:  # Left
            self.state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 3:  # Right
            self.state = (self.state[0], min(self.state[1] + 1, self.size - 1))

        reward = self.rewards[self.state]
        done = self.state == (self.size - 1, self.size - 1)  # Goal state
        return self.state, reward, done

# Step 2: Monte Carlo Learning
class MonteCarlo:
    def __init__(self, env, num_episodes=1000):
        self.env = env
        self.num_episodes = num_episodes
        self.state_values = np.zeros((env.size, env.size))
        self.returns = {}  # Store returns for each state
        for i in range(env.size):
            for j in range(env.size):
                self.returns[(i, j)] = []

    def learn(self):
        for _ in range(self.num_episodes):
            episode = []
            state = self.env.reset()
            done = False

            # Generate an episode
            while not done:
                action = np.random.choice(4)  # Random policy
                next_state, reward, done = self.env.step(action)
                episode.append((state, reward))
                state = next_state

            # Calculate returns and update state values
            G = 0
            for state, reward in reversed(episode):
                G = reward + G  # Discount factor γ = 1
                if state not in [x[0] for x in episode[:-1]]:
                    self.returns[state].append(G)
                    self.state_values[state] = np.mean(self.returns[state])

        return self.state_values

# Step 3: Temporal-Difference Learning
class TemporalDifference:
    def __init__(self, env, num_episodes=1000, alpha=0.1):
        self.env = env
        self.num_episodes = num_episodes
        self.state_values = np.zeros((env.size, env.size))
        self.alpha = alpha

    def learn(self):
        for _ in range(self.num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = np.random.choice(4)  # Random policy
                next_state, reward, done = self.env.step(action)

                # TD update
                self.state_values[state] += self.alpha * (
                    reward + self.state_values[next_state] - self.state_values[state]
                )
                state = next_state

        return self.state_values

# Step 4: Running the Algorithms and Comparing
# Create the environment
env = GridWorld(size=4)

# Monte Carlo Learning
mc = MonteCarlo(env, num_episodes=1000)
mc_values = mc.learn()
print("Monte Carlo State Values:")
print(mc_values)

# Temporal-Difference Learning
td = TemporalDifference(env, num_episodes=1000, alpha=0.1)
td_values = td.learn()
print("\nTemporal-Difference State Values:")
print(td_values)

# Step 5: Visualization (Optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Monte Carlo Results
plt.subplot(1, 2, 1)
plt.title("Monte Carlo Values")
plt.imshow(mc_values, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="State Value")
plt.xticks(range(env.size))
plt.yticks(range(env.size))

# Temporal Difference Results
plt.subplot(1, 2, 2)
plt.title("Temporal Difference Values")
plt.imshow(td_values, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="State Value")
plt.xticks(range(env.size))
plt.yticks(range(env.size))

plt.tight_layout()
plt.show()
