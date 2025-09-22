import gym
import numpy as np

# --- Discretization helper ---
def discretize_state(state, bins):
    """Discretize continuous state into tuple of indices"""
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indices)


# --- Environment ---
env = gym.make("CartPole-v1")

# Discretization bins (for each dimension of state)
n_bins = [6, 12, 6, 12]   # coarse discretization
bins = [
    np.linspace(-4.8, 4.8, n_bins[0]),   # cart position[-4.8, -3.2, -1.6, 0, 1.6, 3.2, 4.8]
    np.linspace(-5, 5, n_bins[1]),       # cart velocity[-5, -4.2, -3.4, -2.6, -1.8, -1.0, -0.2, 0.6, 1.4, 2.2, 3.0, 3.8, 4.6, 5]
    np.linspace(-0.418, 0.418, n_bins[2]), # pole angle[-0.418, --0.278, -0.139, 0, 0.139, 0.278, 0.418]
    np.linspace(-5, 5, n_bins[3])        # pole angular velocity[-5, -4.2, -3.4, -2.6, -1.8, -1.0, -0.2, 0.6, 1.4, 2.2, 3.0, 3.8, 4.6, 5]
]

# Q-table
state_space = tuple(n_bins)
n_actions = env.action_space.n
Q = np.zeros(state_space + (n_actions,))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 500

# --- Training Loop ---
for episode in range(episodes):
    state = env.reset()
    state = discretize_state(state, bins)
    done = False
    total_reward = 0

    while not done:
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Step in environment
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state, bins)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action] * (1 - int(done))
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()
