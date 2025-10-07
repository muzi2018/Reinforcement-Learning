import numpy as np
import matplotlib.pyplot as plt

# Simple gridworld environment
class Gridworld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), gamma=0.9):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        self.gamma = gamma

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            next_state = (max(0, x - 1), y)
        elif action == 1:  # down
            next_state = (min(self.size - 1, x + 1), y)
        elif action == 2:  # left
            next_state = (x, max(0, y - 1))
        elif action == 3:  # right
            next_state = (x, min(self.size - 1, y + 1))
        
        reward = -1  # step cost
        done = next_state == self.goal
        if done:
            reward = 0  # no penalty at the goal

        self.state = next_state
        return next_state, reward, done

    def reset(self):
        self.state = self.start
        return self.start

# n-step Sarsa implementation
def n_step_sarsa(env, n, alpha, epsilon, episodes):
    Q = np.zeros((env.size, env.size, 4))  # state-action values
    returns = []

    for ep in range(episodes):
        state = env.reset()
        action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(Q[state])
        
        states = [state]
        actions = [action]
        rewards = [0]  # placeholder for initial reward

        T = float('inf')  # time until episode ends
        t = 0  # current time step
        while True:
            if t < T:
                next_state, reward, done = env.step(action)
                rewards.append(reward)
                if done:
                    T = t + 1
                else:
                    next_action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(Q[next_state])
                    states.append(next_state)
                    actions.append(next_action)
                    action = next_action
            
            tau = t - n + 1
            if tau >= 0:
                G = sum(env.gamma ** (i - tau) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T:
                    G += env.gamma ** n * Q[states[tau + n]][actions[tau + n]]
                Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]])
            
            if tau == T - 1:
                break
            t += 1

        # Collect return for analysis
        returns.append(np.mean([np.max(Q[x, y]) for x in range(env.size) for y in range(env.size)]))
    
    return Q, returns

# Experiment to measure bias and variance
def experiment():
    env = Gridworld()
    episodes = 500
    alpha = 0.1
    epsilon = 0.1
    gamma = env.gamma
    true_Q = np.zeros((env.size, env.size, 4))  # Assume known true Q-values for simplicity
    
    # Run for different n
    n_values = [1, 2, 4, 8, 16]
    results = {}
    for n in n_values:
        Q, returns = n_step_sarsa(env, n, alpha, epsilon, episodes)
        bias = np.mean((Q - true_Q) ** 2)
        variance = np.var(Q)
        results[n] = (bias, variance, returns)

    # Plot bias and variance
    biases = [results[n][0] for n in n_values]
    variances = [results[n][1] for n in n_values]
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, biases, label='Bias', marker='o')
    plt.plot(n_values, variances, label='Variance', marker='o')
    plt.xlabel('n-step')
    plt.ylabel('Value')
    plt.title('Bias and Variance of Q Estimates vs n-step')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot returns
    plt.figure(figsize=(12, 6))
    for n in n_values:
        plt.plot(results[n][2], label=f'n={n}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Returns vs Episodes')
    plt.legend()
    plt.grid()
    plt.show()

experiment()
