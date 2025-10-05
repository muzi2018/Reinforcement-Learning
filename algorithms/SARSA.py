import numpy as np
import matplotlib.pyplot as plt

# ===== 环境定义 =====
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = [0, 1, 2, 3]  # 上下左右
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # 上
            x = max(x - 1, 0)
        elif action == 1: # 下
            x = min(x + 1, self.size - 1)
        elif action == 2: # 左
            y = max(y - 1, 0)
        elif action == 3: # 右
            y = min(y + 1, self.size - 1)

        next_state = (x, y)
        reward = 0 if next_state == self.goal else -1
        done = next_state == self.goal
        self.state = next_state
        return next_state, reward, done

# ===== SARSA 算法 =====
def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size, env.size, 4))
    rewards_per_episode = []

    def choose_action(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.actions)
        return np.argmax(Q[state])

    for ep in range(episodes):
        state = env.reset()
        action = choose_action(state)
        total_reward = 0

        while True:
            next_state, reward, done = env.step(action)
            next_action = choose_action(next_state)
            
            # SARSA 更新
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

# ===== 运行 SARSA =====
env = GridWorld(size=4)
Q, rewards = sarsa(env)

# ===== 可视化结果 =====
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward per Episode")
plt.title("SARSA Learning Curve (GridWorld)")
plt.grid(True)
plt.show()

# ===== 输出最终策略 =====
directions = {0: "↑", 1: "↓", 2: "←", 3: "→"}
policy = np.full((env.size, env.size), "·", dtype=str)
for i in range(env.size):
    for j in range(env.size):
        if (i, j) == env.goal:
            policy[i, j] = "G"
        else:
            policy[i, j] = directions[np.argmax(Q[i, j])]
print("Learned SARSA Policy:")
print(policy)
