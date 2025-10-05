import numpy as np
import gymnasium as gym

# 创建环境，使用 GUI 渲染
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# 初始化 Q 表
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# Q-learning 超参数
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 2000
max_steps = 100

# Q-learning 主循环
for episode in range(num_episodes):
    state, info = env.reset()
    state = int(state)
    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = int(next_state)

        # Q-learning 更新公式
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        if done:
            break

print("训练完成！")

# 测试策略，图像展示
state, info = env.reset()
state = int(state)
done = False

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = int(next_state)
