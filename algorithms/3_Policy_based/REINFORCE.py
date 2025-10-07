import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import animation
# -------------------------
# 1. 定义策略网络 (Policy Network)
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# -------------------------
# 2. REINFORCE Algorithm
# -------------------------
def reinforce(env_name='CartPole-v1', gamma=0.99, lr=1e-2, num_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        states = []
        actions = []
        rewards = []

        state, _ = env.reset(seed=42)
        done = False

        # -------------------------
        # 采样一条完整轨迹
        # -------------------------
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            action = np.random.choice(action_dim, p=probs.detach().numpy())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # -------------------------
        # Monte Carlo 回报计算
        # G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
        # -------------------------
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # -------------------------
        # 策略梯度更新
        # -------------------------
        optimizer.zero_grad()
        for state, action, Gt in zip(states, actions, returns):
            probs = policy(state)
            log_prob = torch.log(probs[action])
            loss = -log_prob * Gt  # REINFORCE loss (maximize expected return)
            loss.backward()
        optimizer.step()

        # -------------------------
        # 输出信息
        # -------------------------
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {sum(rewards)}")

    env.close()
    return policy

# -------------------------
# 3. 训练策略
# -------------------------
trained_policy = reinforce(num_episodes=500)


# -------------------------
# 训练好的策略生成轨迹
# -------------------------
def generate_trajectory(policy, env_name='CartPole-v1'):
    env = gym.make(env_name, render_mode='rgb_array')
    frames = []

    state, _ = env.reset(seed=42)
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = policy(state_tensor)
        action = np.random.choice(env.action_space.n, p=probs.detach().numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

    env.close()
    return frames

# -------------------------
# 显示动画
# -------------------------
def display_animation(frames):
    fig = plt.figure()
    im = plt.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.axis('off')
    plt.show()

# -------------------------
# Example
# -------------------------
# 使用前面训练好的策略生成动画
frames = generate_trajectory(trained_policy)
display_animation(frames)