import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# -------------------------
# Actor网络
# -------------------------
class Actor(nn.Module):
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
# Critic网络
# -------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# -------------------------
# Actor-Critic训练函数（带动画）
# -------------------------
def actor_critic_render(env_name='CartPole-v1', gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, num_episodes=200):
    env = gym.make(env_name, render_mode='rgb_array')  # 使用rgb_array生成动画帧
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    frames = []  # 存储动画帧

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Actor选择动作
            probs = actor(state_tensor)
            action = np.random.choice(action_dim, p=probs.detach().numpy())
            
            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            
            # Critic更新
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            td_target = reward + gamma * next_value * (1 - int(done))
            td_error = td_target - value
            
            optimizer_critic.zero_grad()
            critic_loss = td_error.pow(2)
            critic_loss.backward()
            optimizer_critic.step()
            
            # Actor更新
            optimizer_actor.zero_grad()
            log_prob = torch.log(probs[action])
            actor_loss = -log_prob * td_error.detach()
            actor_loss.backward()
            optimizer_actor.step()
            
            state = next_state

            # 存储动画帧
            frame = env.render()
            frames.append(frame)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    env.close()
    return actor, critic, frames

# -------------------------
# 训练并获取动画帧
# -------------------------
trained_actor, trained_critic, frames = actor_critic_render(num_episodes=100)

# -------------------------
# 展示动画（在Jupyter Notebook中）
# -------------------------
fig = plt.figure()
patch = plt.imshow(frames[0])

def animate(i):
    patch.set_data(frames[i])
    return [patch]

# 保留anim对象，防止被垃圾回收
anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=30, blit=True)

# 在Notebook中显示
HTML(anim.to_jshtml())
