import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

"""
DQN: two networks (q_net, target_net) , Predict Q Network (the q value of each action in the current state) + Target Q Network(the q value of each action in the next state)
"""


# ---------- Q-Network ----------
class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.fc(x)

# ---------- DQN Agent ----------
class DQN:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.q_net = QNet(obs_dim, act_dim)
        self.target_net = QNet(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict()) # gets the weights of the online Q-network.

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=50000)  # replay buffer
        self.batch_size = 64
        self.act_dim = act_dim

    def get_action(self, obs):
        if random.random() < self.epsilon:  # exploration
            return random.randrange(self.act_dim)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs)
        return torch.argmax(q_values, dim=1).item()

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, done = zip(*batch)

        obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.long).unsqueeze(1)
        rew = torch.as_tensor(rew, dtype=torch.float32).unsqueeze(1)
        next_obs = torch.as_tensor(np.array(next_obs), dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32).unsqueeze(1)

        # Compute Q-values
        q_values = self.q_net(obs).gather(1, act)

        # Compute target
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(1, keepdim=True)[0]
            target = rew + (1 - done) * self.gamma * max_next_q

        loss = ((q_values - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ---------- Training Loop ----------
def train(env_name="CartPole-v1", episodes=200):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQN(obs_dim, act_dim)
    target_update_interval = 20

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store((obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward

            # Train on random batch from replay buffer
            agent.update()

        # Update target network every few episodes
        if ep % target_update_interval == 0:
            agent.update_target()

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {ep}, total reward = {total_reward}, epsilon = {agent.epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    train()
