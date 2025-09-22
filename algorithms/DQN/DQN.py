import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt


# --- Q-network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, dropout=0.2):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.layers(x)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, dropout=0.2):
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim, dropout)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim, dropout)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return 0.0

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Compute Q(s,a)
        q_values = self.q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.gamma * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --- Training Loop ---
def train_dqn(env_name="CartPole-v1", episodes=500, batch_size=64, target_update=10):
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, dropout=0.2)
    replay_buffer = ReplayBuffer(capacity=10000)

    rewards, losses = [], []

    for episode in range(episodes):
        state = env.reset()
        done, total_reward = False, 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update(replay_buffer, batch_size)

            state = next_state
            total_reward += reward

        agent.update_epsilon()
        rewards.append(total_reward)
        if loss: losses.append(loss)

        # Update target network
        if episode % target_update == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Plotting
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title("Rewards")
    plt.subplot(1,2,2)
    plt.plot(losses)
    plt.title("Loss")
    plt.show()

    env.close()
    test_env.close()


if __name__ == "__main__":
    train_dqn()
