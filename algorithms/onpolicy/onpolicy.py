import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --------- policy network ---------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.fc(x)
        return self.actor(hidden), self.critic(hidden)

# --------- PPO algorithm ---------
class PPO:
    def __init__(self, obs_dim, act_dim, clip_ratio=0.2, lr=3e-3):
        self.policy = PolicyNet(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio

    def get_action(self, obs):
        logits, value = self.policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, obs, actions, log_probs_old, returns, advantages, epochs=10):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        actions = torch.as_tensor(actions)
        log_probs_old = torch.as_tensor(log_probs_old, dtype=torch.float32)
        returns = torch.as_tensor(returns, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)

        for _ in range(epochs):
            logits, values = self.policy(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)

            # calculate ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(log_probs - log_probs_old)

            # PPO objective
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = ((returns - values.squeeze()) ** 2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# --------- train loop ---------
def train(env_name="CartPole-v1", episodes=200):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPO(obs_dim, act_dim)
    gamma = 0.99

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        log_probs, values, rewards, states, actions = [], [], [], [], []

        while not done:
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob.detach())
            values.append(value.detach())
            rewards.append(reward)

            obs = next_obs

        # calculate returns and advantages
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - torch.cat(values).squeeze()

        agent.update(states, actions, log_probs, returns, advantages)

        print(f"Episode {ep}, total reward = {sum(rewards)}")

    env.close()

if __name__ == "__main__":
    train()
