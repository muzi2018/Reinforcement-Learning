# train.py
import argparse
import gym
from algorithms.ppo.ppo import PPO

def train(env_name, total_timesteps, lr, gamma):
    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    # initialize PPO agent
    agent = PPO(obs_dim, act_dim, lr=lr, gamma=gamma)

    obs, _ = env.reset()
    for t in range(total_timesteps):
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.update()  # PPO update step
        obs = next_obs if not done else env.reset()[0]

        if t % 1000 == 0:
            print(f"Step {t} completed")

    env.close()
    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    args = parser.parse_args()

    train(args.env, args.timesteps, args.lr, args.gamma)


