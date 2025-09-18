# Reinforcement-Learning

Minimal and clear **Reinforcement Learning (RL)** implementations in PyTorch.

🤖 机器人 | 🎮 游戏 | 📊 研究

Currently includes  **PPO** , with more algorithms coming soon.

---

# 📦 Implemented Algorithms

* [X] **Proximal Policy Optimization (PPO)**
* [ ] Deep Q-Learning (DQN)
* [ ] Advantage Actor-Critic (A2C)
* [ ] Soft Actor-Critic (SAC)
* [ ] More coming soon...

---

# ▶️ Usage

Run PPO on a selected environment (e.g., CartPole):

```bash
git clone git@github.com:muzi2018/Reinforcement-Learning.git
cd Reinforcement-Learning
conda create -n rl_env python=3.11
conda activate rl_env
pip install -r requirements.txtcd algorithms/ppo
python main.py
```

---

# 🧪 Example Environments

* **Gym Classic Control** (CartPole, MountainCar, LunarLander)
* **Atari Games** (e.g., Pong, Breakout)
* **Robotics** (coming soon with PyBullet / Isaac Gym)
