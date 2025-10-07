# Reinforcement-Learning

Minimal and clear **Reinforcement Learning (RL)** implementations in PyTorch.

🤖 机器人 | 🎮 游戏 | 📊 研究

Currently includes  **PPO** , with more algorithms coming soon.

---

# 📦 Implemented Algorithms

* [X] **1_Value_calculation**

  * [X] **Action_value_calculation**
    * [X] **Bellman_matrix**
    * [X] **Monte_Carlo**
  * [X] **State_value_calculation**
    * [X] **Bellman_analytical**
    * [X] **Bellman_iteration**
* [X] **2_Value_based**

  * [X] **BOE(Analytic)**

    * [X] **Value_iteration**
    * [X] **Policy_iteration**
  * [X] **TD**

    * [X] **Robbins-Monro(理论基础)**
    * [X] **DQN(TD算法在函数近似上的实现)**
    * [X] **TD_SARSA_1step_action_value.py**
    * [X] **TD_SARSA_1step_state_value.py**
    * [X] **TD_SARSA_nstep.py**
    * [X] **td0_linear_approx.py**
* [X] **3_Policy_based**
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
conda env create -f environment.yml
conda activate rl_env
python main.py
```

---

# 🧪 Example Environments

* **Gym Classic Control** (CartPole, MountainCar, LunarLander)
* **Atari Games** (e.g., Pong, Breakout)
* **Robotics** (coming soon with PyBullet / Isaac Gym)
