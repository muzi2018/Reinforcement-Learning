import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1️⃣ 定义目标函数 g(w)
# ================================
# 我们想找到 w，使 g(w) = 0
# 这里 g(w) = w - E[X]，根就是期望 E[X]
def g(w, E_X):
    return w - E_X  # 如果 w = E_X，则 g(w) = 0，正是我们想找的

# ================================
# 2️⃣ 模拟带噪声的观测
# ================================
# 我们不能直接知道 E[X]，只能通过样本观测
def noisy_observation(w, E_X, noise_std=1.0):
    x = np.random.normal(E_X, noise_std)  # 从正态分布生成一个样本 x ~ N(E[X], 1)
    eta = E_X - x  # 噪声 = 真实期望 - 样本
    return g(w, E_X) + eta  # 返回带噪声的函数值，模拟现实情况：测量带误差

# ================================
# 3️⃣ Robbins-Monro 随机逼近算法
# ================================
# 核心思想：每一步根据噪声观测调整 w，使它慢慢逼近真实解
def robbins_monro(E_X, initial_w=6.0, max_iter=100, alpha0=0.5):
    w = initial_w       # 我们一开始猜的 w，比如 w=6，离真实 E[X]=5 有差距
    estimates = [w]     # 记录每次迭代的 w 值，用于可视化

    for k in range(1, max_iter + 1):
        # ① 获取噪声观测
        noisy_g_w = noisy_observation(w, E_X)
        
        # ② 学习率 α_k
        # 随着迭代次数增加，步子慢慢变小，这样可以稳定收敛
        alpha_k = alpha0 / k

        # ③ 更新 w
        # w_new = w_old - α_k * noisy_g_w
        # 如果 noisy_g_w>0，说明 w 太大，需要减小
        # 如果 noisy_g_w<0，说明 w 太小，需要增大
        w = w - alpha_k * noisy_g_w

        # ④ 保存当前 w
        estimates.append(w)

    return np.array(estimates)

# ================================
# 4️⃣ 设置参数
# ================================
np.random.seed(42)     # 保证结果可复现
E_X = 5                # 我们想找的期望
initial_w = 6.0        # 初始猜测
max_iter = 100         # 最大迭代次数

# ================================
# 5️⃣ 运行算法
# ================================
w_estimates = robbins_monro(E_X, initial_w, max_iter)

# ================================
# 6️⃣ 可视化收敛过程
# ================================
plt.figure(figsize=(8, 5))
plt.plot(w_estimates, label="Estimate w_k")   # 每次迭代的 w 值
plt.axhline(E_X, color="red", linestyle="--", label=f"True E[X] = {E_X}")  # 真实解
plt.xlabel("Iteration k")
plt.ylabel("w_k")
plt.title("Robbins-Monro Stochastic Approximation")
plt.legend()
plt.grid(True)
plt.show()
