import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1️⃣ 定义问题
# ===========================
# 我们假设 X 是一个随机变量，真实期望值 E[X] = 5
E_X = 5
std_X = 1  # 标准差
num_samples = 1000  # 我们可以生成一些样本来模拟观测

# 生成样本
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子保证可复现
np.random.seed(42)

# 参数
E_X = 5      # 期望
std_X = 1    # 标准差
num_samples = 1000  # 样本数量

# 生成样本
X_samples = np.random.normal(loc=E_X, scale=std_X, size=num_samples)

# # 绘图
# plt.figure(figsize=(8, 6))
# plt.hist(X_samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
# plt.axvline(E_X, color='red', linestyle='--', label=f"E[X] = {E_X}")
# plt.xlabel("Sample values")
# plt.ylabel("Frequency")
# plt.title("Histogram of Generated Samples from N(E[X], std^2)")
# plt.legend()
# plt.grid(True)
# plt.show()
# exit()

# 定义损失函数 J(w) = E[(w - X)^2]
def J(w, x):
    """计算 w 对样本 x 的平方误差"""
    return (w - x)**2

# 定义梯度
def grad_J(w, x):
    """计算 w 对样本 x 的梯度"""
    return 2 * (w - x)

# ===========================
# 2️⃣ SGD算法
# ===========================
alpha = 0.1  # 学习率
n_iterations = 50  # 迭代次数

w = 0  # 初始猜测
w_history = [w]

for k in range(n_iterations):
    # 随机选一个样本 x_k
    x_k = np.random.choice(X_samples)
    
    # 计算梯度
    gradient = grad_J(w, x_k)
    
    # SGD 更新
    w = w - alpha * gradient
    
    # 保存历史
    w_history.append(w)

# ===========================
# 3️⃣ 输出结果
# ===========================
print(f"SGD 最终估计 w ≈ {w:.4f}")
print(f"真实期望值 E[X] = {E_X}")

# ===========================
# 4️⃣ 可视化
# ===========================
# 绘制 J(w) 曲线
w_vals = np.linspace(0, 10, 100)
J_vals = [np.mean(J(wv, X_samples)) for wv in w_vals]

plt.figure(figsize=(8, 6))
plt.plot(w_vals, J_vals, label="J(w) = E[(w - X)^2]", color='blue')
plt.scatter(w_history, [np.mean(J(wv, X_samples)) for wv in w_history], color='red', label="SGD path")
plt.axvline(E_X, color='green', linestyle='--', label="True E[X]")
plt.xlabel("w")
plt.ylabel("J(w)")
plt.title("SGD Convergence to Minimize J(w)")
plt.legend()
plt.grid(True)
plt.show()
