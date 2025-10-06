import numpy as np

# Define the immediate reward vector r_pi
r_pi = np.array([0.5 * 0 + 0.5 * -1, 1, 1, 1])  # Example rewards

# Define the transition probability matrix P_pi
P_pi = np.array([
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])

# Discount factor gamma
gamma = 0.9

# Identity matrix I
I = np.eye(len(r_pi))

# Closed-form solution: v_pi = (I - gamma * P_pi)^(-1) * r_pi
v_pi = np.linalg.inv(I - gamma * P_pi) @ r_pi

print("Value function v_pi:")
print(v_pi)
