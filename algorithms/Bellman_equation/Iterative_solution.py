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

# Initial guess for v_pi
v_k = np.zeros(len(r_pi))

# Convergence threshold and maximum iterations
tolerance = 1e-6
max_iterations = 1000

# Iterative computation
for iteration in range(max_iterations):
    v_k_next = r_pi + gamma * np.dot(P_pi, v_k)  # Bellman update
    if np.linalg.norm(v_k_next - v_k, ord=np.inf) < tolerance:  # Check for convergence
        print(f"Converged after {iteration + 1} iterations.")
        break
    v_k = v_k_next

# Print the final value function
print("Value function v_pi:")
print(v_k)
