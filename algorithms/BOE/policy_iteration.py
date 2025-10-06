import numpy as np

# Define the MDP parameters
gamma = 0.9  # Discount factor
states = [0, 1]  # State space
actions = [0, 1]  # Action space
num_states = len(states)
num_actions = len(actions)

# Reward function p[r | s, a]: Expected rewards for each state-action pair
p_r = {
    (0, 0): 0,  # Reward for taking action 0 in state 0
    (0, 1): 1,  # Reward for taking action 1 in state 0
    (1, 0): 0,  # Reward for taking action 0 in state 1
    (1, 1): 1   # Reward for taking action 1 in state 1
}

# Transition probabilities p[s' | s, a]: Probabilities of transitioning to s' given (s, a)
p_s_prime = {
    (0, 0): {0: 0.8, 1: 0.2},  # From state 0, action 0 leads to state 0 (80%) or state 1 (20%)
    (0, 1): {0: 0.1, 1: 0.9},  # From state 0, action 1 leads to state 0 (10%) or state 1 (90%)
    (1, 0): {0: 0.7, 1: 0.3},  # From state 1, action 0 leads to state 0 (70%) or state 1 (30%)
    (1, 1): {0: 0.4, 1: 0.6}   # From state 1, action 1 leads to state 0 (40%) or state 1 (60%)
}

# Initialize a random policy (uniform random over actions) and value function
policy = np.ones((num_states, num_actions)) / num_actions  # Equal probability for each action
value_function = np.zeros(num_states)  # Initial value function

# Policy Iteration Algorithm
def policy_iteration():
    global policy, value_function
    is_policy_stable = False  # To check if the policy has converged

    while not is_policy_stable:
        # POLICY EVALUATION
        while True:
            delta = 0
            for s in range(num_states):
                # Store the current value
                v = value_function[s]
                # Compute the value function under the current policy
                value_function[s] = sum(policy[s, a] * (
                    p_r[(s, a)] + gamma * sum(
                        p_s_prime[(s, a)][s_prime] * value_function[s_prime] for s_prime in range(num_states)
                    )
                ) for a in range(num_actions))
                # Update the maximum change (delta)
                delta = max(delta, abs(v - value_function[s]))
            if delta < 1e-6:  # Stop if values converge
                break

        # POLICY IMPROVEMENT
        is_policy_stable = True  # Assume policy is stable
        for s in range(num_states):
            old_action = np.argmax(policy[s])  # Current best action
            # Compute the Q-values for all actions in the current state
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_values[a] = p_r[(s, a)] + gamma * sum(
                    p_s_prime[(s, a)][s_prime] * value_function[s_prime] for s_prime in range(num_states)
                )
            # Determine the best action based on Q-values
            best_action = np.argmax(q_values)
            # Update policy to be greedy (100% probability on the best action)
            policy[s] = np.zeros(num_actions)
            policy[s][best_action] = 1
            # If the policy changes, it's not stable
            if old_action != best_action:
                is_policy_stable = False

    return policy, value_function

# Run the policy iteration algorithm
optimal_policy, optimal_value_function = policy_iteration()

# Print the results
print("Optimal Policy (each row corresponds to a state):")
for s in range(num_states):
    print(f"State {s}: {optimal_policy[s]} (Action probabilities)")

print("\nOptimal Value Function:")
for s in range(num_states):
    print(f"State {s}: {optimal_value_function[s]:.4f}")





# import numpy as np

# # Define the number of states and actions
# num_states = 4  # Number of states
# num_actions = 2  # Number of actions

# # Create a 3D transition probability matrix P[state, action, next_state]
# P = np.array([
#     [[0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]],  # Transitions from state 0 under action 0 and 1
#     [[0.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0]],  # Transitions from state 1 under action 0 and 1
#     [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],  # Transitions from state 2 (absorbing)
#     [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],  # Transitions from state 3 (absorbing)
# ])

# # Define the reward function for each state and action
# r = np.array([
#     [1, 0],  # Rewards for actions at state 0
#     [0, 1],  # Rewards for actions at state 1
#     [0, 0],  # Rewards for actions at state 2
#     [0, 0],  # Rewards for actions at state 3
# ])

# gamma = 0.9  # Discount factor

# # Initialize policies
# policy_k = np.random.choice(num_actions, size=num_states)  # Random policy
# policy_k1 = np.zeros(num_states, dtype=int)  # Initializing the next policy

# # Function for policy evaluation
# def policy_evaluation(P, r, gamma, policy, tol=1e-6, max_iterations=1000):
#     num_states = len(r)
#     v = np.zeros(num_states)  # Initial value function
#     v_history = np.zeros((max_iterations, num_states))
#     for iteration in range(max_iterations):
#         v_new = np.zeros(num_states)
#         for state in range(num_states):
#             action = policy[state]
#             v_new[state] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
#         v_history[iteration] = v_new  # Store the current v_new in history
#         if np.linalg.norm(v_new - v, ord=np.inf) < tol:
#             v_history = v_history[:iteration + 1]  # Trim unused rows
#             break
#         v = v_new
    
#     return v, v_history

# # Function for policy improvement
# def policy_improvement(P, r, gamma, v):
#     num_states = len(r)
#     new_policy = np.zeros(num_states, dtype=int)
    
#     for state in range(num_states):
#         action_values = np.zeros(num_actions)
#         for action in range(num_actions):
#             action_values[action] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
#         new_policy[state] = np.argmax(action_values)
    
#     return new_policy

# # Run policy iteration for 100 iterations
# num_iterations = 100
# for i in range(num_iterations):
#     # Evaluate the current policy
#     v_policy_k, v_policy_k_buff = policy_evaluation(P, r, gamma, policy_k)
    
#     # Improve the policy based on the value function

    
#     # Check the change in policy
#     if np.array_equal(policy_k, policy_k1):
#         print(f"Policy converged after {i+1} iterations.")
#         break
#     policy_k1 = policy_improvement(P, r, gamma, v_policy_k)
    
#     v_policy_k1, v_policy_k1_buff = policy_evaluation(P, r, gamma, policy_k1)
#     difference = v_policy_k_buff - v_policy_k1_buff
#     print("Difference between v_{π_k} and v_{π_{k+1}}:", difference)
#     # Update policy_k for the next iteration
#     policy_k = policy_k1

# # Print final policies and value functions
# print("Final policy π_k:", policy_k)
# print("Final policy π_{k+1}:", policy_k1)

# v_policy_k, _ = policy_evaluation(P, r, gamma, policy_k)
# v_policy_k1, _ = policy_evaluation(P, r, gamma, policy_k1)

# print("Value function v_{π_k}:", v_policy_k)
# print("Value function v_{π_{k+1}}:", v_policy_k1)

# exit()








# ## Proof of Lemma 4.1
# import numpy as np

# # Define parameters
# gamma = 0.9  # Discount factor
# num_states = 2  # Number of states
# num_actions = 2  # Number of actions
# iterations = 10  # Number of iterations for the simulation

# # Transition probabilities and rewards
# P = np.array([[0.8, 0.2],  # Transition probabilities for action 0
#               [0.2, 0.8]])  # Transition probabilities for action 1
# rewards = np.array([0, 1])  # Rewards for each state

# # Initialize value functions
# v_pi_k = np.zeros(num_states)
# v_pi_k_plus_1 = np.zeros(num_states)

# # Policy iteration process
# for k in range(iterations):
#     # Policy evaluation step
#     v_pi_k_plus_1 = rewards + gamma * np.dot(P, v_pi_k)

#     # Policy improvement step
#     # Here we assume a simple policy improvement based on the max value
#     v_pi_k = v_pi_k_plus_1

#     # Print the values at each iteration
#     print(f"Iteration {k + 1}: v_pi_k = {v_pi_k}")

#     # Check for convergence
#     if np.allclose(v_pi_k, v_pi_k_plus_1):
#         print("Convergence reached.")
#         break

# # Final values
# print("Final value function:", v_pi_k)