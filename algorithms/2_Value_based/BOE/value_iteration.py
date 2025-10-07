import numpy as np

# Define the parameters
gamma = 0.9  # Discount factor
threshold = 1e-4  # Convergence threshold
states = ['s1', 's2', 's3', 's4']
actions = ['a1', 'a2', 'a3', 'a4', 'a5']
num_states = len(states)
num_actions = len(actions)

# Transition and reward model
# Format: {state: {action: [(next_state, probability, reward)]}}
transition_model = {
    's1': {
        'a1': [('s1', 1, -1)],
        'a2': [('s2', 1, -1)],
        'a3': [('s3', 1, 0)],
        'a4': [('s1', 1, -1)],
        'a5': [('s1', 1, 0)],
    },
    's2': {
        'a1': [('s2', 1, -1)],
        'a2': [('s2', 1, -1)],
        'a3': [('s4', 1, 1)],
        'a4': [('s1', 1, 0)],
        'a5': [('s2', 1, -1)],
    },
    's3': {
        'a1': [('s1', 1, 0)],
        'a2': [('s4', 1, 1)],
        'a3': [('s3', 1, -1)],
        'a4': [('s3', 1, -1)],
        'a5': [('s3', 1, 0)],
    },
    's4': {
        'a1': [('s2', 1, -1)],
        'a2': [('s4', 1, -1)],
        'a3': [('s4', 1, -1)],
        'a4': [('s3', 1, 0)],
        'a5': [('s4', 1, 1)],
    },
}

# Initialization
v = {state: 0 for state in states}  # Initial value function
policy = {state: None for state in states}  # Initial policy

# Value Iteration Algorithm
iteration = 0
while True:
    delta = 0
    q_values = {state: {action: 0 for action in actions} for state in states}  # Store q-values for this iteration
        
    # 1. Update q-values for each state and action
    for s in states:
        for a in actions:
            q_value = 0
            for next_state, prob, reward in transition_model[s][a]:
                q_value += prob * (reward + gamma * v[next_state])
            q_values[s][a] = q_value
        
    # 2. Update value function and policy
    for s in states:
        max_q = max(q_values[s].values())  # Max q-value
        best_action = max(q_values[s], key=q_values[s].get)  # Best action for current state
        delta = max(delta, abs(v[s] - max_q))  # Check for convergence
        v[s] = max_q  # Update value
        policy[s] = best_action  # Update policy
    
    iteration += 1
    if iteration == 1:
        # Print the Q-values table in a formatted manner
        print("\nTable 4.2: The value of q(s, a) at ", iteration)
        print("q-table      a1    a2    a3    a4    a5")
        # Format and print each state and its corresponding Q-values
        for state in states:
            print(f"{state: <10} {' '.join(f'{q_values[state][a]: >4.2f}' for a in actions)}")
    if delta < threshold:  # Check convergence
        break

# Display results
print(f"Converged after {iteration} iterations.")
print("\nOptimal Value Function:")
for s in states:
    print(f"V({s}) = {v[s]:.4f}")

print("\nOptimal Policy:")
for s in states:
    print(f"π({s}) = {policy[s]}")
