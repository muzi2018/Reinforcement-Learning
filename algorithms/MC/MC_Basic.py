import numpy as np
import matplotlib.pyplot as plt

# Gridworld setup
grid_size = 5
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]

# Rewards
reward_boundary = -1
reward_forbidden = -10
reward_target = 1
target_state = (2, 2)  # Target state
forbidden_states = [(1, 1), (1, 3), (3, 1), (3, 3)]  # Forbidden states
discount_rate = 0.9

# Define rewards for each state
rewards = {state: reward_boundary for state in states}
for state in forbidden_states:
    rewards[state] = reward_forbidden
rewards[target_state] = reward_target

# Actions (up, down, left, right)
actions = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Helper functions
def is_valid_state(state):
    """Check if a state is valid (within bounds and not forbidden)."""
    return (
        0 <= state[0] < grid_size
        and 0 <= state[1] < grid_size
        and state not in forbidden_states
    )

def get_next_state(state, action):
    """Get the next state given a current state and action."""
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    return next_state if is_valid_state(next_state) else state

# Monte Carlo algorithm
def monte_carlo(grid_size, rewards, episode_length, num_episodes):
    # Initialize state value estimates and returns
    value_estimates = {state: 0 for state in states}
    returns = {state: [] for state in states}
    policy = {state: "up" for state in states}  # Random initial policy

    for _ in range(num_episodes):
        state = (0, 0)  # Start at top-left corner
        episode = []

        # Generate an episode of fixed length
        for _ in range(10):
            if state == target_state:
                break
            action = np.random.choice(list(actions.keys()))  # Random action
            next_state = get_next_state(state, action)
            episode.append((state, action, rewards[next_state]))
            state = next_state

        # Backward pass to compute returns and update value estimates
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + discount_rate * G
            if state not in [s for s, _, _ in episode[:-1]]: # first-visit
                returns[state].append(G)
                value_estimates[state] = np.mean(returns[state])

    # Update policy greedily based on value estimates
    for state in states:
        if state == target_state or state in forbidden_states:
            continue
        action_values = {
            action: rewards[get_next_state(state, action)]
            + discount_rate * value_estimates[get_next_state(state, action)]
            for action in actions
        }
        policy[state] = max(action_values, key=action_values.get)

    return value_estimates, policy

# Visualization
def visualize(grid_size, value_estimates, policy, title):
    """Visualize the grid with value estimates and policy arrows."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for (i, j), value in value_estimates.items():
        ax.text(j, grid_size - i - 1, f"{value:.1f}", ha="center", va="center")
        if (i, j) in policy and (i, j) != target_state:
            action = policy[(i, j)]
            if action == "up":
                ax.arrow(j, grid_size - i - 1, 0, 0.3, head_width=0.2, color="green")
            elif action == "down":
                ax.arrow(j, grid_size - i - 1, 0, -0.3, head_width=0.2, color="green")
            elif action == "left":
                ax.arrow(j, grid_size - i - 1, -0.3, 0, head_width=0.2, color="green")
            elif action == "right":
                ax.arrow(j, grid_size - i - 1, 0.3, 0, head_width=0.2, color="green")

    ax.add_patch(plt.Rectangle((target_state[1] - 0.5, grid_size - target_state[0] - 1.5), 1, 1, fill=True, color="cyan"))
    for forbidden in forbidden_states:
        ax.add_patch(plt.Rectangle((forbidden[1] - 0.5, grid_size - forbidden[0] - 1.5), 1, 1, fill=True, color="orange"))
    ax.set_title(title)
    plt.show()

# Experiment with different episode lengths
episode_lengths = [1, 2, 3, 4]
num_episodes = 1000

for episode_length in episode_lengths:
    value_estimates, policy = monte_carlo(grid_size, rewards, episode_length, num_episodes)
    visualize(grid_size, value_estimates, policy, f"Episode length = {episode_length}")
