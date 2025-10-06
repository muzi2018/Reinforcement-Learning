import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.grid_world import GridWorld
import random
import numpy as np
import random
# 3 - Chapter 2 State Values and Bellman Equation
# 1. state value: which is defined as the average reward that an agent can obtain if it follows a given policy
# 2. Bellman equation which is an important tool for analyzing state values, in a nutshell, Bellman equation describes the relationships between the values of all states. By solving the Bellman equation , we can obtain the state values
# 3. policy evaluation: By solving the Bellman equation to obtain the state values.
# 4. action value

#
if __name__ == "__main__":      
    # 2.5 Examples for illustrating the Bellman equation
    env = GridWorld()
    ## aciton 
    ### up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = {
        'up': env.action_space[2],    # Action for moving up
        'down': env.action_space[0],   # Action for moving down
        'left': env.action_space[3],   # Action for moving left
        'right': env.action_space[1],  # Action for moving right
        'stay': env.action_space[4]     # Action for staying in place
    }

    ## state
    states = {
        's1': (0, 0),  # State s1 at coordinates (0, 0)
        's2': (1, 0),  # State s2 at coordinates (1, 0)
        's3': (0, 1),  # State s3 at coordinates (0, 1)
        's4': (1, 1)   # State s4 at coordinates (1, 1)
    }
    ## reward
    env.reward_forbidden = -1
    env.reward_step = 0
    env.reward_target = 1
    # ## env, row->x, column->y
    env.env_size = (2, 2)
    env.num_states = 4
    env.start_state = states['s2']
    env.forbidden_states = [(1, 0)]
    env.target_state = (1, 1)
    
    ## Policy    
    policy = {
        's1': {
            'up': 0.0,    # Probability of taking action 'up' in state s1
            'down': 0.5,  # Probability of taking action 'down' in state s1
            'left': 0.0,  # Probability of taking action 'left' in state s1
            'right': 0.5, # Probability of taking action 'right' in state s1
            'stay': 0.0   # Probability of taking action 'stay' in state s1
        },
        's2': {
            'up': 0.0,    # Define probabilities for state s2 (example values)
            'down': 1.0,
            'left': 0.0,
            'right': 0.0,
            'stay': 0.0   # Example: only staying in state s2
        },
        's3': {
            'up': 0.0,    
            'down': 0.0,
            'left': 0.0,
            'right': 1.0,
            'stay': 0.0   # Example: only staying in state s3
        },
        's4': {
            'up': 0.0,    
            'down': 0.0,
            'left': 0.0,
            'right': 0.0,
            'stay': 1.0   # Example: only staying in state s4
        }
    }

    action_values = {state: {action: 0 for action in actions.keys()} for state in states.keys()}
    sum_action_values = {state: {action: [] for action in actions.keys()} for state in states.keys()}

    gamma_ = 0.9
    num_iterations = 100  # Number of iterations for convergence

    # Function to generate an episode
    def generate_episode():
        episode = []
        # start from s1
        env.start_state = states['s1']
        env.reset()
        _state_name = 's1'
        while True:
            actions_list = list(policy[_state_name].keys())
            probobilities = list(policy[_state_name].values())
            chosen_action =  random.choices(actions_list, probobilities)[0]
            next_state, reward, done, info = env.step(actions[chosen_action])
            episode.append((_state_name, chosen_action, reward))
            
            for name, coords in states.items():
                if coords == next_state:
                    next_state_name = name
                    break
            _state_name = next_state_name
            if _state_name == 's4':
                break
        return episode
        print('generate_episode')
    # generate_episode()

    def update_action_values(episodes):
        for episode in episodes:
            G = 0 # return
            for state_name, action, reward in reversed(episode):
                G = reward + gamma_ * G
                sum_action_values[state_name][action].append(G)
                action_values[state_name][action] = np.mean(sum_action_values[state_name][action])
# Generate episodes and update action values
num_episodes = 300
episodes = [generate_episode() for _ in range(num_episodes)]
update_action_values(episodes)

# Print the action values
print("Estimated action values (q-values):")
for state_name, actions_dict in action_values.items():  # Outer loop for states
    for action, value in actions_dict.items():  # Inner loop for actions
        print(f"q({state_name}, {action}) = {value:.2f}")
