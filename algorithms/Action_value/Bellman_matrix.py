import sys
import os
from grid_world import GridWorld
import random
import numpy as np
import random

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
    ##reward prob
    
    reward_probs = {
        ('s1', 'up'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},  
        ('s1', 'down'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},
        ('s1', 'left'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s1', 'right'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s1', 'stay'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},

        ('s2', 'up'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},  
        ('s2', 'down'): {env.reward_forbidden: 0, env.reward_step: 0, env.reward_target: 1},
        ('s2', 'left'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},
        ('s2', 'right'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s2', 'stay'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},


        ('s3', 'up'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},  
        ('s3', 'down'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s3', 'left'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s3', 'right'): {env.reward_forbidden: 0, env.reward_step: 0, env.reward_target: 1},   
        ('s3', 'stay'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},

        ('s4', 'up'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},  
        ('s4', 'down'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s4', 'left'): {env.reward_forbidden: 0, env.reward_step: 1, env.reward_target: 0},
        ('s4', 'right'): {env.reward_forbidden: 1, env.reward_step: 0, env.reward_target: 0},
        ('s4', 'stay'): {env.reward_forbidden: 0, env.reward_step: 0, env.reward_target: 1}

    }    
    
    action_values = {state: {action: 0 for action in actions.keys()} for state in states.keys()}
    sum_action_values = {state: {action: [] for action in actions.keys()} for state in states.keys()}

    gamma_ = 0.9
    num_iterations = 100  # Number of iterations for convergence

    # q_{\pi}
    _num_state = 4
    _num_action = 5
    q_pi = np.zeros((_num_state, _num_action))
    print("Expected Reward Matrix (q_pi):")
    print(q_pi)

    tilde_r = np.zeros((_num_state, _num_action))
    # Create a mapping of state names to indices
    state_to_index = {state: i for i, state in enumerate(states.keys())}
    action_to_index = {action: j for j, action in enumerate(actions.keys())}
    # Populate the tilde_r matrix
    for state_name, state_index in state_to_index.items():
        for action_name, action_index in action_to_index.items():
            # Get the reward probabilities for this state-action pair
            if (state_name, action_name) in reward_probs:
                reward_distribution = reward_probs[(state_name, action_name)]
                
                # Compute the expected reward (weighted sum)
                expected_reward = sum(r * prob for r, prob in reward_distribution.items()) #r_{i,j}=\sum_{r\in R}p(r|s_i, a_j)r
                tilde_r[state_index, action_index] = expected_reward
            else:
                # Default to 0 if no reward probabilities are defined
                tilde_r[state_index, action_index] = 0

    # Print the tilde_r matrix
    print("Expected Reward Matrix (tilde_r):")
    print(tilde_r)
    print("ok")






    # # Example: Populate the q_pi matrix with some arbitrary values
    # # Replace this with your actual computation of q_pi values
    # for i in range(_num_state):
    #     for j in range(_num_action):
    #         q_pi[i, j] = np.random.random()  # Random values as a placeholder

    # # Print the q_pi matrix
    # print("Action-Value Function (q_pi):")
    # print(q_pi)

    # # Access individual elements (e.g., q_pi(s1, a1))
    # state_index = 0  # State s1 corresponds to index 0
    # action_index = 0  # Action a1 corresponds to index 0
    # print(f"q_pi(s1, a1): {q_pi[state_index, action_index]:.4f}")


