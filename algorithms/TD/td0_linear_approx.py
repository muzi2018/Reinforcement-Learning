import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.grid_world import GridWorld
import random
import numpy as np
import random
import matplotlib.pyplot as plt


# TD-Linear algorithm implementation
class TDLinear:
    def __init__(self, grid_world, alpha=0.1, gamma=0.9):
        self.grid_world = grid_world
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.weights = np.zeros(grid_world.num_states)  # Initialize weights

    def feature_vector(self, state):
        """
        Create a feature vector using one-hot encoding for the given state.
        """
        features = np.zeros(self.grid_world.num_states)
        state_idx = self.grid_world.state_index[state]  # Convert (x, y) to a single index
        features[state_idx] = 1
        return features

    def td_update(self, state, reward, next_state):
        """
        Apply the TD(0) update rule to adjust weights.
        """
        phi_s = self.feature_vector(state)  # Feature vector for current state
        phi_s_next = self.feature_vector(next_state)  # Feature vector for next state
        
        # TD(0) target calculation
        target = reward + self.gamma * np.dot(self.weights, phi_s_next)
        
        # TD(0) weight update
        self.weights += self.alpha * (target - np.dot(self.weights, phi_s)) * phi_s

    def estimate_state_values(self, num_episodes, policy):
        """
        Estimate state values using the TD(0) algorithm over multiple episodes.
        """
        for _ in range(num_episodes):
            current_state = random.choice(self.grid_world.states)
            while True:
                actions_and_probabilities = policy[current_state]
                actions_list = list(actions_and_probabilities.keys())
                probabilities = list(actions_and_probabilities.values())
                chosen_action =  random.choices(actions_list, probabilities)[0]
                next_state, reward, done, info = self.grid_world.step(chosen_action)
                
                self.td_update(current_state, reward, next_state)
                
                # Stop when reaching a terminal state or predefined condition (if applicable)
                # In this example, there are no terminal states, so the loop runs indefinitely
                # until the episode length reaches a limit (not implemented here)
                current_state = next_state


#
if __name__ == "__main__":      
    env = GridWorld()
    # Parameters
    gamma = 0.9  # Discount factor
    epsilon = 0.5  # Exploration rate
    num_episodes = 1
    grid_size = 5  # 5x5 grid

    # aciton 
    # up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = [(0,-1), (0, 1), (-1,0), (1,0)]
    num_actions = len(actions)
    '''
    reward setup
    rboundary = -1
    rforbidden = -1
    rstep = 0
    rtarget = 1
    '''
    # reward
    env.reward_boundary = -1
    env.reward_forbidden = -1
    env.reward_step = 0
    env.reward_target = 1
    # Create grid 
    # env, row->x, column->y
    env.env_size = (grid_size, grid_size)
    env.num_states = grid_size * grid_size
    env.forbidden_states = [(1, 0)]
    env.target_state = (1, 1)
    env.reset()
    td_linear = TDLinear(env, alpha=0.1, gamma=0.9)

    policy = {
        (i, j): {action: 1 / num_actions for action in actions} 
        for i in range(grid_size) for j in range(grid_size)
    }
    
    td_linear.estimate_state_values(1, policy)




