import gym
import numpy as np

# 1. Intialize the environment
env = gym.make('CartPole-v1', render_mode='human')

# 2. Define Q-Learning Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.99 # Discount factor
epsilon = 1.0 # Initial exploration probability
epsilon_min = 0.01 # Minimum exploration probasbility
epsilon_decay = 0.995 # Exploration decay rate
episodes = 1000 # Number of training episodes
 
# 3. Discretize the state space
state_bins = [10, 10, 10, 10] # Number of bins for each state variable (position, velocity, etc.)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5] # Clip velocity range
state_bounds[3] = [-50, 50] # Clip angular velocity range

def descrete_state(state):
    """
    Convert continuous state values into discrete bins.
    """
    state_descrete = []
    for i in range(len(state)):
        scaled = (state[i]-state_bounds[i][0]/state_bounds[i][1]-state_bounds[i][0])
        bin_index = int(scaled*state_bins[i])
        bin_index = min(max(bin_index, 0), state_bins[i]-1)  # Clip to valid bin range
        state_descrete.append(bin_index)
    return tuple(state_descrete)

# 4. Create Q-Table
q_table = np.zeros(state_bins+[env.action_space.n])  # Shape: (10, 10, 10, 10, 2)
