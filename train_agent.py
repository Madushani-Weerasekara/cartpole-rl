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
 
 


