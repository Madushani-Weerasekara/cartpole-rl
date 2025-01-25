import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the nueral network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.f2(x))
        return self.fc3(x)

    # Hyperparameters
    EPISODES = 100
    GAMMA = 0.1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPSILON = 1.0 # Initial exploration rate
    EPSILON_DECAY =0.995
    EPSILON_MIN =0.01
    MEMORY_SIZE = 100000

