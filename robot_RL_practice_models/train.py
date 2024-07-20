'''DQN training
'''
import gym 
import torch
import random
import numpy as np
from tqdm import tqdm
from nn import BasicNN
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque

''' Things to alter if changing game:
        Line 20: Game-Name
        Line 148: How to quantify success
        Line 149: Name of model file
'''

# setting up environment
env = gym.make('MountainCar-v0', render_mode='human')
n_actions = env.action_space.n 
states, info = env.reset()
n_observations = len(states)


# hyperparamteres
batch_size = 512 # number of transitions sampled from replay buffer
gamma = 0.99 # discount factor
eps_start = 0.9 # starting value of epsilon
eps_end = 1e-4 # end value of epsilon
eps_decay = 100 # exponential decay rate of epsilon (higher is slower decay as -ve)
tau = 0.05 # update rate of target network
lr = 3e-4 # learning rate
no_episodes = 150 # how many episodes to run

# a list but with names for each position
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) 

class MemoryReplay(object):
    ''' To store previous episodes data in order to improve learning
    '''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # a list but can be modified faster

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# setting up policies, create two copies in order to prevent oscillation
policy_nn = BasicNN(n_observations, n_actions)
target_nn = BasicNN(n_observations, n_actions) 
target_nn.load_state_dict(policy_nn.state_dict()) # copies over parameters

optimizer = torch.optim.AdamW(policy_nn.parameters(), lr=lr, amsgrad=True) # chose optimiser
memory = MemoryReplay(100000)

no_steps = 0

def select_action(state):
    """ Function to select appropiate action following
    epsillon-greedy algorithm (sometimes the model will
    be used to choose the action, while at other times, 
    the action will be chosen uniformly at random)"""
    global no_steps
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(- no_steps / eps_decay)
    no_steps += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1).indices.view(1,1) # finds highest q_value
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long) # random action
    
def train_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # take transpose in order to use batch-array of transitions
    batch = Transition(*zip(*transitions))

    # makes 1D array of all non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q-values
    state_action_values = policy_nn(state_batch).gather(1, action_batch)

    # copmute V-values using target_nn
    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_nn(non_final_next_states).max(1)[0]
    # compute the expected Q-value
    expected_state_action_values = (next_state_values * gamma) + reward_batch 

    # use hubber loss instead of MSE
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimise and backward
    optimizer.zero_grad()
    loss.backward()
    # gradient-clipping to prevent huge gradients
    torch.nn.utils.clip_grad_value_(policy_nn.parameters(), 100)
    optimizer.step()

"""Training model
"""
eps_lengths = [] # to keep track of length of episode (or how well the model did)
for i_episode in tqdm(range(no_episodes)):
    # initialise envornment and get state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    for t in count():
        action = select_action(state)
        obs, rew, terminated, truncated, _ = env.step(action.item())
        rew = torch.tensor([rew])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # store transition in memory and move to next state
        memory.push(state, action, next_state, rew)
        state = next_state

        # perform one step of optimisation and update target wheights
        train_model()
        target_nn_state_dict = target_nn.state_dict()
        policy_nn_state_dict = policy_nn.state_dict()
        for key in policy_nn_state_dict:
            target_nn_state_dict[key] = policy_nn_state_dict[key] * tau + target_nn_state_dict[key] * (1-tau)
        target_nn.load_state_dict(target_nn_state_dict) # update wheights 

        if done:
            '''Saves the model'''
            # if len(eps_lengths) > 1:
            #     if t+1 <= min(eps_lengths):
            #         torch.save(policy_nn, 'src/robot_RL/models/taxi_dqn.pth')
            eps_lengths.append(t+1)
            break

