import time
import torch
import rclpy
import random
import numpy as np
from torch import nn 
from tqdm import tqdm
from itertools import count
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from collections import namedtuple, deque
from dqn_sim_funcs import launch, reset, end_sim, step

class action_publisher(Node):
    '''Action node, which performs given actions for 1 second
    '''
    def __init__(self):
        super().__init__('action_publisher')
        self.cmd_vel_publisher = self.create_publisher(
                Twist, '/cmd_vel', 10)
        
    def forward(self):
        msg = Twist()
        msg.linear.x = 0.5
        self.cmd_vel_publisher.publish(msg)
        time.sleep(1)
        msg.linear.x = 0.0
        self.cmd_vel_publisher.publish(msg)
    
    def backward(self):
        msg = Twist()
        msg.linear.x = -0.5
        self.cmd_vel_publisher.publish(msg)
        time.sleep(1)
        msg.linear.x = 0.0
        self.cmd_vel_publisher.publish(msg)
    
    def left(self):
        msg = Twist()
        msg.angular.z = 0.2
        self.cmd_vel_publisher.publish(msg)
        time.sleep(1)
        msg.angular.x = 0.0
        self.cmd_vel_publisher.publish(msg)
    
    def right(self):
        msg = Twist()
        msg.angular.z = -0.2
        self.cmd_vel_publisher.publish(msg)
        time.sleep(1)
        msg.angular.x = 0.0
        self.cmd_vel_publisher.publish(msg)

class odom_laser_sub(Node):
    '''Subscriber for observation
    '''
    def __init__(self):
        super().__init__('odom_laser_sub')
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom = []
        self.laser = []

    def odom_callback(self, odom: Odometry,):
        current_bearing = np.arctan2(2 * odom.pose.pose.orientation.w * odom.pose.pose.orientation.z,
                                      1 - 2 * odom.pose.pose.orientation.z * odom.pose.pose.orientation.z)
        #Make sure bearing is between 0 and 2pi
        if current_bearing < 0:
            current_bearing += 2 * np.pi
        if current_bearing > 2* np.pi:
            current_bearing -= 2*np.pi
        self.odom = [odom.pose.pose.position.x, odom.pose.pose.position.y, current_bearing]
    
    def laser_callback(self, laser: LaserScan):
        for i in [270, 170, 90]:
            self.laser.append(laser.ranges[i])
    
    def observation(self):
        obs = []
        for i in self.odom:
            obs.append(i)
        for j in self.laser:
            obs.append(j)
        return obs

class BasicNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(BasicNN, self).__init__()

        #Set up neural network
        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)
    
    def forward(self, obs):
        obs = nn.functional.relu(self.layer1(obs))
        obs = nn.functional.relu(self.layer2(obs))
        return self.layer3(obs)

# set up nodes and simulation
rclpy.init(args=None)
subscriber = odom_laser_sub()
publisher = action_publisher()
env = launch()
action_space = [0, 1, 2, 3] # possible actions [forward, backward, left, right]
n_actions = len(action_space)
states, info = reset(env, subscriber)
n_observations = len(states)

# hyperparameters
batch_size = 1024 # number of transitions sampled from replay buffer
gamma = 0.99 # discount factor
eps_start = 0.99 # starting value of epsilon
eps_end = 1e-45 # end value of epsilon
eps_decay = 150 # exponential decay rate of epsilon (higher is slower decay as -ve)
tau = 0.05 # update rate of target network
lr = 7e-4 # learning rate
no_episodes = 100 # how many episodes to run

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
    state, info = reset(env, subscriber)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    for t in count():
        action = select_action(state)
        obs, rew, terminated, truncated, _ = env.step(action.item(), publisher, subscriber)
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


# at end of training
end_sim(env)
rclpy.shutdown()





