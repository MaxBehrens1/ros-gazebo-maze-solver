import time
import rclpy
import numpy as np
from copy import deepcopy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from dqn_sim_funcs import launch, reset, obs, end_sim

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

# set up nodes and simulation
rclpy.init(args=None)
subscriber = odom_laser_sub()
env = launch()
action_space = [0, 1, 2, 3] # possible actions [forward, backward, left, right]

time.sleep(7)
env, observation = reset(env, subscriber) # resets the environment

print(observation)


def step(action):
    pass

# at end of training
end_sim(env)
rclpy.shutdown()





