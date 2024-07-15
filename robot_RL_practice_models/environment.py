import gym
import rclpy
import numpy as np
from gym import spaces
from rclpy.node import Node
from nav_msgs.msg import Odometry
from robot_controller.robot_controller.odom_laser_sub import odom_laser_sub

def main(args=None):
    rclpy.init(args=args)
    node = odom_laser_sub()
    print(node.laser_data)
    # rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

