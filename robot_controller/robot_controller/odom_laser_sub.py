import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

input_data = []

class odom_laser_sub(Node):

    def __init__(self):
        super().__init__('pub_vel')
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.data = []
    
    def odom_callback(self, odom: Odometry,):
        current_bearing = np.arctan2(2 * odom.pose.pose.orientation.w * odom.pose.pose.orientation.z,
                                      1 - 2 * odom.pose.pose.orientation.z * odom.pose.pose.orientation.z)
        #Make sure bearing is between 0 and 2pi
        if current_bearing < 0:
            current_bearing += 2 * np.pi
        if current_bearing > 2* np.pi:
            current_bearing -= 2*np.pi
        self.data = [odom.pose.pose.position.x, odom.pose.pose.position.y, current_bearing]
        
    
    def laser_callback(self, laser: LaserScan):
        for i in [270, 170, 90]:
            self.data.append(laser.ranges[i])
        input_data = self.data
        print(input_data)


def main(args=None):
    rclpy.init(args=args)
    node = odom_laser_sub()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()