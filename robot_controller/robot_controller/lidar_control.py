import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np

class lidar_control(Node):

    open('src/robot_controller/data/position_data.txt', 'w').close() #to clear file contents
    positions = open('src/robot_controller/data/position_data.txt', 'a')

    def __init__(self):
        super().__init__('lidar_control')
        self.get_logger().info('New node running: lidar_control')
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10) 
        self.odometry = []
        self.turning = False
        self.final_bearing = 0.0

    def laser_callback(self, laser: LaserScan):
        msg = Twist()
        dist_front = laser.ranges[180]
        dist_left = laser.ranges[270]
        dist_right = laser.ranges[90]
        if dist_front > 0.7 and self.turning == False:  # moving forwards
            msg.linear.x = 0.2
            msg.angular.z = 0.00
        else:  
            if self.turning == False:
                if dist_left > dist_right:
                    self.final_bearing = self.odometry[2] + np.pi/2
                else:
                    self.final_bearing = self.odometry[2] + 3*np.pi/2
                if self.final_bearing > 2*np.pi:
                    self.final_bearing -= 2*np.pi
                self.turning = True
            if np.isclose(self.odometry[2], self.final_bearing, atol=0.005) == False:
                msg.angular.z = -0.02
                msg.linear.x = 0.0
            else:
                msg.angular.z = 0.0
                msg.linear.x = 0.0
                self.turning = False
                self.final_bearing = 0.0
        self.cmd_vel_publisher.publish(msg)
    
    def odom_callback(self, odom: Odometry):
        current_bearing = np.arctan2(2 * odom.pose.pose.orientation.w * odom.pose.pose.orientation.z,
                                      1 - 2 * odom.pose.pose.orientation.z * odom.pose.pose.orientation.z)
        if current_bearing < 0:
            current_bearing += 2 * np.pi
        self.odometry = [odom.pose.pose.position.x, odom.pose.pose.position.y, current_bearing]
        self.positions.write(f'{self.odometry[0]}, {self.odometry[1]} \n')

def main(args=None):
    rclpy.init(args=args)
    node = lidar_control()
    rclpy.spin(node)
    node.positions.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()