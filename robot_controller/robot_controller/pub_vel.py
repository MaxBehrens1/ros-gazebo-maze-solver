#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class pub_vel(Node):

    def __init__(self):
        super().__init__('pub_vel')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('New node running: vel_pub')
        self.timer = self.create_timer(1, self.send_vel)
        self.i = 0
    
    def send_vel(self):
        msg = Twist()
        # msg.linear.x = 0.05
        msg.angular.z = 0.05
        self.cmd_vel_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = pub_vel()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()