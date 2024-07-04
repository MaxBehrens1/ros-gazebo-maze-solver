import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

#All bearings have the x-axis as N, and go anti-clockwise

class basic_closed_loop(Node):

    open('src/robot_controller/data/position_data.txt', 'w').close() #to clear file contents
    positions = open('src/robot_controller/data/position_data.txt', 'a')
    coordinates = np.array([[5,0], [5,-3], [-3,-3], [-3,3], [5,3]])

    def __init__(self):
        super().__init__('basic_closed_loop')
        self.get_logger().info('New node running: basic_closed_loop')
        self.pose_subscriber = self.create_subscription(
            Odometry, '/odom', self.pose_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10) 
        self.i = 0
        self.correct_bearing = False
               

    def pose_callback(self, odom: Odometry):
        cmd = Twist()
        final_pos = self.coordinates[self.i]
        rel_vec_to_pos = np.array([final_pos[0] - odom.pose.pose.position.x, final_pos[1] - odom.pose.pose.position.y])
        dist_to_final_pos = np.sqrt(np.dot(rel_vec_to_pos, rel_vec_to_pos))
        current_bearing = np.arctan2(2 * odom.pose.pose.orientation.w * odom.pose.pose.orientation.z,
                                      1 - 2 * odom.pose.pose.orientation.z * odom.pose.pose.orientation.z)

        if rel_vec_to_pos[1] > 0:
            bearing_to_final_pos = np.arccos(np.dot([1,0], rel_vec_to_pos / dist_to_final_pos))
        else:
            bearing_to_final_pos = 2*np.pi - np.arccos(np.dot([1,0], rel_vec_to_pos / dist_to_final_pos)) 
        if current_bearing < 0:
            current_bearing += 2 * np.pi

        if np.isclose(current_bearing, bearing_to_final_pos, atol=0.001) == False and self.correct_bearing == False:
            cmd.angular.z = 0.02
            cmd.linear.x = 0.0
        elif np.isclose(dist_to_final_pos, 0, atol=0.05) == False:
            self.correct_bearing = True
            cmd.angular.z = 0.0
            cmd.linear.x = 0.1
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.correct_bearing = False
            if self.i < len(self.coordinates)-1:
                self.i += 1

        self.positions.write(f'{odom.pose.pose.position.x}, {odom.pose.pose.position.y} \n')
        self.cmd_vel_publisher.publish(cmd)
        
        
def main(args=None):
    rclpy.init(args=args)
    node = basic_closed_loop()
    rclpy.spin(node)
    node.positions.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()