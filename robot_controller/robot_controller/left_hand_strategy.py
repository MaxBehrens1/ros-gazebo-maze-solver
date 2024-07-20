import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np

class left_hand_strategy(Node):

    def __init__(self):
            super().__init__('left_hand_strategy')
            self.get_logger().info('New node running: left_hand_strategy')
            self.laser_subscriber = self.create_subscription(
                LaserScan, '/scan', self.laser_callback, 10)
            self.odom_subscriber = self.create_subscription(
                Odometry, '/odom', self.odom_callback, 10)
            self.cmd_vel_publisher = self.create_publisher(
                Twist, '/cmd_vel', 10) 
            self.dist_front = True
            self.dist_left = [True, 0.0]
            self.dist_right = [True, 0.0]
            self.start_forward = True
            self.start_distance = True
            self.start_turn = True
            self.finish_turn = False
            self.odometry = []
            self.final_pos = []
            self.final_bearing = 0.0

            
    def odom_callback(self, odom: Odometry):
        current_bearing = np.arctan2(2 * odom.pose.pose.orientation.w * odom.pose.pose.orientation.z,
                                      1 - 2 * odom.pose.pose.orientation.z * odom.pose.pose.orientation.z)
        #Make sure bearing is between 0 and 2pi
        if current_bearing < 0:
            current_bearing += 2 * np.pi
        if current_bearing > 2* np.pi:
            current_bearing -= 2*np.pi
        self.odometry = [odom.pose.pose.position.x, odom.pose.pose.position.y, current_bearing]
        
    def move_forward(self, pos):
        msg = Twist()
        if np.isclose(self.odometry[2], 0, atol=0.5) or np.isclose(self.odometry[2], 2*np.pi, atol=0.5) or np.isclose(self.odometry[2], np.pi, atol=0.5):
            if np.isclose(self.odometry[0], pos[0], atol = 0.05) == False:
                msg.linear.x = 0.2
                self.cmd_vel_publisher.publish(msg)
            else:
                msg.linear.x = 0.0
                self.cmd_vel_publisher.publish(msg)
                self.start_distance = True
        else:
            if np.isclose(self.odometry[1], pos[1], atol=0.05) == False:
                msg.linear.x = 0.2
                self.cmd_vel_publisher.publish(msg)
            else:
                msg.linear.x = 0.0
                self.cmd_vel_publisher.publish(msg)
                self.start_distance = True

    def turn(self, angle, direction = 1):
        msg = Twist()
        current_bearing = self.odometry[2]
        if np.isclose(current_bearing, angle, atol=0.01)== False and np.isclose(current_bearing, angle + 2*np.pi, atol=0.005) == False and np.isclose(current_bearing, angle - 2*np.pi, atol=0.005) == False:
            msg.angular.z = direction * 0.02    
            self.cmd_vel_publisher.publish(msg)
            self.finish_turn = False
        else:  
            msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(msg)
            self.finish_turn = True
            
    def laser_callback(self, laser: LaserScan):
        msg = Twist()
        if self.start_distance == True:
            dist_front = laser.ranges[180]
            dist_left = [laser.ranges[260], laser.ranges[270], laser.ranges[280]]
            dist_right = [laser.ranges[80], laser.ranges[90], laser.ranges[100]]
            self.start_distance = False
            self.start_forward = True
            self.start_turn = True

            #Check diagonal distance to make sure its not at the end of a wall
            #True means distance is bigger than 1.5 in that direction
            if dist_front > 1.5:
                self.dist_front = True
            else:
                self.dist_front = False
            if dist_left[0] > 1.7 and dist_left[2] > 1.6 and dist_left[1] > 1.5:
                self.dist_left = [True, dist_left[1]]
            else:
                self.dist_left = [False, dist_left[1]]
            if dist_right[0] > 1.7 and dist_right[2] > 1.6 and dist_right[1] > 1.5:
                self.dist_right = [True, dist_right[1]]
            else:
                self.dist_right = [False, dist_right[1]]
            self.get_logger().info(f'Left:{self.dist_left}, Right:{self.dist_right}, Front:{self.dist_front}')

        #Going along wall
        if self.dist_front == True and self.dist_left[0] == False:
            self.get_logger().info(f'Going straight. Left:{self.dist_left}, Right:{self.dist_right}')
            if self.start_forward == True:  #to find one coordinate forwards
                self.final_pos = np.array([np.round(self.odometry[0]), np.round(self.odometry[1])])
                if np.isclose(self.odometry[2], 0, atol=0.5) or np.isclose(self.odometry[2], 2*np.pi, atol=0.5):
                    self.final_pos += np.array([1, 0])
                elif np.isclose(self.odometry[2], np.pi/2, atol=0.5):
                    self.final_pos += np.array([0, 1])
                elif np.isclose(self.odometry[2], np.pi, atol=0.5):
                    self.final_pos += np.array([-1, 0])
                else:
                    self.final_pos += np.array([0,-1])
                self.start_forward = False
            self.move_forward(self.final_pos)
        
        # At left turning
        elif self.dist_left[0] == True and self.dist_left[1] > 2.5: # a space to the left
            self.get_logger().info('Turning Left')
            if self.start_turn == True: #to find final bearing
                self.final_bearing = np.round(self.odometry[2]/(np.pi/2))*(np.pi/2) + np.pi/2
                if self.final_bearing > 2*np.pi:
                    self.final_bearing -= 2*np.pi
                self.start_turn = False
                self.start_forward = True
            self.turn(self.final_bearing, 1)
            
            if self.finish_turn:
                if self.start_forward == True:  #to find two coordinates forwards
                    self.final_pos = np.array([np.round(self.odometry[0]), np.round(self.odometry[1])])
                    if np.isclose(self.odometry[2], 0, atol=0.5) or np.isclose(self.odometry[2], 2*np.pi, atol=0.5):
                        self.final_pos += np.array([2, 0])
                    elif np.isclose(self.odometry[2], np.pi/2, atol=0.5):
                        self.final_pos += np.array([0, 2])
                    elif np.isclose(self.odometry[2], np.pi, atol=0.5):
                        self.final_pos += np.array([-2, 0])
                    else:
                        self.final_pos += np.array([0,-2])
                    self.start_forward = False
                self.move_forward(self.final_pos)
        
        #At a dead end
        elif self.dist_front == False and self.dist_left[0] == False and self.dist_right[0] == False:
            self.get_logger().info(f'Dead end. Left:{self.dist_left}, Right:{self.dist_right}')
            if self.start_turn == True: #to find final bearing
                self.final_bearing = np.round(self.odometry[2]/(np.pi/2))*(np.pi/2) + np.pi
                if self.final_bearing > 2*np.pi:
                    self.final_bearing -= 2*np.pi
                self.start_turn = False
                self.start_forward = True
            self.turn(self.final_bearing)

            if self.finish_turn:
                self.start_distance = True
        
        #Turning right
        elif self.dist_right[0] == True and self.dist_left[1] < 2.5 and self.dist_front == False:
            self.get_logger().info('Turning Right')
            if self.start_turn == True: #to find final bearing
                self.final_bearing = np.round(self.odometry[2]/(np.pi/2))*(np.pi/2) - np.pi/2
                if self.final_bearing < 0:
                    self.final_bearing += 2*np.pi
                self.start_turn = False
                self.start_forward = True
            self.turn(self.final_bearing, -1)
            
            if self.finish_turn:
                if self.start_forward == True:  #to find two coordinates forwards
                    self.final_pos = np.array([np.round(self.odometry[0]), np.round(self.odometry[1])])
                    if np.isclose(self.odometry[2], 0, atol=0.5) or np.isclose(self.odometry[2], 2*np.pi, atol=0.5):
                        self.final_pos += np.array([2, 0])
                    elif np.isclose(self.odometry[2], np.pi/2, atol=0.5):
                        self.final_pos += np.array([0, 2])
                    elif np.isclose(self.odometry[2], np.pi, atol=0.5):
                        self.final_pos += np.array([-2, 0])
                    else:
                        self.final_pos += np.array([0,-2])
                    self.start_forward = False
                self.move_forward(self.final_pos)

def main(args=None):
    rclpy.init(args=args)
    node = left_hand_strategy()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()