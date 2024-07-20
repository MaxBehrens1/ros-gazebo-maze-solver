import launch
import launch_ros.actions
import os
import time

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='robot_controller',
            executable='left_hand_strategy',
            name='left_hand_strategy'
        )
    ])

