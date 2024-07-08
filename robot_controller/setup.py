from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxsbehrens@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pub_vel = robot_controller.pub_vel:main',
            'basic_closed_loop = robot_controller.basic_closed_loop:main',
            'lidar_control = robot_controller.lidar_control:main',
            'left_hand_strategy = robot_controller.left_hand_strategy:main'
        ],
    },
)