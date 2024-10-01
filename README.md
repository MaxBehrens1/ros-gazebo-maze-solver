# ros-gazebo-maze-solver

## Introduction

This repository, **ros-gazebo-maze-solver**, showcases a complete solution for controlling a differential drive robot using ROS 2 Humble. I completed this project as part of my summer internship at PHINXT robotics. The robot is equipped with a LIDAR sensor and can autonomously solve a complex maze. Two different strategies are implemented:

- **Left-hand rule**: A simple maze-solving technique based on following the left wall.
- **Reinforcement Learning (DQN)**: A Deep Q-Network algorithm that is trained to navigate the maze through trial and error, using experience and rewards to improve performance.

The project is split into three main directories, each representing different components of the system:

1. **robot_RL_practice_models**
2. **robot_controller**
3. **robot_maze_solver**

## 1. robot_RL_practice_models

This directory contains the foundational work where I developed and trained a Deep Q-Network (DQN) algorithm. Here, I used the **OpenAI Gym** environment to train the robot on various games and scenarios, building the reinforcement learning model from the ground up.

- **Neural Network Architecture**: The DQN consists of a two-layer fully connected neural network.
- **Dual Policy Method**: A dual policy approach was used to improve the performance and stability of the models.
- **Achievements**: Successfully trained models that could solve a variety of tasks provided by OpenAI Gym, serving as the baseline for future implementations in the robot controller.

## 2. robot_controller

The **robot_controller** directory houses all the necessary configurations and control logic for the ROS 2 Humble system. Here, two key approaches for solving the maze are implemented:

- **Left-hand rule algorithm**: This traditional maze-solving technique follows the left wall to navigate the maze. It's implemented as a simple yet effective method for guiding the robot.
  
- **Reinforcement Learning (ROS 2 environment)**: I implemented a similar reinforcement learning algorithm as developed in the `robot_RL_practice_models`. However, in this case, it is applied within a custom ROS 2 environment that I created, allowing the robot to learn and adapt in real-time to the maze environment.

The `robot_controller` folder contains:
- ROS 2 node configurations and scripts to manage the robot's movements.
- Reinforcement learning files adapted to ROS 2 for training and inference in the maze.

## 3. robot_maze_solver

The **robot_maze_solver** directory is the Gazebo simulation side of the project. It includes all necessary simulation files for the robot and its environment:

- **Xacro files**: These define the robot's structure and sensor configurations.
- **World files**: These describe the maze environment in which the robot will operate.
- **Launch files**: These files are used to start the simulation and coordinate the robot's interactions within Gazebo.

This section brings everything together by providing a simulated environment in which the robot can solve the maze using either the left-hand rule or reinforcement learning strategies.

## Future Improvements

- [ ] Integrate additional maze-solving algorithms such as A* or Dijkstra's algorithm.
- [ ] Add more sensor feedback mechanisms (e.g., camera, IMU) to enhance robot perception and decision-making
- [ ] Explore multi-agent systems where multiple robots collaborate to solve the maze.
- [ ] Implement real-world deployment using hardware robots instead of simulation.

## Acknowledgments

While this project has been a fun and educational experience, it's worth acknowledging that training a robot to solve any arbitrary maze is a significant challenge. In reality, the robot has learned a solution specific to the maze I created, rather than a general-purpose maze-solving algorithm. Despite this, I gained valuable insights into reinforcement learning, machine learning, and robotics throughout the process, making this a very rewarding project to undertake.

A big thanks to **PHINXT Robotics** for their support and guidance throughout the development of this project!


