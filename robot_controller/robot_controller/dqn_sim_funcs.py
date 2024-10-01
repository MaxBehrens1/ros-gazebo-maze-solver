import rclpy
import time
import subprocess
from signal import SIGINT

def launch():
    sim = subprocess.Popen(["ros2", "launch", "robot_maze_solver", "launch_sim.launch.py", "world:=phinxt_ws/src/robot_maze_solver/worlds/complex_maze.world"],
                            stdout=subprocess.PIPE, cwd='/home/max/')
    return sim

# to take observations reading
def observe(sub):
    for _ in range(2):
        rclpy.spin_once(sub)
    return sub.observation()
    
# to reset simulation
def reset(simulation, sub):
    ''' To reset simulation
    Returns: env, obs
    '''
    end_sim(simulation)
    env = launch()
    time.sleep(10)
    observation = observe(sub)
    return observation, env

def end_sim(sim):
    sim.send_signal(SIGINT)
    sim.wait(timeout=30)

def step(action, pub, sub):
    '''Function to take 1 step in the simulation
    '''
    if action == 0:
        pub.forward()
    elif action == 1:
        pub.backward()
    elif action == 2:
        pub.left()
    elif action == 3:
        pub.right()
    obs = observe(sub)
    if obs[0] < 7:
        rew = 1
        terminated = False
    else:
        rew = 0
        terminated = True
    truncated = False
    return obs, rew, terminated, truncated

