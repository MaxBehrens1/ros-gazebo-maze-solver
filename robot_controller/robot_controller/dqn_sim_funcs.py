import rclpy
import time
import subprocess
from signal import SIGINT

def launch():
    sim = subprocess.Popen(["ros2", "launch", "robot_maze_solver", "launch_sim.launch.py", "world:=phinxt_ws/src/robot_maze_solver/worlds/complex_maze.world"],
                            stdout=subprocess.PIPE, cwd='/home/max/')
    return sim

# to take observations reading
def obs(sub):
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
    time.sleep(1)
    observation = obs(sub)
    return env, observation



def end_sim(sim):
    sim.send_signal(SIGINT)
    sim.wait(timeout=30)

