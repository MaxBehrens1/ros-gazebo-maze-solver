'''Baseline for PPO algorithm, concluded that DQN may be easier to implement
'''
import gym
import torch

policy_nn = torch.load('src/robot_RL_practice_models/models/mountaincar_dqn.pth')
env = gym.make("MountainCar-v0", render_mode="human")

def select_action(state):
    with torch.no_grad():
            return policy_nn(state).max(1).indices.view(1,1)

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

for _ in range(10000):
    action = select_action(state)
    obs, rew, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    if terminated or truncated:
        env.close()
        break



