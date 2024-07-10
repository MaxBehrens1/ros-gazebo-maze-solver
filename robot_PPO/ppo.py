from torch.optim import Adam
from nn import BasicNN
import torch.nn as nn
import numpy as np
import torch

class PPO:

    def __init__(self, env):
        #To set hyperparameters
        self._hyper_parameters()

        #Get environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #Setup actor and critic netowrks
        self.actor = BasicNN(self.obs_dim, self.act_dim)
        self.critic = BasicNN(self.obs_dim, 1)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
    def learn(self, total_time):
        current_time = 0
        while current_time < total_time:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate V
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            # Calculate advantage Q - V
            A_k = batch_rtgs - V.detach()

            # Normalize to increase stability
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  

            for _ in range(self.n_updates_per_iteration):
                # Start of epoch loop
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios from PPO equation
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                #Actor and critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                #Perform back propogation
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()
            current_time += np.sum(batch_lens)

    def _hyper_parameters(self):
        self.timesteps_per_batch = 4800            # timesteps per batch (multiple episodes)
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 
        self.lr = 0.005

    def get_action(self, obs):
        #Sample an action based on probability (not necesserily the biggest prob)
        action_probs = self.actor(obs)
        p = action_probs.cumsum(0)
        idx = torch.searchsorted(p, torch.rand(1))
        action = idx.item()
        log_prob = torch.log(action_probs[idx])
        return action, log_prob
    
    def compute_rtgs(self, batch_rws):  # same as Q-values
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rws):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the discounted rewards into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts): 
        # Run critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze() # inputing multiple episodes of observations

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        action_probs = self.actor(batch_obs)
        log_prob = torch.log(action_probs[batch_acts])
        return V, log_prob

    def rollout(self):
        # Creting empty arrays for data
        batch_obs = []             # list of observations from all epsiode
        batch_acts = []            # list of actions from all episodes
        batch_log_probs = []       # log probs of each action throughout all actions
        batch_rews = []            # 2D list of rewards from each episode 
        batch_rtgs = []            # 2D list of discounted rewards from each episode
        batch_lens = []            # length of each episode
        t = 0                      # number of timesteps so far this batch

        while t < self.timesteps_per_batch:
            rews_ep = [] # rewards this episode
            obs = self.env.reset() # resets the environment
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 # increment time_step

                batch_obs.append(obs) # add observation (including at t=0)

                # collect data
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
                
                # add other data
                batch_acts.append(action)
                rews_ep.append(rew)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # collect episode data
            batch_rews.append(rews_ep)
            batch_lens.append(ep_t + 1) # as time starts at 0

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Converts from total rewards to discounted reward (Q-value)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    


