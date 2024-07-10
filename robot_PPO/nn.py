import torch
from torch import nn
import numpy as np

class BasicNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(BasicNN, self).__init__()

        #Set up neural network
        self.model_layout = torch.nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Softmax()       
        )
    
    def forward(self, obs):
        # Converts data to tensor
        if torch.is_tensor(obs) == False:
            obs = torch.tensor(obs, dtype=torch.float)
        return self.model_layout(obs)

        