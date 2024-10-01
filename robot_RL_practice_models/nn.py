from torch import nn

class BasicNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(BasicNN, self).__init__()

        #Set up neural network
        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)
    
    def forward(self, obs):
        obs = nn.functional.relu(self.layer1(obs))
        obs = nn.functional.relu(self.layer2(obs))
        return self.layer3(obs)
    


        