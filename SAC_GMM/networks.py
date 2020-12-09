import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from Soft_Actor_Critic.networks import PolicyNetwork as ActorNetwork
from Soft_Actor_Critic.networks import SoftQNetwork as CriticNetwork

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, gmm_dim, action_dim, encode_dim=8, *args):
        super(PolicyNetwork, self).__init__()
        self.encode_gmm = nn.Linear(gmm_dim, encode_dim)
        self.actor = ActorNetwork(state_dim + encode_dim, action_dim, *args)

    def forward(self, state, gmm):
        encoded_gmm = F.relu(self.encode_gmm(gmm))
        x = torch.cat((state, encoded_gmm), dim=-1)
        return self.actor(x)

    def getActions(self, state, gmm, *args, **kwargs):
        encoded_gmm = F.relu(self.encode_gmm(gmm))
        x = torch.cat((state, encoded_gmm), dim=-1)
        return self.actor.getActions(x, *args, **kwargs) 

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, gmm_dim, action_dim, encode_dim=8, *args):
        super(SoftQNetwork, self).__init__()
        self.encode_gmm = nn.Linear(gmm_dim, encode_dim)
        self.encode_action = nn.Linear(action_dim, encode_dim)
        self.critic = CriticNetwork(state_dim + encode_dim, encode_dim, *args)

    def forward(self, state, gmm, action):
        encoded_gmm = F.relu(self.encode_gmm(gmm))
        encoded_action = F.relu(self.encode_action(action))
        x = torch.cat((state, encoded_gmm), dim=-1)
        return self.critic(x, encoded_action)
