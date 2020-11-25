from collections import namedtuple
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_capacity=1e6): 
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.max_capacity = max_capacity
    
    def current_capacity(self):
        return len(self._data.states)

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

        if len(self._data.states) > self.max_capacity:
            del self._data.states[0]
            del self._data.actions[0]
            del self._data.next_states[0]
            del self._data.rewards[0]
            del self._data.dones[0]

    def next_batch(self, batch_size, tensor=False):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        
        if tensor:        #Map to tensor
            batch_states = torch.tensor(batch_states, dtype=torch.float, device="cuda") #B,S_D
            batch_actions = torch.tensor(batch_actions, dtype=torch.float, device="cuda") #B,A_D
            batch_next_states = torch.tensor(batch_next_states, dtype=torch.float, device="cuda", requires_grad=False) #B,S_D
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device="cuda", requires_grad=False).unsqueeze(-1) #B,1
            batch_dones = torch.tensor(batch_dones, dtype=torch.uint8, device="cuda", requires_grad=False).unsqueeze(-1) #B,1
      
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
