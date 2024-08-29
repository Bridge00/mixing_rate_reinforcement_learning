import torch
import numpy as np
from typing import NamedTuple
from gym import spaces
from utils import MeasureEstimator
from stable_baselines3.common.preprocessing import (
    get_obs_shape, get_action_dim
)


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    
    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 gamma=0.99,
                 device='cpu'):
        
        self.buffer_size = 100#buffer_size
        #print('buffersize',self.buffer_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = device
        self.obs_shape = (4,)#get_obs_shape(self.observation_space)
        #print(self.obs_shape)
        self.action_dim = 5#(5,)#get_action_dim(self.action_space)
        
        self.reset()

    def change_buffer_size(self, new_size):
        self.buffer_size = new_size
        
    def reset(self):
        #print((self.buffer_size,), self.obs_shape)
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        
        self.full = False
        self.pos = 0
        
    def compute_returns_and_advantage(self, last_value, done):
        
        last_value = last_value.clone().cpu().numpy().flatten()
        
        discounted_reward = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_value = self.values[step + 1]
            discounted_reward = self.rewards[step] + \
                self.gamma * discounted_reward * next_non_terminal
            self.advantages[step] = discounted_reward - self.values[step]
        self.returns = self.advantages + self.values
        
    def add(self, obs, action, reward, episode_start, value, log_prob):
        
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((1,) + self.obs_shape)

        # import pdb; pdb.set_trace()
        # print(self.pos)
        # print(self.observations)
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def get(self, batch_size=None):
        print('full',self.full)
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)
