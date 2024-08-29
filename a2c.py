import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F
import gym
from gym import spaces
import numpy as np
from typing import NamedTuple
from copy import deepcopy

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import (
    get_obs_shape, get_action_dim
)

from wesutils import two_layer_net

from inforatio.utils import MeasureEstimator
import inforatio.models
from inforatio.buffers import RolloutBuffer


class A2CBase:
    def __init__(self,
                 env,
                 policy,
                 value_function,
                 policy_lr,
                 value_lr,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
                 gamma=0.99,
                 buffer_size=2048,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        self.env = env
        self.pi = policy
        self.v = value_function
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

        self.pi_optim = policy_optimizer(self.pi.parameters(),
                                         lr=policy_lr,
                                         weight_decay=weight_decay)
        self.v_optim = value_optimizer(self.v.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius

        self.rollout_buffer = RolloutBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
            gamma=gamma
        )

    @property
    def cuda_enabled(self):
        return self.__cuda_enabled

    def enable_cuda(self, enable_cuda=True, warn=True):
        """Enable or disable cuda and update models."""
        
        if warn:
            warnings.warn("Converting models between 'cpu' and 'cuda' after "
                          "initializing optimizers can give errors when using "
                          "optimizers other than SGD or Adam!")
        
        self.__cuda_enabled = enable_cuda
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.__cuda_enabled \
                else 'cpu')
        self.pi.to(self.device)
        self.v.to(self.device)
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """
        Load policy and value functions. Copy them to target functions.

        This method is for evaluation only. Use load_checkpoint to continue
        training.
        """
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])

        self.pi.eval()
        self.v.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
                'v_state_dict': self.v.state_dict(),
                'v_optimizer_state_dict': self.v_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])
        self.v_optim.load_state_dict(models['v_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
        else:
            self.pi.eval()
            self.v.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def collect_rollout(self, env, rollout_length):
        """
        Perform a rollout and fill the rollout buffer.
        """

        self._last_obs = env.reset()
        self._last_episode_start = np.zeros(1)
        n_steps = 0
        self.rollout_buffer.reset()

        while n_steps < rollout_length:

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device).float()
                action, log_prob = self.pi.sample(obs_tensor)
                value = self.v(obs_tensor)
            action = action.cpu().numpy()

            # Rescale and perform action
            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low,
                                         self.env.action_space.high)
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                clipped_action = int(clipped_action)

            new_obs, reward, done, info = env.step(clipped_action)

            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)

            self.rollout_buffer.add(self._last_obs, action, reward,
                                    self._last_episode_start, value, log_prob)
            self._last_obs = new_obs
            self._last_episode_start = done

            if done:
                env.reset()

        self.rollout_buffer.compute_returns_and_advantage(last_value=value,
                                                          done=done)

        return np.sum(self.rollout_buffer.rewards) / self.rollout_buffer.buffer_size

    def train(self):
        """
        Train on the current rollout buffer.
        """        

        for epoch in range(self.n_epochs):

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                obs = rollout_data.observations
                values = self.v(obs).flatten()
                log_probs = self.pi.log_probs(obs, actions)
                entropies = self.pi.entropy(obs)

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_loss = -(advantages * log_probs).mean() - \
                        self.entropy_coef * entropies.mean()

                self.pi_optim.zero_grad()
                policy_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                   self.grad_clip_radius)
                self.pi_optim.step()

                value_loss = F.mse_loss(rollout_data.returns, values)

                self.v_optim.zero_grad()
                value_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.v.parameters(),
                                                   self.grad_clip_radius)
                self.v_optim.step()

    def update(self, env, rollout_length):
        reward = self.collect_rollout(env, rollout_length)
        self.train()
        return reward, 1, 1


class A2CTwoLayer(A2CBase):
    def __init__(self,
                 env,
                 state_dim,
                 num_actions,
                 policy_lr,
                 value_lr,
                 pi_hidden_layer1_size=64,
                 pi_hidden_layer2_size=64,
                 v_hidden_layer1_size=64,
                 v_hidden_layer2_size=64,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
                 gamma=0.99,
                 buffer_size=2048,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        policy = inforatio.models.CategoricalPolicyTwoLayer(
            state_dim, num_actions,
            hidden_layer1_size=pi_hidden_layer1_size,
            hidden_layer2_size=pi_hidden_layer2_size)

        value_function = two_layer_net(
            state_dim, 1,
            hidden_layer1_size=v_hidden_layer1_size,
            hidden_layer2_size=v_hidden_layer2_size)

        super().__init__(env, policy, value_function, policy_lr, value_lr,
                         entropy_coef=entropy_coef,
                         n_epochs=n_epochs,
                         batch_size=batch_size, weight_decay=weight_decay,
                         gamma=gamma,
                         buffer_size=buffer_size, enable_cuda=enable_cuda,
                         policy_optimizer=policy_optimizer,
                         value_optimizer=value_optimizer,
                         grad_clip_radius=grad_clip_radius)


class A2CConvolutional(A2CBase):
    def __init__(self,
                 env,
                 num_actions,
                 policy_lr,
                 value_lr,
                 in_channels=1,
                 policy_out_channels1=16,
                 policy_kernel_size1=3,
                 policy_stride1=2,
                 policy_padding1=1,
                 policy_in_channels2=16,
                 policy_out_channels2=32,
                 policy_kernel_size2=2,
                 policy_stride2=1,
                 policy_padding2=1,
                 policy_out_features3=256,
                 value_out_channels1=16,
                 value_kernel_size1=3,
                 value_stride1=2,
                 value_padding1=1,
                 value_in_channels2=16,
                 value_out_channels2=32,
                 value_kernel_size2=2,
                 value_stride2=1,
                 value_padding2=1,
                 value_out_features3=256,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
                 gamma=0.99,
                 buffer_size=2048,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        policy = inforatio.models.CategoricalPolicyConvolutional(
            num_actions, torch.FloatTensor(env.reset()),
            in_channels=in_channels,
            out_channels1=policy_out_channels1,
            kernel_size1=policy_kernel_size1,
            stride1=policy_stride1,
            padding1=policy_padding1,
            in_channels2=policy_in_channels2,
            out_channels2=policy_out_channels2,
            kernel_size2=policy_kernel_size2,
            stride2=policy_stride2,
            padding2=policy_padding2,
            out_features3=policy_out_features3,
        )

        value_function = inforatio.models.ConvolutionalNetwork(
            in_channels, 1, # in_channels, out_dim
            torch.FloatTensor(env.reset()),
            out_channels1=value_out_channels1,
            kernel_size1=value_kernel_size1,
            stride1=value_stride1,
            padding1=value_padding1,
            in_channels2=value_in_channels2,
            out_channels2=value_out_channels2,
            kernel_size2=value_kernel_size2,
            stride2=value_stride2,
            padding2=value_padding2,
            out_features3=value_out_features3,
        )

        super().__init__(env, policy, value_function, policy_lr, value_lr,
                         entropy_coef=entropy_coef,
                         n_epochs=n_epochs,
                         batch_size=batch_size, weight_decay=weight_decay,
                         gamma=gamma,
                         buffer_size=buffer_size, enable_cuda=enable_cuda,
                         policy_optimizer=policy_optimizer,
                         value_optimizer=value_optimizer,
                         grad_clip_radius=grad_clip_radius)
