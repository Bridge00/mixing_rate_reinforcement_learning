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


class InfoRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_cost_values: torch.Tensor
    old_shadow_values: torch.Tensor
    old_log_prob: torch.Tensor
    cost_advantages: torch.Tensor
    shadow_advantages: torch.Tensor
    cost_returns: torch.Tensor
    shadow_returns: torch.Tensor


class DiscountedInfoRolloutBuffer:
    
    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 gamma=0.99,
                 device='cpu'):
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = device
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        
        self.reset()
        
    def reset(self):
        
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32
        )
        self.costs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_rewards = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.cost_values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.cost_advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        
        self.full = False
        self.pos = 0
        
    def compute_returns_and_advantage(
        self, last_cost_value, last_shadow_value, done
    ):
        
        last_cost_value = last_cost_value.clone().cpu().numpy().flatten()
        last_shadow_value = last_shadow_value.clone().cpu().numpy().flatten()

        d = MeasureEstimator()
        for obs in self.observations:
            d.append(obs)
        
        discounted_cost = 0
        discounted_shadow_reward = 0
        for step in reversed(range(self.buffer_size)):

            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_cost_value = last_cost_value
                next_shadow_value = last_shadow_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_cost_value = self.cost_values[step + 1]
                next_shadow_value = self.shadow_values[step + 1]

            self.shadow_rewards[step] = -np.log(d(self.observations[step]))

            discounted_cost = self.costs[step] + \
                self.gamma * discounted_cost * next_non_terminal
            discounted_shadow_reward = self.shadow_rewards[step] + \
                self.gamma * discounted_shadow_reward * next_non_terminal

            self.cost_advantages[step] = discounted_cost - self.cost_values[step]
            self.shadow_advantages[step] = discounted_shadow_reward - \
                    self.shadow_values[step]

        self.cost_returns = self.cost_advantages + self.cost_values
        self.shadow_returns = self.shadow_advantages + self.shadow_values
        
    def add(self, obs, action, cost, episode_start,
            cost_value, shadow_value, log_prob):
        
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((1,) + self.obs_shape)
            
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.costs[self.pos] = np.array(cost).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.shadow_values[self.pos] = shadow_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def get(self, batch_size=None):
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
            self.cost_values[batch_inds].flatten(),
            self.shadow_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.shadow_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.shadow_returns[batch_inds].flatten(),
        )
        return InfoRolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


class InfoRolloutBuffer:
    
    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 device='cpu'):
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        
        self.reset()
        
    def reset(self):
        
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32
        )
        self.costs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_rewards = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.cost_values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.cost_advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.shadow_advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        
        self.full = False
        self.pos = 0
        
    def compute_returns_and_advantage(
        self, last_cost_value, last_shadow_value, done
    ):
        
        last_cost_value = last_cost_value.clone().cpu().numpy().flatten()
        last_shadow_value = last_shadow_value.clone().cpu().numpy().flatten()

        d = MeasureEstimator()
        for obs in self.observations:
            d.append(obs)
        for step in range(self.buffer_size):
            self.shadow_rewards[step] = -np.log(d(self.observations[step]))

        mu_H = self.shadow_rewards.mean()
        mu_J = self.costs.mean()
        
        for step in reversed(range(self.buffer_size)):

            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_cost_value = last_cost_value
                next_shadow_value = last_shadow_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_cost_value = self.cost_values[step + 1]
                next_shadow_value = self.shadow_values[step + 1]

            self.cost_advantages[step] = self.costs[step] - mu_J + \
                    + next_cost_value - self.cost_values[step] 
            self.shadow_advantages[step] = self.shadow_rewards[step] - mu_H + \
                    + next_shadow_value - self.shadow_values[step] 

        self.cost_returns = self.cost_advantages + self.cost_values
        self.shadow_returns = self.shadow_advantages + self.shadow_values
        
    def add(self, obs, action, cost, episode_start,
            cost_value, shadow_value, log_prob):
        
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((1,) + self.obs_shape)
            
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.costs[self.pos] = np.array(cost).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.shadow_values[self.pos] = shadow_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def get(self, batch_size=None):
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
            self.cost_values[batch_inds].flatten(),
            self.shadow_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.shadow_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.shadow_returns[batch_inds].flatten(),
        )
        return InfoRolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


class DiscountedInfoAC:
    def __init__(self,
                 env,
                 policy,
                 value_function,
                 policy_lr,
                 value_lr,
                 kappa=0.1,
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
        self.v_c = value_function
        self.v_r = deepcopy(self.v_c)
        self.kappa = kappa
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

        self.pi_optim = policy_optimizer(self.pi.parameters(),
                                         lr=policy_lr,
                                         weight_decay=weight_decay)
        self.v_c_optim = value_optimizer(self.v_c.parameters(), lr=value_lr)
        self.v_r_optim = value_optimizer(self.v_r.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius

        self.rollout_buffer = DiscountedInfoRolloutBuffer(
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
        self.v_c.to(self.device)
        self.v_r.to(self.device)
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """
        Load policy and value functions. Copy them to target functions.

        This method is for evaluation only. Use load_checkpoint to continue
        training.
        """
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v_c.load_state_dict(models['v_c_state_dict'])
        self.v_r.load_state_dict(models['v_r_state_dict'])

        self.pi.eval()
        self.v_c.eval()
        self.v_r.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
                'v_c_state_dict': self.v_c.state_dict(),
                'v_c_optimizer_state_dict': self.v_c_optim.state_dict(),
                'v_r_state_dict': self.v_r.state_dict(),
                'v_r_optimizer_state_dict': self.v_r_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v_c.load_state_dict(models['v_c_state_dict'])
        self.v_c_optim.load_state_dict(models['v_c_optimizer_state_dict'])
        self.v_r.load_state_dict(models['v_r_state_dict'])
        self.v_r_optim.load_state_dict(models['v_r_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v_c.train()
            self.v_r.train()
        else:
            self.pi.eval()
            self.v_c.eval()
            self.v_r.eval()
        
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
                cost_value = self.v_c(obs_tensor)
                shadow_value = self.v_r(obs_tensor)
            action = action.cpu().numpy()

            # Rescale and perform action
            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low,
                                         self.env.action_space.high)
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                clipped_action = int(clipped_action)

            new_obs, cost, done, info = env.step(clipped_action)

            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)

            self.rollout_buffer.add(self._last_obs, action,
                                    cost, self._last_episode_start,
                                    cost_value, shadow_value,
                                    log_prob)
            self._last_obs = new_obs
            self._last_episode_start = done

            if done:
                env.reset()

        self.rollout_buffer.compute_returns_and_advantage(
            last_cost_value=cost_value, last_shadow_value=shadow_value,
            done=done
        )

        rollout_cost = np.mean(self.rollout_buffer.costs)
        rollout_shadow_reward = np.mean(self.rollout_buffer.shadow_rewards)
        rho = rollout_cost / (self.kappa + rollout_shadow_reward)
        total_cost = np.sum(self.rollout_buffer.costs) / np.sum(
            self.rollout_buffer.episode_starts)

        return rollout_cost, rollout_shadow_reward, rho

    def train(self):
        """
        Train on the current rollout buffer.
        """        

        for epoch in range(self.n_epochs):

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                obs = rollout_data.observations
                cost_values = self.v_c(obs).flatten()
                shadow_values = self.v_r(obs).flatten()
                log_probs = self.pi.log_probs(obs, actions)
                entropies = self.pi.entropy(obs)

                cost_advantages = rollout_data.cost_advantages
                shadow_advantages = rollout_data.shadow_advantages

                # TODO: should we normalize?
                # cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                #    cost_advantages.std() + 1e-8)
                # shadow_advantages = (shadow_advantages - shadow_advantages.mean()) / (
                #    shadow_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_probs - rollout_data.old_log_prob)

                J = rollout_data.old_cost_values.mean()
                H = rollout_data.old_shadow_values.mean()
                advantages = (self.kappa + H)**(-2) * (
                    cost_advantages * (self.kappa + H) - \
                    J * shadow_advantages)
                # advantages = J**(-2) * (J * shadow_advantages - \
                #                         cost_advantages * (self.kappa + H))

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * torch.clamp(ratio,
                #                                          1 - self.clip_range,
                #                                          1 + self.clip_range)
                # policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() - \
                #         self.entropy_coef * entropies.mean()

                policy_loss = (advantages * log_probs).mean()

                self.pi_optim.zero_grad()
                policy_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                   self.grad_clip_radius)
                self.pi_optim.step()

                cost_value_loss = F.mse_loss(rollout_data.cost_returns,
                                             cost_values)
                shadow_value_loss = F.mse_loss(rollout_data.shadow_returns,
                                               shadow_values)

                self.v_c_optim.zero_grad()
                cost_value_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.v_c.parameters(),
                                                   self.grad_clip_radius)
                self.v_c_optim.step()

                self.v_r_optim.zero_grad()
                shadow_value_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.v_r.parameters(),
                                                   self.grad_clip_radius)
                self.v_r_optim.step()

    def update(self, env, rollout_length):
        cost, entropy, rho = self.collect_rollout(env, rollout_length)
        self.train()
        return cost, entropy, rho


class InfoAC(DiscountedInfoAC):
    def __init__(self,
                 env,
                 policy,
                 value_function,
                 policy_lr,
                 value_lr,
                 kappa=0.1,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
                 buffer_size=2048,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        super().__init__(env, policy, value_function, policy_lr, value_lr,
                         kappa=kappa, entropy_coef=entropy_coef,
                         n_epochs=n_epochs,
                         batch_size=batch_size, weight_decay=weight_decay,
                         buffer_size=buffer_size, enable_cuda=enable_cuda,
                         policy_optimizer=policy_optimizer,
                         value_optimizer=value_optimizer,
                         grad_clip_radius=grad_clip_radius)

        self.rollout_buffer = InfoRolloutBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
        )


class InfoACTwoLayer(InfoAC):
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
                 kappa=0.1,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
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
                         kappa=kappa, entropy_coef=entropy_coef,
                         n_epochs=n_epochs,
                         batch_size=batch_size, weight_decay=weight_decay,
                         buffer_size=buffer_size, enable_cuda=enable_cuda,
                         policy_optimizer=policy_optimizer,
                         value_optimizer=value_optimizer,
                         grad_clip_radius=grad_clip_radius)


class InfoACConvolutional(InfoAC):
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
                 kappa=0.1,
                 entropy_coef=0.01,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
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
                         kappa=kappa, entropy_coef=entropy_coef,
                         n_epochs=n_epochs,
                         batch_size=batch_size, weight_decay=weight_decay,
                         buffer_size=buffer_size, enable_cuda=enable_cuda,
                         policy_optimizer=policy_optimizer,
                         value_optimizer=value_optimizer,
                         grad_clip_radius=grad_clip_radius)
