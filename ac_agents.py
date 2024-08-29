#import matplotlib.pyplot as plt
#import gym
import gymnasium as gym
import numpy as np
from itertools import product
from copy import deepcopy
import os
from scipy.special import softmax
import concurrent.futures
import argparse
#import pandas as pd
#import seaborn as sns
from pathlib import Path
from random import randint
from utils import (
    MeasureEstimator,
    SoftmaxPolicyTabular,
    DensityEstimatorGenerator,
    LinearApproximatorOneToOneBasisV
)
import torch
from abc import ABC, abstractmethod
from entropy_eig_utils import map_entropy, get_avg_entropy
import models
import wesutils

torch.manual_seed(0)

def get_pg_alg(pg_type, state_dim, action_dim, policy_lr, value_lr, state_lows, state_highs, is_mlmc, discrete = True):

    if 'AC' in pg_type: #or 'PPGAE' in pg_type:
        if is_mlmc:
            opt = torch.optim.Adagrad
        else:
            opt = torch.optim.SGD
        if discrete:
            
            return ACNN_Discrete(state_dim, action_dim, policy_lr, value_lr, 
                    state_lows = state_lows, state_highs = state_highs, 
                    policy_optimizer = opt, 
                    value_optimizer = opt)
    
    if 'REINFORCE' in pg_type:
        if discrete:
            return REINFORCE_Discrete(state_dim, action_dim, policy_lr, optimizer=torch.optim.SGD)

    if 'PPGAE' in pg_type:
        if discrete:
            return ACNN_Discrete(state_dim, action_dim, policy_lr, value_lr, 
                    state_lows = state_lows, state_highs = state_highs,
                    policy_optimizer=torch.optim.SGD)

    

    return None



class DeepRLAgent:
    """Base class for agents corresponding to specific RL algorithms."""

    def __init__(self):
        raise NotImplemented("__init__ not implemented.")

    def sample_action(self, state):
        raise NotImplemented("sample_action not implemented.")

    def enable_cuda(self, enable_cuda=True, warn=True):
        """
        Enable or disable CUDA. Issue warning that converting after
        initializing optimizers can cause undefined behavior when using
        optimizers other than Adam or SGD.
        """

        raise NotImplemented("enable_cuda not implemented.")

    def update(self, reward_cost_tuple, next_state):
        raise NotImplemented("update not implemented.")

    def save_models(self):
        raise NotImplemented("save_models not implemented.")

    def load_models(self):
        raise NotImplemented("load_models not implemented.")




class MaxEntREINFORCE(DeepRLAgent):
    """
    Agent for training a REINFORCE-style entropy maximization algorithm.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    action_dim:         action space dimension
    policy:             object of class PolicyNetwork (e.g., CategoricalPolicyTwoLayer)
    policy_lr:          learning rate for the policy parameters
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, policy, policy_lr,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pi = policy
        self.pi_optim = optimizer(self.pi.parameters(), lr=policy_lr)
        self.grad_clip_radius = grad_clip_radius

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

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
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """Load policy and value functions. Copy them to target functions."""
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        
        if continue_training:
            self.pi.train()
        else:
            self.pi.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            
        else:
            self.pi.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def update(self, env, rollout_length):
        """
        Perform a single rollout and corresponding gradient update.

        Parameters
        ----------
        env:                gym environment on which we're learning
        rollout_length:     length of the episode
        """

        states, log_probs = [], []
 

        state = env.state

        for _ in range(rollout_length):
          
            states.append(state)
            action, log_prob = self.pi.sample(torch.FloatTensor(state).unsqueeze(dim=1))
            log_probs.append(log_prob)
            state, _, done, _ = env.step(action)

            if done:
                break

        rewards = [-np.log(d(state)) for state in states]
        mean_reward = np.mean(rewards)

        pi_loss = 0
        for reward, log_prob in zip(rewards, log_probs):
            pi_loss += (reward - mean_reward) * log_prob

        pi_loss = -pi_loss / len(rewards)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        if self.grad_clip_radius is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)
        self.pi_optim.step()

        return mean_reward

    def train(self, env, num_episodes, rollout_length,
              # output_dir, args_list,
              reset_env=True):
        """
        Train on the environment.
        """

        episode_rewards = []

        for i in range(num_episodes):
            if reset_env:
                env.reset()
            mean_reward = self.update(env, rollout_length)
            print(
                f'Episode {i}: mean reward {mean_reward:.2f}')
            episode_rewards.append(mean_reward)


class REINFORCE(MaxEntREINFORCE):
    """
    Agent for training a REINFORCE-style algorithm to minimize long-run average
    reward.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    action_dim:         action space dimension
    policy:             object of class PolicyNetwork (e.g., CategoricalPolicyTwoLayer)
    policy_lr:          learning rate for the policy parameters
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, policy, policy_lr,
                 tau=0.01,
                 enable_cuda=False,
                 optimizer=torch.optim.Adagrad,
                 grad_clip_radius=None):

        self.tau = tau
        self.mu_r = 0
        self.number_of_samples = 0
        self.r = 0

        super().__init__(
            state_dim, action_dim, policy, policy_lr,
            enable_cuda=enable_cuda, optimizer=optimizer,
            grad_clip_radius=grad_clip_radius
        )

    def _torch_state(self, state):
        torch_state = torch.tensor(state, dtype=torch.float) #.reshape(self.state_dim, 1)
        return torch_state

    def get_rollout(self):
        raise NotImplemented("get_rollout not implemented.")

    def estimate_gradient(self, gradients):
        raise NotImplemented("estimate_gradients not implemented.")

    def update(self, env, max_iterations, episode_samples = 1):
        """
        Perform a single rollout and corresponding gradient update.

        Parameters
        ----------
        env:                gym environment on which we're learning
        rollout_length:     length of the episode
        """
        #print('in update function')
        states, torch_states, rewards, next_states, log_probs = [], [], [], [], []



        state = env.get_state()
        #print(state)
        #print('rollout', self.rollout_length)
        i = 0
        rollout = self.get_rollout()

        done = False
        #while i < rollout and self.number_of_samples < max_iterations and not done:
        while i < rollout and episode_samples < max_iterations and not done:
        #for i in range(self.rollout_length):
            #d.append(state)
            #print('sample', i)
            states.append(state) # need non-torch state for measure estimator
            #print(state)
            torch_state = self._torch_state(state)
            torch_states.append(torch_state) # need torch state for eval
            #print('torch state', torch_state)

            action, log_prob = self.pi.sample(torch_state)
            #print('action', action)#, 'log_prob', log_prob)
            log_probs.append(log_prob)
            #print('log probs', log_probs)

            state, reward, done = env.step(action.item())[:3]

            next_states.append(self._torch_state(state))
            #print(state)
            rewards.append(reward)
            self.number_of_samples += 1
            episode_samples += 1
            i += 1


        mean_reward  = np.mean(rewards)
        total_reward = np.sum(rewards)

        self.mu_r = (1 - self.tau) * self.mu_r + self.tau * mean_reward 

        #r_gradients, v_c_losses, pi_losses = [], [], []
        pi_losses = []
        for reward, log_prob in zip(rewards, log_probs):
            #pi_loss += (float(reward) - self.mu_r) * log_prob
            pi_losses.append((float(reward) - self.mu_r) * -log_prob)
            #pi_losses.append(( self.r - float(reward)) * log_prob)
            #r_gradients.append(float(reward) - self.r)

            #print(v_c_loss, pi_loss, r_gradients) #self.r - float(reward)

        #r_gradients /= len(states)
      
        #r_gradient = self.estimate_gradient(r_gradients, is_tensor = False)

        pi_loss = self.estimate_gradient(pi_losses)

        #self.r = self.r + self.policy_lr * r_gradient #.item()

        self.pi_optim.zero_grad()
       
  

        pi_loss.backward()
        #print('pi backward')
        if self.grad_clip_radius is not None:

            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)

        self.pi_optim.step()


        return total_reward, pi_loss.item(), rollout, len(states), done, rewards




class REINFORCE_Discrete(REINFORCE):
    """
    Agent for training a vanilla REINFORCE-type agent to minimize long-run
    average reward.

    This agent uses a fully-connected single-layer categorical policy network.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    num_actions:        number of elements in the (finite) action space
    action_dim:         action space dimension
    policy_lr:          learning rate for the policy parameters
    num_hidden_units:   number of hidden units in the policy network
    init_std:           standard deviation of network parameter initialization
    tau:                geometric mixing rate parameter
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, policy_lr,
                 num_hidden_units=64,
                 init_std=0.001,
                 tau=0.01,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        # policy = models.CategoricalPolicyTwoLayer(state_dim, num_actions,
        #                                           num_hidden_units,
        #                                           init_std)
        policy = models.CategoricalPolicyTwoLayer(
            state_dim, action_dim,
            hidden_layer1_size=num_hidden_units,
            hidden_layer2_size=num_hidden_units
        )
        super().__init__(
            state_dim, action_dim, policy, policy_lr,
            tau=tau, enable_cuda=enable_cuda, optimizer=optimizer,
            grad_clip_radius=grad_clip_radius
        )


class REINFORCE_Continuous(REINFORCE):
    """
    Agent for training a vanilla REINFORCE-type agent to minimize long-run
    average reward.

    This agent uses a fully-connected single-layer categorical policy network.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    num_actions:        number of elements in the (finite) action space
    action_dim:         action space dimension
    policy_lr:          learning rate for the policy parameters
    num_hidden_units:   number of hidden units in the policy network
    init_std:           standard deviation of network parameter initialization
    tau:                geometric mixing rate parameter
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, policy_lr,
                 num_hidden_units=64,
                 init_std=0.001,
                 tau=0.01,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        # policy = models.CategoricalPolicyTwoLayer(state_dim, num_actions,
        #                                           num_hidden_units,
        #                                           init_std)
        policy = models.GaussianPolicyTwoLayer(
            state_dim, action_dim, min_action_val=min_action_val,
            max_action_val=max_action_val,
            hidden_layer1_size=policy_units1,
            hidden_layer2_size=policy_units2
        )
        super().__init__(
            state_dim, action_dim, policy, policy_lr,
            tau=tau, enable_cuda=enable_cuda, optimizer=optimizer,
            grad_clip_radius=grad_clip_radius
        )


class VanillaREINFORCE_Discrete(REINFORCE_Discrete):
    """
    Agent for training a vanilla REINFORCE-type agent to minimize long-run
    average reward.

    This agent uses a fully-connected single-layer categorical policy network.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    num_actions:        number of elements in the (finite) action space
    action_dim:         action space dimension
    policy_lr:          learning rate for the policy parameters
    num_hidden_units:   number of hidden units in the policy network
    init_std:           standard deviation of network parameter initialization
    tau:                geometric mixing rate parameter
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, policy_lr, rollout_length,
                 num_hidden_units=64,
                 init_std=0.001,
                 tau=0.01,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None):


    
        super().__init__(
                state_dim, action_dim, policy_lr,
                 num_hidden_units=num_hidden_units,
                 init_std=init_std,
                 tau=tau,
                 enable_cuda=enable_cuda,
                 optimizer=optimizer,
                 grad_clip_radius=grad_clip_radius
        )

        self.rollout_length = rollout_length


    def get_rollout(self):
        return self.rollout_length

    def estimate_gradient(self, gradients, is_tensor = True):
        if is_tensor:
            return torch.mean(torch.stack(gradients))
        return np.mean(gradients)

class VanillaREINFORCE_Continuous(REINFORCE_Continuous):
    """
    Agent for training a vanilla REINFORCE-type agent to minimize long-run
    average reward.

    This agent uses a fully-connected single-layer categorical policy network.

    Parameters
    ----------
    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    num_actions:        number of elements in the (finite) action space
    action_dim:         action space dimension
    policy_lr:          learning rate for the policy parameters
    num_hidden_units:   number of hidden units in the policy network
    init_std:           standard deviation of network parameter initialization
    tau:                geometric mixing rate parameter
    optimizer:          which torch optimizer to use
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, 
                 action_dim, policy_lr, rollout_length,
                 num_hidden_units=64,
                 init_std=0.001,
                 tau=0.01,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None):


        
        super().__init__(
                 state_dim, 
                 action_dim, policy_lr,
                 num_hidden_units=num_hidden_units,
                 init_std=init_std,
                 tau=tau,
                 enable_cuda=enable_cuda,
                 optimizer=optimizer,
                 grad_clip_radius=grad_clip_radius
        )


        self.rollout_length = rollout_length


    def get_rollout(self):
        return self.rollout_length

    def estimate_gradient(self, gradients, is_tensor = True):
        if is_tensor:
            return torch.mean(torch.stack(gradients))
        return np.mean(gradients)



### Neural network versions of the actor-critic algorithms

class NeuralIRACBASE(DeepRLAgent):
    """
    Agent for training a neural network version of the info ratio actor-critic
    (IRAC) algorithm. The state space is assumed to be finite.

    Parameters
    ----------

    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    action_dim:         action space dimension
    num_actions:        number of elements in the (finite) action space
    policy:             object of class PolicyNetwork (e.g., CategoricalPolicyTwoLayer)
    value_function:     value function object
    policy_lr:          learning rate for the policy parameters
    value_lr:           learning rate for the value function parameters
    kappa:              nonnegative float scaling the importance of entropy
    tau:                mixing rate for exponential moving averages
    mu_r_init:          initial value for entropy exponential moving average
    mu_c_init:          initial value for reward exponential moving average
    entropy_eps:        minimum value at which to truncate entropy in gradient updates
    policy_optimizer:   torch optimizer to use for policy
    value_optimizer:    torch optimizer to use for value functions
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim,   action_dim,
                 policy,
                 policy_lr, value_lr, #rollout_length,
                 kappa=0.0, tau=0.01,
                 mu_r_init=1.0, mu_c_init=0.0,
                 entropy_eps=0.0001,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None,  state_lows = [-1, -1], state_highs = [1, 1], 
                 mesh_step = 0.1, adaptive_traj = 0, 
                 alpha = 20):
        
        
        #print('in irac init')
        self.state_dim = state_dim
        #self.num_states = num_states
        #print(state_dim)
        self.action_dim = action_dim
        #print(action_dim)
        #self.num_actions = num_actions
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        #print('hello')
        self.r = 0
        self.pi = policy

        self.adaptive_traj = adaptive_traj
        self.alpha = alpha

        value_function = models.two_layer_net(
            state_dim, 1,
            hidden_layer1_size=64,
            hidden_layer2_size=64,
            activation= 'ReLU'#value_activation
        )
        self.v_c = value_function
        self.number_of_samples = 0
        #print('num samples')
        #self.v_h = deepcopy(self.v_c)
        self.kappa = kappa
        self.tau = tau
        self.mu_r = 0#mu_r_init
        self.mu_c = mu_c_init
        #print('init mu')
        self.entropy_eps = entropy_eps
        #self.rollout_length = rollout_length
        self.__cuda_enabled = enable_cuda
        #print('cuda enable')
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

        self.pi_optim = policy_optimizer(self.pi.parameters(), lr=policy_lr)
        self.v_c_optim = value_optimizer(self.v_c.parameters(), lr=value_lr)
        #print('optim')
        #self.v_h_optim = value_optimizer(self.v_h.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius
        #print('initalized')
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
        #self.v_h.to(self.device)
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """
        Load policy and value functions. Copy them to target functions.

        This method is for evaluation only. Use load_checkpoint to continue
        training.
        """
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v_c.load_state_dict(models['v_c_state_dict'])
        #self.v_h.load_state_dict(models['v_h_state_dict'])

        self.pi.eval()
        self.v_c.eval()
        #self.v_h.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
                'v_c_state_dict': self.v_c.state_dict(),
                'v_c_optimizer_state_dict': self.v_c_optim.state_dict(),
                # 'v_h_state_dict': self.v_h.state_dict(),
                # 'v_h_optimizer_state_dict': self.v_h_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v_c.load_state_dict(models['v_c_state_dict'])
        self.v_c_optim.load_state_dict(models['v_c_optimizer_state_dict'])
        #self.v_h.load_state_dict(models['v_h_state_dict'])
        #self.v_h_optim.load_state_dict(models['v_h_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v_c.train()
            #self.v_h.train()
        else:
            self.pi.eval()
            self.v_c.eval()
            #self.v_h.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def _torch_state(self, state):
        torch_state = torch.tensor(state, dtype=torch.float) #.reshape(self.state_dim, 1)
        return torch_state

    def get_rollout(self):
        pass

    def estimate_gradient(self, gradients):
        pass

    def update(self, env, max_iterations, episode_samples = 1):
        """
        Perform a single rollout and corresponding gradient update.

        Parameters
        ----------
        env:                gym environment on which we're learning
        rollout_length:     length of the episode
        """
        #print('in update function')
        states, torch_states, rewards, next_states, log_probs = [], [], [], [], []
        #d = MeasureEstimator()


        state = env.get_state()
        #print(state)
        #print('rollout', self.rollout_length)
        i = 0
        rollout = self.get_rollout()

        done = False
        #while i < rollout and self.number_of_samples < max_iterations and not done:
        while i < rollout and episode_samples < max_iterations and not done:
        #for i in range(self.rollout_length):
            #d.append(state)
            #print('sample', i)
            states.append(state) # need non-torch state for measure estimator
            #print('state', state)
            torch_state = self._torch_state(state)
            torch_states.append(torch_state) # need torch state for eval
            #print('torch state', torch_state)

            action, log_prob = self.pi.sample(torch_state)
            #print('action', action, 'log_prob', log_prob)
            log_probs.append(log_prob)
            #print('log probs', log_probs)
            #print(action)
            state, reward, done = env.step(action.item())[:3]
            #print(done)
            #print('new state', state)
            next_states.append(self._torch_state(state))
            #print(state)
            rewards.append(reward)
            self.number_of_samples += 1
            episode_samples += 1
            i += 1

       
        mean_reward  = np.mean(rewards)
        total_reward = np.sum(rewards)
       

        #r_gradients, v_c_loss, pi_loss = 0, 0, 0

        r_gradients, v_c_losses, pi_losses = [], [], []

        self.mu_r = (1 - self.tau) * self.mu_r + self.tau * mean_reward 

        v_c_loss = 0
        pi_loss  = 0
        for state, reward, next_state, log_prob in zip(
            torch_states, rewards, next_states, log_probs):

            with torch.no_grad():
                #print('in no grad')
                v_c_target = float(reward) - self.mu_r + self.v_c(next_state)
                #v_c_target =  self.r - float(reward) + self.v_c(next_state)
                td_c = v_c_target - self.v_c(state)

            #v_c_loss += (self.v_c(state) - v_c_target) ** 2

            #pi_loss += (td_c * -log_prob)
            
            # r_gradients += (float(reward) - self.r)

            v_c_losses.append((self.v_c(state) - v_c_target) ** 2)

            pi_losses.append(td_c * -log_prob)

            
            #r_gradients.append(float(reward) - self.r)
            #print(v_c_loss, pi_loss, r_gradients)

        #r_gradients /= len(states)
      
        #r_gradient = self.estimate_gradient(r_gradients, is_tensor = False)
        #r_gradient = np.mean(r_gradients)

        v_c_loss = self.estimate_gradient(v_c_losses)

        #v_c_loss /= len(states)
        #v_h_loss /= len(states)
        #pi_loss /= len(states) 

        pi_loss = self.estimate_gradient(pi_losses) #

        #self.r = self.r  self.value_lr * r_gradient #.item()
        #self.r = self.r + self.value_lr * r_gradient #(float(reward) - self.r)

        self.v_c_optim.zero_grad()
        #print('v opt zero')
        #self.v_h_optim.zero_grad()
        self.pi_optim.zero_grad()
        #print('pi optimizer zero')
        v_c_loss.backward()

        pi_loss.backward()
        #print('pi backward')
        if self.grad_clip_radius is not None:
            torch.nn.utils.clip_grad_norm_(self.v_c.parameters(),
                                           self.grad_clip_radius)
            torch.nn.utils.clip_grad_norm_(self.v_h.parameters(),
                                           self.grad_clip_radius)
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)
        self.v_c_optim.step()
        #self.v_h_optim.step()
        #print('v optim step')
        self.pi_optim.step()
        #print('pi optim step')
        #self.number_of_samples += len(states)
        #print(self.number_of_samples)


        return total_reward, pi_loss.item(), rollout, len(states), done, rewards

    def train(self, env, num_episodes, rollout_length,
              reset_env=True):
        """
        Train on the environment.
        """

        rewards, entropies, info_ratios = [], [], []

        for i in range(num_episodes):
            if reset_env:
                env.reset()
            reward, entropy, info_ratio = self.update(env, rollout_length)
            print(
                f'Episode {i}: J {reward:.2f}, H {entropy:.2f}, IR {info_ratio:.2f}'
            )
            rewards.append(reward)
            entropies.append(entropy)
            info_ratios.append(info_ratio)

        return rewards, entropies, info_ratios



class ACNN_Continuous(NeuralIRACBASE):
    """
    Agent for training a neural network version of the info ratio actor-critic
    (IRAC) algorithm. The policy is a two-layer softmax policy, while the value
    functions are two-layer fully connected neural networks.

    The state space is assumed to be finite.

    Parameters
    ----------

    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    action_dim:         action space dimension
    num_actions:        number of elements in the (finite) action space
    policy_lr:          learning rate for the policy parameters
    value_lr:           learning rate for the value function parameters
    policy_units1:      number of hidden units in first policy layer
    policy_units2:      number of hidden units in second policy layer
    value_units1:       number of hidden units in first value function layer
    value_units2:       number of hidden units in second value function layer
    value_activation:   activation function for value networks
    kappa:              nonnegative float scaling the importance of entropy
    tau:                mixing rate for exponential moving averages
    mu_r_init:          initial value for entropy exponential moving average
    mu_c_init:          initial value for reward exponential moving average
    entropy_eps:        minimum value at which to truncate entropy in gradient updates
    policy_optimizer:   torch optimizer to use for policy
    value_optimizer:    torch optimizer to use for value functions
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, 
                 policy_lr, value_lr, sample_length, 
                 min_action_val=-1.0,
                 max_action_val=1.0,
                 policy_units1=64, policy_units2=64,
                 value_units1=64, value_units2=64,
                 value_activation='ReLU',
                 kappa=0.0, tau=0.01,
                 mu_r_init=1.0, mu_c_init=0.0,
                 entropy_eps=0.0001,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None, state_lows = [-1, -1], state_highs = [1, 1], 
                 mesh_step = 0.1, adaptive_traj = 0, 
                 alpha = 20):
        #print('in irac 2 layer')
        #print('state dim', state_dim, 'action_dim', action_dim, 'min action', min_action_val, 'max action', max_action_val)
        
        policy = models.GaussianPolicyTwoLayer(
            state_dim, action_dim, min_action_val=min_action_val,
            max_action_val=max_action_val,
            hidden_layer1_size=policy_units1,
            hidden_layer2_size=policy_units2
        )
        #print('hello')
        #print(policy)
        #print(value_units1, value_units2, value_activation)


        super().__init__(
            state_dim, action_dim, 
            policy, policy_lr, value_lr, #rollout_length,
            kappa=kappa, tau=tau,
            mu_r_init=mu_r_init, mu_c_init=mu_c_init,
            entropy_eps=entropy_eps,
            enable_cuda=enable_cuda,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            grad_clip_radius=grad_clip_radius, state_lows = state_lows, state_highs = state_highs, 
            mesh_step = mesh_step, adaptive_traj = adaptive_traj, 
            max_traj = max_traj
        )




class ACNN_Discrete(NeuralIRACBASE):
    """
    Agent for training a neural network version of the info ratio actor-critic
    (IRAC) algorithm. The policy is a two-layer softmax policy, while the value
    functions are two-layer fully connected neural networks.

    The state space is assumed to be finite.

    Parameters
    ----------

    state_dim:          state space dimension
    num_states:         number of elements in the (finite) state space
    action_dim:         action space dimension
    num_actions:        number of elements in the (finite) action space
    policy_lr:          learning rate for the policy parameters
    value_lr:           learning rate for the value function parameters
    policy_units1:      number of hidden units in first policy layer
    policy_units2:      number of hidden units in second policy layer
    value_units1:       number of hidden units in first value function layer
    value_units2:       number of hidden units in second value function layer
    value_activation:   activation function for value networks
    kappa:              nonnegative float scaling the importance of entropy
    tau:                mixing rate for exponential moving averages
    mu_r_init:          initial value for entropy exponential moving average
    mu_c_init:          initial value for reward exponential moving average
    entropy_eps:        minimum value at which to truncate entropy in gradient updates
    policy_optimizer:   torch optimizer to use for policy
    value_optimizer:    torch optimizer to use for value functions
    grad_clip_radius:   float values enable corresponding gradient clipping
    """

    def __init__(self,
                 state_dim, action_dim, 
                 policy_lr, value_lr, #rollout_length, 
                 min_action_val=-1.0,
                 max_action_val=1.0,
                 policy_units1=200, policy_units2=200,
                 value_units1=200, value_units2=200,
                 value_activation='ReLU',
                 kappa=0.0, tau=0.01,
                 mu_r_init=1.0, mu_c_init=0.0,
                 entropy_eps=0.0001,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adagrad,
                 #policy_optimizer=torch.optim.SGD,
                 value_optimizer=torch.optim.Adagrad,
                 grad_clip_radius=None, state_lows = [-1, -1], state_highs = [1, 1], 
                 mesh_step = 0.1, adaptive_traj = 0, 
                 alpha = 20):
        #print('in irac 2 layer')
        #print('state dim', state_dim, 'action_dim', action_dim, 'min action', min_action_val, 'max action', max_action_val)
        
        policy = models.CategoricalPolicyTwoLayer(
            state_dim, action_dim,
            hidden_layer1_size=policy_units1,
            hidden_layer2_size=policy_units2
        )
        print('hello')
        print(policy)
        #print(value_units1, value_units2, value_activation)


        super().__init__(
            state_dim, action_dim, 
            policy, policy_lr, value_lr, # rollout_length,
            kappa=kappa, tau=tau,
            mu_r_init=mu_r_init, mu_c_init=mu_c_init,
            entropy_eps=entropy_eps,
            enable_cuda=enable_cuda,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            grad_clip_radius=grad_clip_radius, state_lows = state_lows, state_highs = state_highs, 
            mesh_step = mesh_step, adaptive_traj = adaptive_traj, 
            alpha = alpha
        )




class AC:

    def __init__(self, num_states, num_actions, policy, value_function, policy_lr, value_lr, 
                    adaptive_traj = 0, max_iters = 500,
                    max_traj = 20):
        #print('in AC')
        self.num_states = num_states
        self.num_actions = num_actions
        self.pi = policy
        self.v = value_function
        self.r = 0
        self.policy_lr = policy_lr
        self.value_lr = value_lr
       
        self.max_iters = max_iters
        #self.trajectory-length = trajectory_length
        #print('max iters', self.max_iters)
        self.initial_entropy = get_avg_entropy(self.num_states, self.pi)
        self.current_entropy = self.initial_entropy
        self.total_number_of_samples = 0
        self.max_traj = max_traj
        self.value_table = np.zeros(num_states)


    def get_trajectory_length(self):
        pass


    def estimate_gradient(self, gradients):
        pass

    def update_params(self, r_grads, v_grads, pi_grads):
        pass

    def adjust_trajectory(self, mapped_trajectory):
        pass

    def update(self, env, episode_samples, state_visit_freq):
        #print('in update')
        states, actions, rewards, next_states = [], [], [], []
        state = env.state

        trajectory_length = self.get_trajectory_length()
        #print('traj length', trajectory_length)

        sample = 0
        done = False
        
        while sample < trajectory_length and episode_samples < self.max_iters and not done:
            #print('sample', sample)
            states.append(state)
            #print('states', states)
            old_state = state
            #print('old state', old_state)
            action = self.pi.sample(state) 
            #print('action', action)

            actions.append(action)

            state, reward, done = env.step(action)[:3]

            #print('state', state, 'reward', reward, 'done', done)
      
            rewards.append(reward)

            next_states.append(state)

            self.total_number_of_samples += 1 
            sample += 1
            episode_samples += 1

        #     print(states)
        # print(len(states))
    
        mean_rewards = np.mean(rewards)
        total_rewards = np.sum(rewards)

        r_grads, v_grads, pi_grads = [], [], []
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            

            state_visit_freq[state] += 1
            
            td = self.r - float(reward) + self.v(next_state) - self.v(state)
         
            r_gradient = (float(reward) - self.r)

            #td = self.td_loss(float(reward))

            v_gradient = td * self.v.gradient(state)
            
            pi_gradient = td * self.pi.grad_log_policy(state, action)
            
            r_grads.append(r_gradient)
            v_grads.append(v_gradient)
            pi_grads.append(pi_gradient)
            
        # r_grad = self.estimate_grad(r_grads)
        # self.r = self.r + self.value_lr * r_grad

        # v_grad = self.estimate_grad(v_grads)
        # self.v.params = self.v.params + self.value_lr * v_grad

        # pi_grad = self.estimate_grad(pi_grads)
        # self.pi.params = self.pi.params + self.policy_lr * pi_grad
        #print('got gradients')
        pi_grad = self.update_params(r_grads, v_grads, pi_grads)
        #print('pi grad', pi_grad)
        for state in range(196):
            
            self.value_table[state] = self.v(state)
            
        # if done:
        #     self.value_table[next_states[-1]] = self.v(next_states[-1])


        # if self.adaptive_traj:
        #     self.current_entropy, _ = get_avg_entropy(self.num_states, self.pi)
        #     #print('current_entropy', current_entropy)
        #     mapped_trajectory = map_entropy(self.initial_entropy, self.initial_rollout, self.max_traj, self.current_entropy)
        #     #print(x)

        #     self.adjust_trajectory(mapped_trajectory)


        return total_rewards, np.linalg.norm(pi_grad), trajectory_length, len(states), self.total_number_of_samples, self.current_entropy, done, self.pi, self.value_table





class VAC(AC):

    def __init__(self, num_states, num_actions, 
                    policy_lr, value_lr, policy_cov_constant=1.0,
                    value_cov_constant=1.0,
                    adaptive_traj = 0, max_iters = 500,
                    max_traj = 20, trajectory_length = 16):

        print('in VAC')
        self.trajectory_length = trajectory_length
        
        states  = list(range(num_states))
        actions = list(range(num_actions))
        policy = SoftmaxPolicyTabular(
            states, actions, policy_cov_constant)
        value_function = LinearApproximatorOneToOneBasisV(
            states, init_cov_constant=value_cov_constant)
        print('made value function')

        super().__init__(num_states, num_actions, policy, 
                    value_function, policy_lr, value_lr, 
                    adaptive_traj = adaptive_traj, max_iters = max_iters,
                    max_traj = max_traj)

    def get_trajectory_length(self):
        return self.trajectory_length

    def estimate_gradients(self, gradients):
        #print('in estimate gradients', gradients)
        return np.mean(gradients)
        # avg_grad = 0
        # for g in gradients:
        #     avg_grad += g
        # return avg_grad/len(gradients)

    def update_params(self, r_grads, v_grads, pi_grads):

        r_grad = self.estimate_gradients(r_grads)
        self.r = self.r + self.value_lr * r_grad

        v_grad = self.estimate_gradients(v_grads)
        self.v.params = self.v.params + self.value_lr * v_grad

        pi_grad = self.estimate_gradients(pi_grads)
        self.pi.params = self.pi.params + self.policy_lr * pi_grad

        return pi_grad

    def adjust_trajectory(self, mapped_trajectory):

        if mapped_trajectory != self.trajectory_length:
            self.trajectory_length = mapped_trajectory
            #print('new rollout', mapped_rollout, self.iteration_num)  


class VAC_NN(AC):

    def __init__(self, state_dim, action_dim, 
                    policy_lr, value_lr, policy_cov_constant=1.0,
                    value_cov_constant=1.0,
                    adaptive_traj = 0, max_iters = 500,
                    max_traj = 20, trajectory_length = 16):

        print('in VAC')
        self.trajectory_length = trajectory_length
        
        states  = list(range(num_states))
        actions = list(range(num_actions))
        policy = SoftmaxPolicyTabular(
            states, actions, policy_cov_constant)
        value_function = LinearApproximatorOneToOneBasisV(
            states, init_cov_constant=value_cov_constant)
        print('made value function')

        super().__init__(num_states, num_actions, policy, 
                    value_function, policy_lr, value_lr, 
                    adaptive_traj = adaptive_traj, max_iters = max_iters,
                    max_traj = max_traj)

    def get_trajectory_length(self):
        return self.trajectory_length

    def estimate_gradients(self, gradients):
        #print('in estimate gradients', gradients)
        return np.mean(gradients)
        # avg_grad = 0
        # for g in gradients:
        #     avg_grad += g
        # return avg_grad/len(gradients)

    def update_params(self, r_grads, v_grads, pi_grads):

        r_grad = self.estimate_gradients(r_grads)
        self.r = self.r + self.value_lr * r_grad

        v_grad = self.estimate_gradients(v_grads)
        self.v.params = self.v.params + self.value_lr * v_grad

        pi_grad = self.estimate_gradients(pi_grads)
        self.pi.params = self.pi.params + self.policy_lr * pi_grad

        return pi_grad

    def adjust_trajectory(self, mapped_trajectory):

        if mapped_trajectory != self.trajectory_length:
            self.trajectory_length = mapped_trajectory
            #print('new rollout', mapped_rollout, self.iteration_num) 


class MAC(AC):

    def __init__(self, num_states, num_actions, policy_lr, value_lr, policy_cov_constant=1.0,
                    value_cov_constant=1.0,
                    adaptive_traj = 0, max_iters = 500,
                    max_traj = 20, log_tmax = 16):
        print('in mac')
        self.log_tmax = log_tmax
        self.reward_grad_sum = 0
        self.value_grad_sum = 0
        self.pi_grad_sum = 0

        states  = list(range(num_states))
        actions = list(range(num_actions))
        policy = SoftmaxPolicyTabular(
            states, actions, policy_cov_constant)
        value_function = LinearApproximatorOneToOneBasisV(
            states, init_cov_constant=value_cov_constant)

        super().__init__(num_states, num_actions, policy, 
                    value_function, policy_lr, value_lr, 
                    adaptive_traj = adaptive_traj, max_iters = max_iters,
                    max_traj = max_traj)

    def get_trajectory_length(self):
        geom_draw = np.random.geometric(p=0.50, size=1)[0]
        #print(f'log tmax = {np.log2(self.t_max)}')
        #print(geom_draw)
        
        while 2**geom_draw > self.max_iters:
            
            geom_draw = (np.random.geometric(p=0.50, size=1)[0])
        
        return 2**geom_draw

    def estimate_gradients(self, gradients):
        #print('estimate gradients', gradients)
        if 1 < len(gradients) < (2**self.log_tmax):
            #print('in if')
            g1 = np.mean(gradients)
            #print(g1)
            
            g2 = np.mean(gradients[:len(gradients)//2])
            #print(g2)
            return gradients[0] + len(gradients)*(g1 - g2)     

       
        return gradients[0]


    def update_params(self, r_grads, v_grads, pi_grads):
        #print('in update params')
        r_mlmc = self.estimate_gradients(r_grads)

        self.reward_grad_sum += (np.linalg.norm(r_mlmc)**2)

        x = np.sqrt(self.reward_grad_sum)
        if x == 0:
            x = 1

        self.r = self.r + self.value_lr/x * r_mlmc


        value_mlmc = self.estimate_gradients(v_grads)
        self.value_grad_sum += (np.linalg.norm(value_mlmc)**2)
        x = np.sqrt(self.value_grad_sum)
        if x == 0:
            x = 1

        self.v.params = self.v.params + self.value_lr/x * value_mlmc


        pi_mlmc = self.estimate_gradients(pi_grads)
        #print('pi mlmc', pi_mlmc)
        self.pi_grad_sum += (np.linalg.norm(pi_mlmc)**2)
        x = np.sqrt(self.pi_grad_sum)
        if x == 0:
            x = 1

        self.pi.params = self.pi.params + self.policy_lr/x * pi_mlmc

        return pi_mlmc

    def adjust_trajectory(self, mapped_trajectory):

        if mapped_trajectory != self.log_tmax:
            self.log_tmax = mapped_trajectory


def chooseAC(ac_type, num_states, num_actions, policy_lr, value_lr, 
            sample_length, adaptive_traj, iterations, max_traj):
    print('in choose agent')
    print(ac_type)
    if ac_type == 'MAC':
        print('MAC')
        print(f'tmax {sample_length}')
        return MAC(num_states, num_actions, policy_lr, value_lr, adaptive_traj = adaptive_traj, max_iters= iterations, max_traj = max_traj, log_tmax = sample_length,)
    # elif ac_type == 'MAG':
    #     print(f'tmax {t_max}')
    # 
    #     ac = TabularMAG(num_states, num_actions, policy_lr, value_lr, t_max = t_max, adaptive_traj = adaptive_traj, max_iters = iterations, max_traj = max_traj)
#     elif ac_type == 'VAC_NN':
#         print('VAC NN')
#         print(f'rollout length {sample_length}')

#         return VAC_NN(num_states, num_actions, policy_lr, value_lr, adaptive_traj = adaptive_traj, max_iters = iterations, max_traj = max_traj, trajectory_length = sample_length)

# state_dim, action_dim, 
#                  policy_lr, value_lr, rollout_length, min_action_val=-1.0,
#                  max_action_val=1.0,
#                  policy_units1=64, policy_units2=64,
#                  value_units1=64, value_units2=64,
#                  value_activation='ReLU',
#                  kappa=0.0, tau=0.01,
#                  mu_r_init=1.0, mu_c_init=0.0,
#                  entropy_eps=0.0001,
#                  enable_cuda=False,
#                  policy_optimizer=torch.optim.Adam,
#                  value_optimizer=torch.optim.Adam,
#                  grad_clip_radius=None, state_lows = [-1, -1], state_highs = [1, 1], 
#                  mesh_step = 0.1, adaptive_traj = 0, 
#                  max_traj = 20):


    else:
        print(f'rollout length {sample_length}')
        return VAC(num_states, num_actions, policy_lr, value_lr, adaptive_traj = adaptive_traj, max_iters = iterations, max_traj = max_traj, trajectory_length = sample_length, )



# class AC:

#     def __init__(self, num_states, num_actions,
#                  policy, value_function,
#                  policy_lr, value_lr, rollout_length = 2,
#                  mu_c_init=0.0, tau=0.01,
#                  entropy_eps=0.0001,
#                  kappa=0.0, adaptive_traj = 0, 
#                  iter_change = 5000,
#                  max_iters = 4e5, t_max = 4, max_traj = 20):

#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.pi = policy
#         self.v = value_function
#         self.r = 0
#         self.policy_lr = policy_lr
#         self.value_lr = value_lr
#         self.tau = tau
#         self.mu_c = mu_c_init
#         self.entropy_eps = entropy_eps
#         self.kappa = kappa
#         #self.initial_rollout_length = rollout_length
#         # self.rollout_length = rollout_length
#         self.initial_rollout = rollout_length
#         self.reinit_params()
#         # self.t_max = t_max
#         # self.initial_t_max_log2 = np.log2(t_max)
#         # print(self.initial_t_max_log2)
#         self.adaptive_traj = adaptive_traj
#         self.iteration_num = 0
#         self.iter_change = iter_change
#         self.max_iters = max_iters
#         print('max iters', self.max_iters)
#         self.initial_entropy, self.pi_to_list  = get_avg_entropy(self.num_states, self.pi)
#         self.current_entropy = self.initial_entropy
#         self.number_of_samples = 0
#         self.max_traj = max_traj


#     def reinit_params(self):
#         self.pi.reinit_params()
#         self.v.reinit_params()

 
#     def get_rollout(self):
#         pass

   
#     def estimate_grad(self, gradients):
#         pass


#     def adjust_traj(self, mapped_rollout):
#         pass

#     def update(self, env, gym = False, num_states = 625):
#         """
#         Perform a single rollout and corresponding gradient update.
#         Parameters
#         ----------
#         env:                gym environment on which we're learning
#         rollout_length:     length of the episode
#         """

#         states, actions, rewards, next_states = [], [], [], []
     
#         if gym:
#             state = env.s
#         else:
#             state = env.state
     
#         trajectory_length = self.get_trajectory_length()

#         for _ in range(trajectory_length):
            
#             states.append(state)
#             old_state = state

#             action = self.pi.sample(state) 

#             actions.append(action)

#             #results = env.step(action)

#             #state, reward, done = results[0], results[1], results[2]
#             # if gym:
#             #     state, reward, done, _ , _ = env.step(action)[:-2]
#             #     env.s = state
#             #     if env.s == num_states - 1:

#             #         reward = 1
#             #     if done and env.s != num_states - 1:
#             #         env.s = 0
#             #         state = 0
#             #         #print('fell into back to beginning')
#             # else:
#             state, reward, done = env.step(action)[:3]

      
#             rewards.append(reward)

#             next_states.append(state)


#         mean_rewards = np.mean(rewards)
#         total_rewards = np.sum(rewards)

#         r_grads, v_grads, pi_grads = [], [], []
#         for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            
            
#             td = float(reward) - self.r + self.v(next_state) - self.v(state)
         
#             r_gradient = (float(reward) - self.r)
#             v_gradient = td * self.v.gradient(state)
            
#             pi_gradient = td * self.pi.grad_log_policy(state, action)
            
#             r_grads.append(r_gradient)
#             v_grads.append(v_gradient)
#             pi_grads.append(pi_gradient)
            
#         r_grad = self.estimate_grad(r_grads)
#         self.r = self.r + self.value_lr * r_grad

#         v_grad = self.estimate_grad(v_grads)
#         self.v.params = self.v.params + self.value_lr * v_grad

#         pi_grad = self.estimate_grad(pi_grads)
#         self.pi.params = self.pi.params + self.policy_lr * pi_grad

            
#         #print(entropy)
#         self.iteration_num += 1
#         self.number_of_samples += len(states) #self.rollout_length

        
#         if self.adaptive_traj:
#             self.current_entropy, self.pi_to_list = get_avg_entropy(self.num_states, self.pi)
#             #print('current_entropy', current_entropy)
#             mapped_rollout = map_entropy(self.initial_entropy, self.initial_rollout, self.max_traj, self.current_entropy)
#             #print(x)

#             def adjust_traj(self, mapped_rollout):
#                 if mapped_rollout != self.rollout_length:
#                     self.rollout_length = mapped_rollout
#                     print('new rollout', mapped_rollout, self.iteration_num)



#         return total_rewards, np.linalg.norm(pi_grad), self.rollout_length, self.pi, self.number_of_samples, self.current_entropy, self.pi_to_list, done




# class VanillaACBASEReward(AC):
#     """
#     Agent for training an actor-critic algorithm to minimize average reward.
#     This is a BASE CLASS for this type of agent. A policy must be specified.
#     """

#     def __init__(self, num_states, num_actions,
#                  policy, value_function,
#                  policy_lr, value_lr, rollout_length,
#                  mu_c_init=0.0, tau=0.01,
#                  entropy_eps=0.0001,
#                  kappa=0.0, adaptive_traj = 0, 
#                  iter_change = 5000,
#                  max_iters = 4e5, max_traj = 20):

#         super().__init__(num_states, num_actions,
#                          policy, value_function,
#                          policy_lr, value_lr, rollout_length = rollout_length,
#                          mu_c_init=mu_c_init, tau=tau,
#                          entropy_eps=entropy_eps,
#                          kappa=kappa, 
#                          adaptive_traj = adaptive_traj,
#                          iter_change = iter_change, max_iters= max_iters, max_traj = max_traj)


#         #print(self.initial_entropy)
#         # if self.adaptive_traj:
#         #     self.rollout_length = rollout_length
#     def get_rollout(self):
#         return self.rollout_length
#     def estimate_grad(self, gradients):
#         #print(np.mean(np.array(gradients)))
#         print(gradients)
#         return np.mean(np.array(gradients))

    
#     # def adjust_traj(self, mapped_rollout):
#     #     if mapped_rollout != self.rollout_length:
#     #         self.rollout_length = mapped_rollout
#     #         print('new rollout', mapped_rollout, self.iteration_num)



# class VanillaACTabularReward(VanillaACBASEReward):
#     """
#     Agent for training an actor-critic algorithm to minimize cost
#     in the tabular setting.
#     States and actions must be scalar.
#     """

#     def __init__(self, num_states, num_actions,
#                  policy_lr, value_lr, rollout_length,
#                  policy_cov_constant=1.0,
#                  value_cov_constant=1.0,
#                  mu_c_init=0.0, tau=0.01,
#                  entropy_eps=0.0001,
#                  kappa=0.0, adaptive_traj = 0, 
#                  iter_change = 5000,
#                  max_iters = 4e5, max_traj = 20):
    
#         states  = list(range(num_states))
#         actions = list(range(num_actions))
#         policy = SoftmaxPolicyTabular(
#             states, actions, policy_cov_constant)
#         value_function = LinearApproximatorOneToOneBasisV(
#             states, init_cov_constant=value_cov_constant)

#         super().__init__(num_states, num_actions,
#                          policy, value_function,
#                          policy_lr, value_lr, rollout_length = rollout_length,
#                          mu_c_init=mu_c_init, tau=tau,
#                          entropy_eps=entropy_eps,
#                          kappa=kappa, 
#                          adaptive_traj = adaptive_traj,
#                          iter_change = iter_change, max_iters= max_iters, max_traj = max_traj)


# # # Defining MLMC AC


# class ACMLMC(AC):
#     """
#     Agent for training an actor-critic algorithm to minimize average reward.
#     This is a BASE CLASS for this type of agent. A policy must be specified.
#     """

#     def __init__(self, num_states, num_actions,
#                  policy, value_function,
#                  policy_lr, value_lr,
#                  mu_c_init=0.0, tau=0.01,
#                  entropy_eps=0.0001,
#                  kappa=0.0, t_max = 4, adaptive_traj = 0, iter_change = 5000,
#                  max_iters = 4e5, max_traj = 20):

#         super().__init__(num_states, num_actions,
#                          policy, value_function,
#                          policy_lr, value_lr, rollout_length = rollout_length,
#                          mu_c_init=mu_c_init, tau=tau,
#                          entropy_eps=entropy_eps,
#                          kappa=kappa, t_max = t_max,
#                          adaptive_traj = adaptive_traj,
#                          iter_change = iter_change, max_iters= max_iters, max_traj = max_traj)


#     def get_rollout(self):

#         if self.t_max == 1:
#             #print('equal 1')
#             return 1

#         else:
        
#             geom_draw = (np.random.geometric(p=0.50, size=1)[0])
#             #print(f'log tmax = {np.log2(self.t_max)}')
#             #print(geom_draw)
            
#             while geom_draw > np.log2(self.t_max):
                
#                 geom_draw = (np.random.geometric(p=0.50, size=1)[0])
#                 #print(geom_draw)
#             #print(self.t_max)
#             #print(geom_draw)
#             return 2**geom_draw


#     def estimate_gradient(self, gradients):
#         rollout_length = len(gradients)

#         if rollout_length == 1:
#             return gradients[0]
#         g1 = np.mean(gradients)
#         #print('g1', g1)
#         g2 = np.mean(gradients[:rollout_length//2])
#         #print('g2', g2)
#         return gradients[0] + (rollout_length)*(g1 - g2)

#     def adjust_traj(self, mapped_rollout):
#         if 2**mapped_rollout != self.t_max:
#             self.t_max = (2**mapped_rollout)
#             print('new t_max', 2**mapped_rollout, self.iteration_num)
    
# class ACTabularMLMC(ACMLMC):
#     """
#     Agent for training an actor-critic algorithm to minimize cost
#     in the tabular setting.
#     States and actions must be scalar.
#     """

#     def __init__(self, num_states, num_actions,
#                  policy_lr, value_lr,
#                  policy_cov_constant=1.0,
#                  value_cov_constant=1.0,
#                  mu_c_init=0.0, tau=0.01,
#                  entropy_eps=0.0001,
#                  kappa=0.0, t_max = 1, adaptive_traj = 0, iter_change = 5000, max_iters = 4e5,
#                  max_traj = 20):
    
#         states = list(range(num_states))
#         actions = list(range(num_actions))
#         policy = SoftmaxPolicyTabular(
#             states, actions, policy_cov_constant)
#         value_function = LinearApproximatorOneToOneBasisV(
#             states, init_cov_constant=value_cov_constant)

#         super().__init__(num_states, num_actions,
#                          policy, value_function,
#                          policy_lr, value_lr,
#                          mu_c_init=mu_c_init, tau=tau,
#                          entropy_eps=entropy_eps,
#                          kappa=kappa, t_max = t_max, 
#                          adaptive_traj = adaptive_traj, 
#                          iter_change = iter_change, max_iters = max_iters, max_traj = max_traj)








# # class MAG:
# #     """
# #     Agent for training an actor-critic algorithm to minimize average reward.
# #     This is a BASE CLASS for this type of agent. A policy must be specified.
# #     """

# #     def __init__(self, num_states, num_actions,
# #                  policy, value_function,
# #                  policy_lr, value_lr,
# #                  mu_c_init=0.0, tau=0.01,
# #                  entropy_eps=0.0001,
# #                  kappa=0.0, t_max = 1, adaptive_traj = 0, iter_change = 5000,
# #                  max_iters = 4e5, range_divider= 2500):

# #         self.num_states = num_states
# #         self.num_actions = num_actions
# #         self.pi = policy
# #         self.v = value_function
# #         self.r = 0
# #         self.policy_lr = policy_lr
# #         self.value_lr = value_lr
# #         self.reward_grad_sum = 0
# #         self.value_grad_sum = 0
# #         self.pi_grad_sum = 0
# #         self.tau = tau
# #         self.mu_c = mu_c_init
# #         self.entropy_eps = entropy_eps
# #         self.kappa = kappa
# #         self.t_max = t_max
# #         self.initial_t_max_log2 = np.log2(t_max)
# #         print(self.initial_t_max_log2)
# #         self.reinit_params()
# #         self.adaptive_traj = adaptive_traj
# #         self.iteration_num = 0
# #         self.max_iters = max_iters
# #         self.initial_entropy = get_avg_entropy(self.num_states, self.pi)
# #         self.iter_change = iter_change
# #         self.number_of_samples = 0
# #         self.range_divider = range_divider

# #         # if self.adaptive_traj:
# #         #     self.rollout_length = rollout_length
# #     def reinit_params(self):
# #         self.pi.reinit_params()
# #         self.v.reinit_params()



# #     def MLMC(self, gradients):
# #         rollout_length = len(gradients)

# #         if rollout_length == 1 or rollout_length > self.t_max:
# #             return gradients[0]
# #         g1 = np.mean(gradients)
# #         #print('g1', g1)
# #         g2 = np.mean(gradients[:rollout_length//2])
# #         #print('g2', g2)
# #         return gradients[0] + (rollout_length)*(g1 - g2)

        
# #     def update(self, env, gym = False):
# #         """
# #         Perform a single rollout and corresponding gradient update.
# #         Parameters
# #         ----------
# #         env:                gym environment on which we're learning
# #         rollout_length:     length of the episode
# #         """
        
# #         states, actions, rewards, next_states = [], [], [], []
    
# #         if gym:
# #             state = env.s
# #         else:
# #             state = env.state

# #         if self.t_max == 1:
# #             #print('equal 1')
# #             rollout_length_mlmc = 1

# #         else:
        
# #             geom_draw = (np.random.geometric(p=0.50, size=1)[0])
# #             #print(f'log tmax = {np.log2(self.t_max)}')
# #             #print(geom_draw)
            
# #             # while geom_draw > np.log2(self.t_max):
                
# #             #     geom_draw = (np.random.geometric(p=0.50, size=1)[0])
# #                 #print(geom_draw)
# #             #print(self.t_max)
# #             #print(geom_draw)
# #             rollout_length_mlmc =  2**geom_draw #2**(geom(1/2)) 
# #         #rollout_length = 10
# #         #print('rollout length set')
# #         for _ in range(rollout_length_mlmc):
            
# #             states.append(state)
# #             #print('state appended', state)
# #             #print('state', state)
# #             action = self.pi.sample(state)
# #             #print('action sampled', action)
# #             #print('action', action)
# #             actions.append(action)
# #             if gym:
# #                 state, reward, done, _ , _ = env.step(action)
# #             else:
# #                 state, reward, done, _ = env.step(action)
# #             #print('env stepped', state, reward, done)
# #             rewards.append(reward)
# #             #print('reward', reward)
# #             next_states.append(state)
# #             #print('next state appended', state)
# #             if done:
# #                 break

# #         #next_states.append(env.reset())


# #         mean_rewards = np.mean(rewards) #\eta
# #         #print('mean taken')

# #         #self.mu_c = (1 - self.tau) * self.mu_c + self.tau * mean_rewards

# #         r_grads, v_grads, pi_grads = [], [], []

# #         #v_gradient, pi_gradient = 0, 0
# #         for state, action, reward, next_state in zip(states, actions, rewards, next_states):


# #             td = float(reward) - self.r + self.v(next_state) - self.v(state)
            
# #             r_gradient = float(reward) - self.r

# #             v_gradient = td * self.v.gradient(state) 

# #             pi_gradient = td * self.pi.grad_log_policy(state, action)

# #             r_grads.append(r_gradient)
# #             v_grads.append(v_gradient)
# #             pi_grads.append(pi_gradient)


# #         reward_mlmc = self.MLMC(r_grads)

# #         self.reward_grad_sum += (np.linalg.norm(reward_mlmc)**2)
# #         x = np.sqrt(self.reward_grad_sum)
# #         if x == 0:
# #             x = 1

# #         self.r = self.r + self.value_lr/x * reward_mlmc
# #         #print('reward', x)
# #         #self.r = self.r + self.value_lr * reward_mlmc
# #         #print('r updated')


# #         value_mlmc = self.MLMC(v_grads)
# #         self.value_grad_sum += (np.linalg.norm(value_mlmc)**2)
# #         x = np.sqrt(self.value_grad_sum)
# #         if x == 0:
# #             x = 1

# #         #print('value', x)
# #         self.v.params = self.v.params + self.value_lr/x * value_mlmc
# #         #self.v.params = self.v.params + self.value_lr * value_mlmc
        
# #         #print('v updated')
# #         pi_mlmc = self.MLMC(pi_grads)

# #         self.pi_grad_sum += (np.linalg.norm(pi_mlmc)**2)

# #         #print('pi', self.pi_grad_sum)
# #         #print('pi lr', self.policy_lr/np.sqrt(self.pi_grad_sum))
# #         x = np.sqrt(self.pi_grad_sum)
# #         if x == 0:
# #             x = 1

# #         #print(x)
  
# #         self.pi.params = self.pi.params + self.policy_lr/x * pi_mlmc
# #         #self.pi.params = self.pi.params + self.policy_lr * pi_mlmc
# #         #print('pi updated')
# #         self.iteration_num += 1
# #         self.number_of_samples += rollout_length_mlmc

# #         if self.adaptive_traj and (self.iteration_num % 100 == 0):
# #             current_entropy = get_avg_entropy(self.num_states, self.pi)
# #             #print('current_entropy', current_entropy)
# #             x = map_entropy(self.initial_entropy, self.initial_t_max_log2, self.max_iters/self.range_divider, current_entropy)
# #             #print(x)
# #             if 2**x != self.t_max:
# #                 self.t_max = (2**x)
# #                 print('new t_max', 2**x, self.iteration_num)


# #         # if self.adaptive_traj and (self.iteration_num == 10000):

# #         #     self.t_max *= 2
# #         #     print('new tmax', self.t_max)

# #         return mean_rewards, np.linalg.norm(pi_mlmc), rollout_length_mlmc, self.pi, self.number_of_samples, done


# # class TabularMAG(MAG):
# #     """
# #     Agent for training an actor-critic algorithm to minimize cost
# #     in the tabular setting.
# #     States and actions must be scalar.
# #     """

# #     def __init__(self, num_states, num_actions,
# #                  policy_lr, value_lr,
# #                  policy_cov_constant=1.0,
# #                  value_cov_constant=1.0,
# #                  mu_c_init=0.0, tau=0.01,
# #                  entropy_eps=0.0001,
# #                  kappa=0.0, t_max = 1, adaptive_traj = 0, iter_change = 5000, max_iters = 4e5,
# #                  range_divider = 2500):
    
# #         states = list(range(num_states))
# #         actions = list(range(num_actions))
# #         policy = SoftmaxPolicyTabular(
# #             states, actions, policy_cov_constant)
# #         value_function = LinearApproximatorOneToOneBasisV(
# #             states, init_cov_constant=value_cov_constant)

# #         super().__init__(num_states, num_actions,
# #                          policy, value_function,
# #                          policy_lr, value_lr,
# #                          mu_c_init=mu_c_init, tau=tau,
# #                          entropy_eps=entropy_eps,
# #                          kappa=kappa, t_max = t_max, 
# #                          adaptive_traj = adaptive_traj, 
# #                          iter_change = iter_change, max_iters = max_iters, range_divider = range_divider)


# # def chooseAC(ac_type, num_states, num_actions, policy_lr, value_lr, t_max, rollout_length, adaptive_traj, iterations, max_traj):
# #     if ac_type == 'MLMC':
# #         print('MLMC model')
# #         print(f'tmax {t_max}')
# #         return ACTabularMLMC(num_states, num_actions, policy_lr, value_lr, t_max = t_max, adaptive_traj = adaptive_traj, max_iters= iterations, max_traj = max_traj)
# #     # elif ac_type == 'MAG':
# #     #     print(f'tmax {t_max}')
# #     #     ac = TabularMAG(num_states, num_actions, policy_lr, value_lr, t_max = t_max, adaptive_traj = adaptive_traj, max_iters = iterations, max_traj = max_traj)
# #     else:
# #         print(f'rollout length {rollout_length}')
# #         return VanillaACTabularReward(num_states, num_actions, policy_lr, value_lr, rollout_length = rollout_length, adaptive_traj = adaptive_traj, max_iters = iterations, max_traj = max_traj)

def MLMC(self, gradients):
    """
    Multi-level gradient estimator
    """
    rollout_length = len(gradients)

    if rollout_length == 1:
        return gradients[0]
    g1 = np.mean(gradients)
    #print('g1', g1)
    g2 = np.mean(gradients[:rollout_length//2])
    #print('g2', g2)
    return gradients[0] + (rollout_length)*(g1 - g2)

