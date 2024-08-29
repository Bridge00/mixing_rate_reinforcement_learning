import torch
import numpy as np
import os
import yaml
from copy import deepcopy
#import wesutils
from scipy import integrate

from utils import (
    MeasureEstimator,
    SoftmaxPolicyTabular,
    DensityEstimatorGenerator,
    LinearApproximatorOneToOneBasisV
)










### Actor-critic versions of the above algorithms



class VanillaACBASEReward:
    """
    Agent for training an actor-critic algorithm to minimize average reward.

    This is a BASE CLASS for this type of agent. A policy must be specified.
    """

    def __init__(self, num_states, num_actions,
                 policy, value_function,
                 policy_lr, value_lr,
                 mu_c_init=0.0, tau=0.01,
                 entropy_eps=0.0001,
                 kappa=0.0):

        self.num_states = num_states
        self.num_actions = num_actions
        self.pi = policy
        self.v = value_function
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.tau = tau
        self.mu_c = mu_c_init
        self.entropy_eps = entropy_eps
        self.kappa = kappa
        
        self.reinit_params()

    def reinit_params(self):
        self.pi.reinit_params()
        self.v.reinit_params()

    def update(self, env, rollout_length):
        """
        Perform a single rollout and corresponding gradient update.

        Parameters
        ----------
        env:                gym environment on which we're learning
        rollout_length:     length of the episode
        """

        states, actions, rewards, next_states = [], [], [], []
     

        state = env.state

        for _ in range(rollout_length):
         
            states.append(state)
            action = self.pi.sample(state)
            actions.append(action)
            state, rewards, done, _ = env.step(action)
            rewards.append(rewards)
            next_states.append(state)

            if done:
                break

        next_states.append(env.reset())

 
        mean_rewards= np.mean(rewards)

        self.mu_c = (1 - self.tau) * self.mu_c + self.tau * mean_rewards

        v_gradient, pi_gradient = 0, 0
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):

            td = float(reward) - mean_rewards + self.v(next_state) - self.v(state)

            v_gradient += td * self.v.gradient(state)
            pi_gradient += td * self.pi.grad_log_policy(state, action)

        v_gradient /= len(states)
        pi_gradient /= len(states)

        self.v.params = self.v.params + self.value_lr * v_gradient
        self.pi.params = self.pi.params + self.policy_lr * pi_gradient

        return mean_rewards


class VanillaACTabularReward(VanillaACBASEReward):
    """
    Agent for training an actor-critic algorithm to minimize cost
    in the tabular setting.

    States and actions must be scalar.
    """

    def __init__(self, num_states, num_actions,
                 policy_lr, value_lr,
                 policy_cov_constant=1.0,
                 value_cov_constant=1.0,
                 mu_c_init=0.0, tau=0.01,
                 entropy_eps=0.0001,
                 kappa=0.0):
    
        states = list(range(num_states))
        actions = list(range(num_actions))
        policy = SoftmaxPolicyTabular(
            states, actions, policy_cov_constant)
        value_function = LinearApproximatorOneToOneBasisV(
            states, init_cov_constant=value_cov_constant)

        super().__init__(num_states, num_actions,
                         policy, value_function,
                         policy_lr, value_lr,
                         mu_c_init=mu_c_init, tau=tau,
                         entropy_eps=entropy_eps,
                         kappa=kappa)






class VanillaACBASERewardMLMC:
    """
    Agent for training an actor-critic algorithm to minimize average reward.

    This is a BASE CLASS for this type of agent. A policy must be specified.
    """

    def __init__(self, num_states, num_actions,
                 policy, value_function,
                 policy_lr, value_lr,
                 mu_c_init=0.0, tau=0.01,
                 entropy_eps=0.0001,
                 kappa=0.0, t_max = 100):

        self.num_states = num_states
        self.num_actions = num_actions
        self.pi = policy
        self.v = value_function
        self.r = 0
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.tau = tau
        self.mu_c = mu_c_init
        self.entropy_eps = entropy_eps
        self.kappa = kappa
        self.t_max = t_max
        self.reinit_params()

    def reinit_params(self):
        self.pi.reinit_params()
        self.v.reinit_params()

    def MLMC(values):
        
        mlmc = values[0]
        if len(values) <= self.t_max:
            mlmc += (len(values)*(np.mean(values[len(values)/2:])))
        return mlmc

    def update(self, env):
        """
        Perform a single rollout and corresponding gradient update.

        Parameters
        ----------
        env:                gym environment on which we're learning
        rollout_length:     length of the episode
        """
        
        states, actions, rewards, next_states = [], [], [], []
       

        state = env.state
        
        rollout_length = 2**(geom(1/2)) 

        for _ in range(rollout_length):
            
            states.append(state)
            action = self.pi.sample(state)
            actions.append(action)
            state, rewards, done, _ = env.step(action)
            rewards.append(rewards)
            next_states.append(state)

            if done:
                break

        next_states.append(env.reset())


        #mean_rewards= np.mean(rewards) #\eta

        #self.mu_c = (1 - self.tau) * self.mu_c + self.tau * mean_rewards

        r_grads, v_grads, pi_grads = [], [], []

        #v_gradient, pi_gradient = 0, 0
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):


            r_gradient = float(reward) - self.r

            td = r_gradient + self.v(next_state) - self.v(state)

            v_gradient = td * self.v.gradient(state) #Check

            pi_gradient = td * self.pi.grad_log_policy(state, action)

            r_grads.append(r_gradient)
            v_grads.append(v_gradient)
            pi_grads.append(pi_gradient)
        

         
        # v_gradient /= len(states)
        # pi_gradient /= len(states)
        self.r = self.r + self.value_lr * MLMC(r_grads)
        self.v.params = self.v.params + self.value_lr * MLMC(v_grads)
        self.pi.params = self.pi.params + self.policy_lr * MLMC(pi_grads)

        return mean_rewards


class VanillaACTabularRewardMLMC(VanillaACBASERewardMLMC):
    """
    Agent for training an actor-critic algorithm to minimize cost
    in the tabular setting.

    States and actions must be scalar.
    """

    def __init__(self, num_states, num_actions,
                 policy_lr, value_lr,
                 policy_cov_constant=1.0,
                 value_cov_constant=1.0,
                 mu_c_init=0.0, tau=0.01,
                 entropy_eps=0.0001,
                 kappa=0.0):
    
        states = list(range(num_states))
        actions = list(range(num_actions))
        policy = SoftmaxPolicyTabular(
            states, actions, policy_cov_constant)
        value_function = LinearApproximatorOneToOneBasisV(
            states, init_cov_constant=value_cov_constant)

        super().__init__(num_states, num_actions,
                         policy, value_function,
                         policy_lr, value_lr,
                         mu_c_init=mu_c_init, tau=tau,
                         entropy_eps=entropy_eps,
                         kappa=kappa)