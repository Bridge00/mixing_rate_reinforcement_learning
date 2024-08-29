import numpy as np
from collections import defaultdict, deque
from itertools import product
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import math
import torch

def estimate_loss(gradients, is_mlmc, traj_parameter=None, is_tensor = True):
    if is_mlmc:
          
        if 1 < len(gradients) <= (2**traj_parameter):
            if is_tensor:
                g1 = torch.mean(torch.stack(gradients))
    
                g2 = torch.mean(torch.stack(gradients[:len(gradients)//2]))
            else:
                g1 = np.mean(gradients)
                g2 = np.mean(gradients[:len(gradients)//2])
       
            return gradients[0] + len(gradients)*(g1 - g2)  

        return gradients[0] 

    
    if is_tensor:
        return torch.mean(torch.stack(gradients))
    return np.mean(gradients)



def advantage_estimation(states, actions, rewards, N, s, a, pi_a_s):

    i = 0
    tau = 0
    taus = []
    ys = []
    #print(rewards)
    #print(states)
    while tau < len(states) - N:
        #print(states[tau], s)
        #if states[tau].item() == s:
        if torch.equal(states[tau], s):
            i += 1

            taus.append(tau)
            ys.append(np.sum([rewards[t] for t in range(int(tau), int(tau + N-1))]))
            tau += int(2*N)
            #print(tau)
        else:
            tau += int(1)

    if i > 0:
        #print('rewards', ys)
        V_est = np.mean(ys)
        #print(a)
        cond_sum = 0
        for index, y_i in enumerate(ys):
            if actions[taus[index]].item() == a:
                cond_sum += y_i

        # print('pi(a|s)',pi_a_s)
        # print('1/pi(a|s)', 1/pi_a_s)
        Q_est = 1/pi_a_s * 1/len(ys) * cond_sum
        # print('Q_est', Q_est)

        return Q_est - V_est
    else:
        return 0


class DensityEstimatorGenerator:
    """
    Generates kernel density estimators of the occupancy measure of a
    continuous-state Markov chain.

    Parameters
    ----------

    maxlen:         maximum number of samples to store and use in estimator
    kernel:         kernel type to use, default is 'gaussian'
    bandwidth:      bandwidth parameter
    """

    def __init__(self, maxlen=1000,
                 kernel='gaussian', **kwargs):
        self._samples = deque(maxlen=maxlen)
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        self._grid = GridSearchCV(KernelDensity(), params)

    def __len__(self):
        return len(self._samples)

    def append(self, sample):
        self._samples.append(sample)

    def extend(self, samples):
        self._samples.extend(samples)

    def clear(self):
        self._samples.clear()

    def get_density(self):
        assert len(self._samples) > 0, 'must add samples before calling'

        num_samples = len(self._samples)
        sample_dim = 1 if not isinstance(self._samples[0], np.ndarray) \
                else len(self._samples[0])
        samples = np.array(self._samples).reshape(num_samples, sample_dim)
        self._grid.fit(samples)

        kde = self._grid.best_estimator_

        return lambda x: np.exp(kde.score(x.reshape(1, sample_dim)))


class MeasureEstimator:
    """
    Running estimator of the occupancy measure of a finite-state Markov chain.

    Parameters
    ----------
    hashable_states:    stipulate whether states are hashable (numpy arrays are not,
                        hence the default is False); if not, they will be converted
                        to byte representations, which are always hashable
    """

    def __init__(self, hashable_states=False):
        self._d = defaultdict(int)
        self.num_increments = 0
        self.hashable_states = hashable_states

    def append(self, state):
        """
        Add another state to the estimator and increment the counts.
        """
        if not self.hashable_states:
            state = bytes(state)
        self._d[state] += 1
        self.num_increments += 1

    def __len__(self):
        """
        Return the number of states encountered so far.
        """
        return len(self._d)

    def __getitem__(self, state):
        """
        Return the fraction of all visits that have been made to state.
        """
        if not self.hashable_states:
            state = bytes(state)
        
        if state not in self._d:
            return 0
        
        return self._d[state] / self.num_increments

    def __call__(self, state):
        return self[state]

    @property
    def vector(self):
        # TODO: allow specifying keys to include states that went unvisited
        return np.array([self[i] for i in self._d.keys()])

    @property
    def entropy(self):
        return entropy([self[key] for key in self._d.keys()])


class SoftmaxPolicy:
    """"
    Parent class for use in softmax policies with specific
    h functions.
    """
    
    def __init__(self, actions):
        self.actions = tuple(actions) if isinstance(actions, dict) else \
                actions
        self.action_indices = {a: i for i, a in enumerate(actions)}
        self.h = None
        
    @property
    def params(self):
        return self.h.params

    @params.setter
    def params(self, new_params):
        self.h.params = new_params
        
    def reinit_params(self):
        self.h.reinit_params()

    def _hvals(self, state):
        #print(state)
        #print(self.h.params)
        return np.array([self.h(np.append(state, action))
                         for action in self.actions])
    
    def _get_probs(self, state):
        #print('self._hvals', self._hvals(state))
        return softmax(self._hvals(state) - np.min(self._hvals(state)))
        
    def sample(self, state):
        #print('sample state', state)
        probs = self._get_probs(state)
        #print('probs', probs)
        #print('actions', self.actions)
        # for x in probs:
        #     if math.isnan(x):
        #         print(probs)
        return np.random.choice(self.actions, p=probs)
    
    def _hgrads(self, state):
        return np.array([self.h.gradient(np.append(state, action))
                        for action in self.actions])
        
    def grad_log_policy(self, state, action):    
        hgrads = self._hgrads(state)
        action_index = self.action_indices[action]
        return hgrads[action_index] - np.dot(
            self._get_probs(state).flatten(), hgrads)
    
    def pdf(self, action, state):
        return self._get_probs(state)[self.action_indices[action]]


class OneToOneBasis:
    """
    Maps each state-action pair to a unique standard basis vector.

    Policies using this basis will have one parameter for each pair.
    """

    def __init__(self, states, actions):

        def basis_vector(i):
            v = np.zeros(len(states) * len(actions))
            v[i] = 1.0
            return v

        self._feature_map = {elem: basis_vector(i) for i, elem in enumerate(
            product(states, actions))}

    def __call__(self, state, action):
        return self._feature_map[(state, action)]


class LinearApproximatorOneToOneBasis:
    """
    Linear approximator using a unique standard basis vector for each
    state-action pair.
    """

    def __init__(self, states, actions, init_cov_constant=1):
        self.feature_map = OneToOneBasis(states, actions)
        self.init_cov_constant = init_cov_constant
        self.num_params = len(list(self.feature_map._feature_map.values())[0])
        self.reinit_params()

    def reinit_params(self):
        np.random.seed(0)
        #print('setting seed')
        self.params = np.random.multivariate_normal(
            mean=np.zeros(self.num_params),
            cov=self.init_cov_constant * np.eye(self.num_params))
        #print(self.params)
    def gradient(self, state_action):
        return self.feature_map(*state_action)

    def __call__(self, state_action):
        return np.dot(self.params, self.feature_map(*state_action))


class SoftmaxPolicyTabular(SoftmaxPolicy):
    """
    Linear softmax policy using unique feature vectors for each
    state-action pair.
    """

    def __init__(self, states, actions, init_cov_constant=1):
        super().__init__(actions)
        self.h = LinearApproximatorOneToOneBasis(states, actions,
                                                 init_cov_constant)


class OneToOneBasisV:
    """
    One-to-one feature mapping for states. Maps each state to a unique
    standard basis vector.

    Value functions using this representation are essentially 'tabular'.
    """

    def __init__(self, states):

        def basis_vector(i):
            v = np.zeros(len(states))
            v[i] = 1.0
            return v

        self._feature_map = {elem: basis_vector(i) 
                             for i, elem in enumerate(states)}

    def __call__(self, state):
        return self._feature_map[state]


class LinearApproximatorOneToOneBasisV:
    """
    Linear approximator for the value function using a unique standard basis
    vector for each state.
    """

    def __init__(self, states, init_cov_constant=1):
        self.feature_map = OneToOneBasisV(states)
        self.init_cov_constant = init_cov_constant
        self.num_params = len(states)
        self.reinit_params()

    def reinit_params(self):
        self.params = np.random.multivariate_normal(
            mean=np.zeros(self.num_params),
            cov=self.init_cov_constant * np.eye(self.num_params))

    def gradient(self, state):
        return self.feature_map(state)

    def __call__(self, state):
        #print('in call')
        #print('state', state)
        return np.dot(self.params, self.feature_map(state))
