import gym
import numpy as np
from itertools import product
from copy import deepcopy
import os
from inforatio.utils import OneToOneBasisV


class Toy1D(gym.Env):
    """
    A one-dimensional toy environment from Matheron et al.'s 2019 paper,
    'The Problem with DDPG'.
    """

    def __init__(self):

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(1,))

        super().__init__()

    def reset(self):
        """
        Reset state to 0 and return it.
        """

        self.state = self.observation_space.low

        return self.state

    def step(self, action):
        """
        Take an action, compute reward, transition to next state.

        Parameters
        ----------
        action: an element of the action space
        """

        assert action in self.action_space, 'action must be in action_space'

        next_state = self.state + action
        reward = int(next_state < 0)
        self.state = np.clip(next_state, 0, 1)

        return self.state, reward, bool(reward), {}


class DiscreteToy1D:
    """
    Discretized version of the one-dimensional toy environment from
    Matheron et al.'s 2019 paper, 'The Problem with DDPG'.

    Parameters
    ----------

    state_steps:        number of discretization steps for the state space
    action_steps:       number of discretization steps for the action space
    """

    def __init__(self, num_states, num_actions):

        self.orig_env = Toy1D()
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(num_actions)

    def _discretize_state(self, state):

        if state < 0:
            return np.array([0])

        bins = self.observation_space.n - 2
        return 1 + np.floor(bins * state)

    def _undiscretize_action(self, action):

        bins = self.action_space.n
        return -0.1 + (0.2 / (bins - 1)) * np.array([action])

    def reset(self):

        self.state = self._discretize_state(self.orig_env.reset())

        return self.state

    def step(self, action):

        assert action in self.action_space, "action must be in action_space"

        action = self._undiscretize_action(action)
        state, reward, done, _ = self.orig_env.step(action)
        state = self._discretize_state(state)

        return state, reward, done, {}


class DiscreteToy1DCost(DiscreteToy1D):
    """
    Cost (not reward) version of DiscreteToy1D.
    """

    def __init__(self, num_states, num_actions,
                 cost_fn=lambda reward: 1 + (1 - reward) * 1000):

        super().__init__(num_states, num_actions)
        self.cost_fn = cost_fn

    def step(self, action):

        state, reward, done, info = super().step(action)

        return state, self.cost_fn(reward), done, info


class DummyEnv(gym.Env):
    """
    Dummy environment for use in debugging.

    Parameters
    ----------
    num_states:         number of states
    cost_fn:            cost function to use; default just returns 1
    """

    def __init__(self, num_states=5, cost_fn=lambda s, a: 1):

        self.num_states = num_states
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(3)
        self.cost_fn = cost_fn
        
        super().__init__()

    def reset(self):
        self.state = np.array([self.observation_space.sample()], dtype=np.float32)

        return self.state

    def step(self, action):
        """
        Move left if 0, right if 2, stay put if 1.
        """

        action = int(action)
        assert action in self.action_space, "Action must be in action_space"

        reward = self.cost_fn(self.state, action)
        action -= 1
        self.state = np.clip(self.state + action, 0, self.observation_space.n - 1)

        return self.state, reward, False, {}


class MDPEnv(gym.Env):
    """
    A Gym-compatible environment that takes fully-specified average-cost MDPs.
    """
    
    def __init__(self, states, actions, transition_probabilities, costs):
        """
        Parameters
        ----------
        states:                   a list of states 
        actions:                  a list of actions (*descriptions* of actions)
        transition_probabilities: a dictionary that returns a state distribution
                                  for a given (state, action) pair
        costs:                    a numpy array of dimension num_states by num_actions
                                  specifying costs for each state-action pair
        """
        self.states                   = {s: i for i, s in enumerate(states)}
        self.actions                  = {a: i for i, a in enumerate(actions)}
        self.costs                    = self._array_to_fun(costs)
        self.transition_probabilities = self._dict_to_fun(transition_probabilities)
        
        self.observation_space = gym.spaces.Discrete(len(states))
        self.action_space      = gym.spaces.Discrete(len(actions))

    def _array_to_fun(self, array):
        """
        Wraps a function around an array.

        Parameters
        ----------

        array:      a 2D numpy array

        Returns
        -------
        f:  a function

        f(s,a) = array[s][a]
        """

        return lambda s, a: array[s][a]
        
    def _dict_to_fun(self, dictionary):
        """
        Wraps a function around a dictionary.

        Parameters
        ----------

        dictionary: a dict

        Returns
        -------
        f: a function

        f(a,b,c,...) == X if and only if dictionary[(a,b,c,...)] == X
        """
        if callable(dictionary):
            return dictionary
        else:
            return lambda *keys: dictionary[keys]

    def step(self, action):
        """
        Parameters
        ----------
        action: an element of the action space
        """
        action = self.actions[action]
        cost = self.costs(self.state, action)
        distribution = self.transition_probabilities(self.state, action)
        self.state = np.random.choice(self.observation_space.n, p=distribution)

        return self.state, cost, False, {}

    def reset(self):
        """
        """
        self.state = self.observation_space.sample()

        return self.state


class RandomMDPEnv(MDPEnv):
    """
    Takes a matrix of costs and generates an MDPEnv with random transition
    probabilities.
    """

    def __init__(self, num_states, num_actions,
                 costs=None,
                 transition_seed=None, training_seed=None):

        assert costs is not None, "Specify costs"

        costs = np.array(costs).reshape(num_states, num_actions)
        
        np.random.seed(transition_seed)
    
        states  = [s for s in range(num_states)]
        actions = [a for a in range(num_actions)]
    
        probs = {}
        for elem in product(states, actions):
            dist = np.random.random(num_states)
            probs[elem] = dist / np.sum(dist)

        def transition_matrix(state, action):
            return probs[(state, action)]
    
        np.random.seed(training_seed)
    
        super().__init__(
            states, 
            actions, 
            transition_matrix, 
            costs
        )


class EasyMDPEnv(MDPEnv):
    """
    Takes a 1D array of costs and 1D array of transition probabilities.
    Creates a corresponding MDP.
    """

    def __init__(self, num_states, num_actions,
                 costs, probs):

        costs = np.array(costs).reshape(num_states, num_actions)
        probs = np.array(probs).reshape(num_states * num_actions,
                                        num_states)

        states  = [s for s in range(num_states)]
        actions = [a for a in range(num_actions)]

        probs_dict = {}
        for i, elem in enumerate(product(states, actions)):
            probs_dict[elem] = probs[i]

        def transition_matrix(state, action):
            return probs_dict[(state, action)]

        super().__init__(
            states,
            actions,
            transition_matrix,
            costs
        )


class GridWorldEnvBASEReward:
    """
    Representation of the classic gridworld environment.

    This is a BASE class. The reward function must be specified!

    States are represented as a rectangular table with values 0 for open,
    1 for blocked, 2 for start, and 3 for goal. Actions are 'up', 'down',
    'left', 'right'. If the agent is in a state s marked 0 and chooses an
    action moving off grid or bumping into a blocked state, it remains in s.
    Otherwise, it moves to the state on the grid corresponding to the action
    it chose.

    Parameters
    ----------

    gridfile:   location of *.npy file containing the grid
    dims:       tuple (n, m) representing horizontal, vertical grid dimensions
    start:      start state indices (x, y)
    goal:       goal state indices (x, y)
    blocked:    tuple or list [(x_1, y_1), ..., (x_k, y_k)] of blocked square indices
    reward_fn:  reward function taking state tuple s, action a, returning c(s,a)
    allow_done: bool specifying whether to return done=True when goal is reached
    """

    def __init__(self, reward_fn, gridfile=None,
                 dims=None,
                 random_start=False, start=None, goal=None, blocked=None,
                 allow_done=False, max_steps=np.inf):

        if gridfile is not None:
            self.grid = np.load(gridfile)
            self.start = tuple(int(index) for index in np.where(self.grid == 2))
            self.goal = tuple(int(index) for index in np.where(self.grid == 3))
            blocked = list(list(elem) for elem in np.where(self.grid == 1))
            self.blocked = tuple(zip(blocked[0], blocked[1]))

        else:
            assert None not in [dims, start, goal, blocked], 'Specify all'

            self.grid = np.zeros(shape=dims)
            self.start = start
            self.goal = goal
            self.blocked = tuple(tuple(elem) for elem in blocked)
            self.grid[start] = 2
            self.grid[goal] = 3
            for indices in self.blocked:
                self.grid[indices] = 1

        self.random_start = random_start
        self.dims = self.grid.shape

        self.height, self.width = self.grid.shape

        self.reward_fn = reward_fn
        self.actions = {'up', 'down', 'left', 'right', 'stay'}
        self.allow_done = allow_done
        self.max_steps = max_steps

    def reset(self):
        if self.random_start:
            state = None
            while state is None:
                i = np.random.randint(0, self.dims[0])
                j = np.random.randint(0, self.dims[1])
                if self.grid[i, j] != 1.0:
                    state = (i, j)
        else:
            state = self.start
        self._state = state
        self.curr_step = 0
        return self.state

    @property
    def state(self):
        return self._state, self.grid[self._state]

    @property
    def done(self):
        if self.allow_done and (self.curr_step >= self.max_steps):
            done = True
        else:
            done = False
        return done

    def _available_actions(self, indices):
        available_actions = set()
        available_actions.add('stay')
        i, j = indices

        assert (0 <= i <= self.height - 1) and (0 <= j <= self.width - 1), \
                'Invalid indices'

        # check if 'up' is available
        if (i > 0) and (self.grid[(i - 1, j)] != 1):
            available_actions.add('up')

        # check if 'down' is available
        if (i < self.height - 1) and (self.grid[(i + 1, j)] != 1):
            available_actions.add('down')

        # check if 'left' is available
        if (j > 0) and (self.grid[(i, j - 1)] != 1):
            available_actions.add('left')

        # check if 'right' is available
        if (j < self.width - 1) and (self.grid[(i, j + 1)] != 1):
            available_actions.add('right')

        return available_actions

    def _transition(self, action):
        if action not in self._available_actions(self._state):
            pass
        else:
            i, j = self._state
            if action == 'up':
                self._state = (i - 1, j)
            if action == 'down':
                self._state = (i + 1, j)
            if action == 'left':
                self._state = (i, j - 1)
            if action == 'right':
                self._state = (i, j + 1)

    def _force_transition(self, state, action):
        """
        Manually set the current state, then transition according to action.
        """
        self._state = state[0]
        self._transition(action)
        return self.state

    def step(self, action):

        self.curr_step += 1
        prev_state = self.state
        self._transition(action)

        return self.state, self.reward_fn(prev_state, action), self.done, {}


class GridWorldEnvReward(GridWorldEnvBASEReward):
    """
    Representation of the classic gridworld environment.

    States are represented as a rectangular table with values 0 for open,
    1 for blocked, 2 for start, and 3 for goal. Actions are 'up', 'down',
    'left', 'right'. If the agent is in a state s marked 0 and chooses an
    action moving off grid or bumping into a blocked state, it remains in s.
    Otherwise, it moves to the state on the grid corresponding to the action
    it chose.

    Parameters
    ----------

    gridfile:   location of *.npy file containing the grid
    dims:       tuple (n, m) representing horizontal, vertical grid dimensions
    start:      start state indices (x, y)
    goal:       goal state indices (x, y)
    blocked:    tuple or list [(x_1, y_1), ..., (x_k, y_k)] of blocked square indices
    rewards:      tuple (r_G, r_0, r_1), goal reward r_G, reward r_0 of choosing
                    available action, reward r_1 of choosing unavailable action
    allow_done: bool specifying whether to return done=True when goal is reached
    """

    def __init__(self, gridfile=None, dims=None,
                 random_start=False, start=None, goal=None,
                 blocked=None, rewards=(1, 10, 100),
                 allow_done=False, max_steps=np.inf):

        rG, r0, r1 = rewards
        def reward_fn(state, action):
            indices, state_type = state
            available_actions = self._available_actions(indices)
            if (state_type == 3) and (action in available_actions):
                return cG
            return c0 if action in available_actions else c1

        super().__init__(reward_fn, gridfile=gridfile,
                         dims=dims, random_start=random_start,
                         start=start, goal=goal, blocked=blocked,
                         allow_done=allow_done, max_steps=max_steps)





class GridWorldGymEnvReward(gym.Env):
    """
    Gym-compliant version of the classic gridworld environment.

    States are represented as a rectangular table with values 0 for open,
    1 for blocked, 2 for start, and 3 for goal. Actions are 'up', 'down',
    'left', 'right'. If the agent is in a state s marked 0 and chooses an
    action moving off grid or bumping into a blocked state, it remains in s.
    Otherwise, it moves to the state on the grid corresponding to the action
    it chose.

    Parameters
    ----------

    gridfile:   location of *.npy file containing the grid
    dims:       list [n, m] representing horizontal, vertical grid dimensions
    start:      start state indices [x, y]
    goal:       goal state indices [x, y]
    blocked:    tuple or list [x_1, y_1, ..., x_k, y_k] of blocked square indices
    rewards:      list [r_G, r_0, r_1], goal reward r_G, reward r_0 of choosing
                    available action, reward r_1 of choosing unavailable action
    allow_done: bool specifying whether to return done=True when goal is reached
    """

    def __init__(self, gridfile=None, dims=None,
                 random_start=False,
                 start=None, goal=None,
                 blocked=None, rewards=(100, 10, 1), numpy_state=False,
                 allow_done=False, max_steps=np.inf):

        if gridfile is None:
            assert None not in [dims, start, goal, blocked], 'Specify environment'
            assert len(blocked) % 2 == 0, 'blocked must be even'

            dims, start, goal, rewards = tuple(dims), tuple(start), tuple(goal), tuple(rewards)
            blocked = np.array(blocked).reshape(len(blocked) // 2, 2).tolist()

        self._env = GridWorldEnvReward(gridfile=gridfile, dims=dims,
                                 random_start=random_start,
                                 start=start, goal=goal, blocked=blocked,
                                 rewards=rewards, allow_done=allow_done,
                                 max_steps=max_steps)
        self.numpy_state = numpy_state
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Discrete(np.product(self._env.grid.shape))
        self.action_space = gym.spaces.Discrete(5)

        self.state_to_repr = {
            state: i for i, state in enumerate(zip(product(
                range(self._env.height), range(self._env.width)),
                self._env.grid.flatten()))
        }
        self.repr_to_state = {v: k for k, v in self.state_to_repr.items()}

        self.action_to_repr = {
            'stay':     0,
            'up':       1,
            'down':     2,
            'left':     3,
            'right':    4
        }
        self.repr_to_action = {v: k for k, v in self.action_to_repr.items()}

    def _numpy_state(self, state):
        state_repr = self.state_to_repr[state]
        if self.numpy_state:
            state_repr = np.array([state_repr], np.float32)
        return state_repr

    @property
    def state(self):
        return self._numpy_state(self._env.state)

    def reset(self):
        return self._numpy_state(self._env.reset())

    def step(self, action):
        state, rewards, done, _ = self._env.step(self.repr_to_action[int(action)])
        return self._numpy_state(state), rewards, done, {}

    def get_c_and_p(self):

        num_actions = self.action_space.n
        num_states = self.observation_space.n

        c = [self._env.rewards_fn(state, action) for
             state, action in product(self.state_to_repr.keys(),
                                      self.action_to_repr.keys())]
        c = np.array(c).reshape(num_states, num_actions)

        p = np.zeros(shape=(num_states * num_actions, num_states))

        for i, (state, action) in enumerate(product(
            range(num_states), range(num_actions))):

            state = self.repr_to_state[int(state)]
            action = self.repr_to_action[int(action)]
            new_state = self._env._force_transition(state, action)
            new_state = self.state_to_repr[new_state]

            p[i, new_state] = 1

        return c, p



class GridWorldGymEnvFrames(GridWorldGymEnv):
    """
    Gym-compliant version of the classic gridworld environment using grid
    frames to represent the state.

    States are represented as a rectangular grid with values 0 for open,
    1 for blocked, 2 for start, 3 for goal, and 4 for agent location.

    Actions are 'up', 'down', 'left', 'right'.

    If the agent is in a state s marked 0 and chooses an action moving off
    grid or bumping into a blocked state, it remains in s. Otherwise, it moves
    to the state on the grid corresponding to the action it chose.

    Parameters
    ----------

    gridfile:   location of *.npy file containing the grid
    dims:       list [n, m] representing horizontal, vertical grid dimensions
    start:      start state indices [x, y]
    goal:       goal state indices [x, y]
    blocked:    tuple or list [x_1, y_1, ..., x_k, y_k] of blocked square indices
    costs:      list [c_G, c_0, c_1], goal cost c_G, cost c_0 of choosing
                    available action, cost c_1 of choosing unavailable action
    allow_done: bool specifying whether to return done=True when goal is reached
    """

    def __init__(self, gridfile=None, dims=None,
                 random_start=False, start=None, goal=None,
                 blocked=None, costs=(1, 10, 100),
                 numpy_state=False, allow_done=False,
                 max_steps=500):

        super().__init__(gridfile=gridfile, dims=dims,
                         random_start=random_start,
                         start=start, goal=goal,
                         blocked=blocked, costs=costs, numpy_state=numpy_state,
                         allow_done=allow_done, max_steps=max_steps)

        self.observation_space = gym.spaces.Box(0.0, 4.0, dims)

        self.state_to_repr = dict()
        for i, state in enumerate(zip(product(
            range(self._env.height), range(self._env.width)),
            self._env.grid.flatten())):
            if state[1] != 1.0:
                new_grid = deepcopy(self._env.grid)
                new_grid[state[0]] = 4.0
                self.state_to_repr[state] = new_grid

        class BytesDict:
            def __init__(self):
                self._d = dict()

            def __getitem__(self, key):
                key = bytes(key)
                return self._d[key]
            
            def __setitem__(self, key, value):
                key = bytes(key)
                self._d[key] = value

        self.repr_to_state = BytesDict()
        for k, v in self.state_to_repr.items():
            self.repr_to_state[v] = k

    def _numpy_state(self, state):
        state_repr = self.state_to_repr[state]
        if self.numpy_state:
            state_repr = np.array([state_repr], np.float32)
        return state_repr # .reshape(1, 1, self._env.height, self._env.width)



### Modified FrozenLake environment

import importlib

def module_from_path(module_name, module_path):
    """Returns a module from an absolute path to the module."""

    spec = importlib.util.spec_from_file_location(module_name,
                                                  module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

frozen_lake_file = deepcopy(*gym.envs.__path__) + '/toy_text/frozen_lake.py'
frozen_lake_path = os.path.abspath(frozen_lake_file)
FrozenLakeEnv = module_from_path('frozen_lake', frozen_lake_path).FrozenLakeEnv

class FrozenLakeCostEnv(FrozenLakeEnv):
    """
    Modification of FrozenLakeEnv that returns positive costs instead of rewards.

    Passing rewards=True multiplies the costs by -1.
    """

    def __init__(self, max_cost=2.0, max_steps=None, rewards=False):
        FrozenLakeEnv.__init__(self)
        self.max_cost = max_cost
        self.max_steps = max_steps
        self.rewards = (-1)**int(rewards)

    def reset(self):
        self.num_steps = 0
        state = FrozenLakeEnv.reset(self)
        return np.array(state, ndmin=1)

    def step(self, action):
        state, reward, done, d = FrozenLakeEnv.step(self, action)
        self.num_steps += 1
        cost = self.max_cost - reward
        if self.max_steps is not None:
            done = True if (self.num_steps >= self.max_steps) else False

        return np.array(state, ndmin=1), self.rewards * cost, done, d


def n_to_grid_trans(n):
    
    tp = {}
    
    for state in range(n**2):
        for action in range(5):
            tp[(state, action)] = [0] * (n**2)
            if action == 0:
                tp[(state, action)][state] = 1
            
            if action == 1:  #Up
                if state - n > -1:
                    tp[(state, action)][state-n] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 2: # Down
                if state + n < n**2:
                    tp[(state, action)][state+n] = 1
                else:
                    
                    tp[(state, action)][state] = 1
                    
            if action == 3: # Left
                if (state - 1) % n < n-1:
                    tp[(state, action)][state-1] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 4: #Right
                if (state + 1) % n > 0:
                    tp[(state, action)][state+1] = 1   
                else:
                    tp[(state, action)][state] = 1
    return tp


def n_to_rewards(n, g, value):
    rewards = [[value] * 5] * (n**2)
    #print(rewards)

    #print(rewards[state])

    for state in range(n**2):
        if state == g:
            #print('up')
            rewards[state] = [1, value, value, value, value]   
            
        if state - n == g:
            #print('up')
            rewards[state] = [value, 1, value, value, value]
        if state + n == g:
            #print('down')
            rewards[state] = [value, value, 1, value, value]
        if state - 1 == g:
            #print('left')
            rewards[state] = [value, value, value, 1, value]
        if state + 1 == g:
            #print('right')
            rewards[state] = [value, value, value, value, 1]
            
    return rewards

class GridWorld(MDPEnv):


    def __init__(self, states, actions):
        self.state = 0

        tp = n_to_grid_trans(states)
        rewards = n_to_rewards(states, states**2 - 1, 0)

        super().__init__(list(range(states**2)), list(range(actions)), tp, rewards)




