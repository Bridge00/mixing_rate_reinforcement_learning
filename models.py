import torch
import torch.distributions as td
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import wesutils


class ConvolutionalNetwork(nn.Module):
    """
    Convolutional neural network containing two convolutional layers followed 
    by two fully connected layers with ReLU activation functions.
    """

    def __init__(self, in_channels, out_dim,
                 example_state_tensor,  # example state converted to torch tensor
                 out_channels1=16, kernel_size1=3, stride1=2, padding1=1,
                 in_channels2=16, out_channels2=32, kernel_size2=2, stride2=1,
                 padding2=1,
                 out_features3=256):

        super().__init__()

        example_state_tensor = self._reshape_state(example_state_tensor)

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size1,
                               stride1, padding1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size2,
                               stride2, padding2)

        in_features3 = self.conv2(
            self.conv1(example_state_tensor.to(torch.device('cpu')))).view(-1).shape[0]

        self.fc3 = nn.Linear(in_features3, out_features3, bias=True)
        self.head = nn.Linear(out_features3, out_dim, bias=True)

    def forward(self, x):
        x = self._reshape_state(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))  # flattens Conv2d output
        return self.head(x)

    def _reshape_state(self, x):
        num_elems = len(x) if len(x.shape) > 2 else 1
        dims = x.shape[-2:]
        return x.reshape(num_elems, 1, *dims)


class PolicyNetwork(nn.Module):
    """Base class for stochastic policy networks."""

    def __init__(self):
        super().__init__()

    def forward(self, state):
        """Take state as input, then output the parameters of the policy."""

        raise NotImplemented("forward not implemented.")

    def sample(self, state):
        """
        Sample an action based on the model parameters given the current state.
        """

        raise NotImplemented("sample not implemented.")

    def log_probs(self, obs, actions):
        """
        Return log probabilities for each state-action pair.
        """

        raise NotImplemented("log_probs not implemented.")

    def entropy(self, obs):
        """
        Return entropy of the policy for each state.
        """

        raise NotImplemented("entropy not implemented.")


class GaussianPolicyBase(PolicyNetwork):
    """
    Base class for Gaussian policy.

    Desired network needs to be implemented.
    """

    def __init__(self, action_dim):

        super().__init__()

        self.action_dim = action_dim
      

    def _get_covs(self, log_stds):
        batch_size = log_stds.shape[0]
        stds = log_stds.exp().reshape(batch_size, 1, 1)
        covs = stds * torch.eye(self.action_dim).repeat(batch_size, 1, 1)
        return covs

    def sample(self, obs, no_log_prob=False):
        mean, log_std = self.forward(obs)
        #print('mean in sample', mean, 'log_std in sample', log_std)
        cov = log_std.exp() * torch.eye(self.action_dim)
        #print('covariance', cov)
        dist = td.MultivariateNormal(mean, cov)
        #print('dist', dist)
        action = dist.rsample()
        #print('action sampled', action)
        return action if no_log_prob else (action, dist.log_prob(action))

    def log_probs(self, obs, actions):
        means, log_stds = self.forward(obs)
        covs = self._get_covs(log_stds)
        dists = td.MultivariateNormal(means, covs)
    
        return dists.log_prob(actions)

    def entropy(self, obs):
        means, log_stds = self.forward(obs)
        covs = self._get_covs(log_stds)
        dists = td.MultivariateNormal(means, covs)
        return dists.entropy()


class GaussianPolicy(GaussianPolicyBase):
    """
    Gaussian policy using a two-layer, two-headed MLP with ReLU activation.
    """

    def __init__(self, obs_dim, action_dim,
                 min_action_val=-1.0,
                 max_action_val=1.0,
                 hidden_layer1_size=64,
                 hidden_layer2_size=64):

        super().__init__(action_dim)

        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer1_size),
            nn.ReLU(),
            nn.Linear(hidden_layer1_size, hidden_layer2_size),
            nn.ReLU(),
        )

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, action_dim),
            nn.Hardtanh(min_action_val, max_action_val),
        )

        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, 1),
            #nn.Sigmoid()
            
            #nn.Hardtanh(0, max_action_val),
        )

    def forward(self, obs):
        #print(obs)
        x = self.base_net(obs)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        #print(log_std)
        #print('mean', mean, 'log_std', log_std)
        return mean, log_std



class GaussianPolicyNetworkBase(PolicyNetwork):
    """
    Base class for Gaussian policies.

    Desired two-headed network outputting mean and covariance needs to be
    implemented.
    """

    def __init__(self, state_dim, action_dim,
                 log_std_min=-20, log_std_max=5):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.device = torch.device('cpu')

    def set_device(self, device):
        """Set the device."""

        self.device = device
        self.to(self.device)

    def sample(self, state, no_log_prob=False):
        """
        Sample from the Gaussian distribution with mean and covariance
        output by the two-headed policy network evaluated on the current state.
        """
        #print('in sample')
        mean, cov = self.forward(state)
        #print('mean', mean, 'cov', cov)
        dist = td.multivariate_normal.MultivariateNormal(
            mean, torch.eye(self.action_dim).to(self.device) * cov)
        #print('dist', dist)
        action = dist.rsample()
        #print('action', action)
        return_val = (action, dist.log_prob(action)) if not no_log_prob else action
        #print('return_val from sample', return_val)
        return return_val


class GaussianPolicyTwoLayer(GaussianPolicyNetworkBase):
    """
    Simple two-layer, two-headed Gaussian policy network.

    If simple_cov == True, the covariance matrix always takes the form
        
        sigma * I,

    where sigma is a scalar and I is an identity matrix with appropriate
    dimensions.
    """

    def __init__(self, state_dim, action_dim,
                 min_action_val=-1.0,
                 max_action_val=1.0,
                 simple_cov=True,
                 hidden_layer1_size=256,
                 hidden_layer2_size=256,
                 activation='relu',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001):

        super().__init__(state_dim, action_dim,
                         log_std_min, log_std_max)
        print('in gauss two layer')
        self.simple_cov = simple_cov
        self.activation = eval('F.' + activation) # same activation everywhere
        
        # set the output dimension of the log_std network
        cov_output_dim = 1 if self.simple_cov else self.action_dim

        # define the network layers
        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.mean = nn.Linear(hidden_layer2_size, self.action_dim)
        self.log_std = nn.Linear(hidden_layer2_size, cov_output_dim)
        self.ht = nn.Hardtanh(min_action_val, max_action_val)

        # initialize the weights of the layers
        nn.init.normal_(self.linear1.weight, std=weight_init_std)
        nn.init.normal_(self.linear1.bias, std=weight_init_std)
        nn.init.normal_(self.linear2.weight, std=weight_init_std)
        nn.init.normal_(self.linear2.bias, std=weight_init_std)
        nn.init.normal_(self.mean.weight, std=weight_init_std)
        nn.init.normal_(self.mean.bias, std=weight_init_std)
        nn.init.normal_(self.log_std.weight, std=weight_init_std)
        nn.init.normal_(self.log_std.bias, std=weight_init_std)
        
    def forward(self, state):
        #print('forward')
        #print(self.state_dim)
        #print(state, type(state))
        x = self.activation(self.linear1(state))
        #print('first lienar and relu', x)
        x = self.activation(self.linear2(x))
        #print('second linear and relu', x)
        mean = self.mean(x)
        #print('mean', mean)
        mean = self.ht(mean)
        #print('ht', mean)
        cov = torch.clamp(self.log_std(x),
                          self.log_std_min, self.log_std_max).exp()
        #print('cov', cov)
        #cov = cov.unsqueeze(dim=2) * torch.eye(self.action_dim).to(self.device)
        cov = cov * torch.eye(self.action_dim).to(self.device)
        #print('cov unsqueeze', cov)
        return mean, cov


def two_layer_net(input_dim, output_dim,
                  hidden_layer1_size=256,
                  hidden_layer2_size=256,
                  activation='ReLU'):
    """
    Generate a fully-connected two-layer network for quick use.
    """

    activ = eval('nn.' + activation)

    net = nn.Sequential(
        nn.Linear(input_dim, hidden_layer1_size),
        activ(),
        nn.Linear(hidden_layer1_size, hidden_layer2_size),
        activ(),
        nn.Linear(hidden_layer2_size, output_dim)
    )
    return net

class CategoricalPolicy(PolicyNetwork):
    """
    Base class for categorical policy.

    Desired network needs to be implemented.
    """

    def __init__(self, num_actions):

        super().__init__()

        self.num_actions = num_actions

    def sample(self, obs, no_log_prob=False):
        #print('obs',obs)
        logits = self.forward(obs)
        #print('logits',logits)
        dist = td.Categorical(logits=logits)
        #dist = td.Categorical(probs=logits)
        #print([dist.log_prob(i) for torch.tensor([i]) in range(5)])
        
        action = dist.sample(sample_shape=torch.tensor([1]))
        #print('action', action)
       
        return action if no_log_prob else (action, dist.log_prob(action), logits[action])

    def log_probs(self, obs, actions):
        dists = td.Categorical(logits=self.forward(obs))
        #print(dists)
        return dists.log_prob(actions.flatten())

    def entropy(self, obs):
        dists = td.Categorical(logits=self.forward(obs))
        return dists.entropy()


class CategoricalPolicyTwoLayer(CategoricalPolicy):
    """
    Categorical policy using a fully connected two-layer network.
    """

    def __init__(self, state_dim, num_actions,
                 hidden_layer1_size=64,
                 hidden_layer2_size=64,
                 init_std=0.001):

        super().__init__(num_actions)
        #print('in policy', num_actions)
        #print('state dim', state_dim)
        self.init_std = init_std

        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.linear3 = nn.Linear(hidden_layer2_size, num_actions)
        #print('got layers')
        #print('init_std', init_std)
        torch.manual_seed(0)
        nn.init.normal_(self.linear1.weight, std=init_std)
        nn.init.normal_(self.linear2.weight, std=init_std)
        nn.init.normal_(self.linear3.weight, std=init_std)
        #print(self.linear1.weight)
        #print('initialized normal')
        self.softmax = nn.Softmax()
    def forward(self, state):
        #print('state in forward', state)
        x = F.relu(self.linear1(state))
        #print('self linear 1', x)
        x = F.relu(self.linear2(x))
        #print('self linear 2', x)
        x = self.linear3(x)
        #print('self linear 3', x)
        #output = self.softmax(x)
        #print('sum output', torch.sum(output))
        #print(output)
        return x
        #return output

class CategoricalValueTwoLayer(CategoricalPolicy):
    """
    Categorical policy using a fully connected two-layer network.
    """

    def __init__(self, state_dim, num_actions,
                 hidden_layer1_size=64,
                 hidden_layer2_size=64,
                 init_std=0.001):

        super().__init__(num_actions)
        #print('init cat pol')
        self.init_std = init_std
        #print('init_std', self.init_std)
        #print('state dim', state_dim, 'hidden 1', hidden_layer1_size)
        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        #print('linear 1')
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        #print('lienar 2')
        self.linear3 = nn.Linear(hidden_layer2_size, num_actions)
    
        #print('got layers')
        nn.init.normal_(self.linear1.weight, std=init_std)
        nn.init.normal_(self.linear2.weight, std=init_std)
        nn.init.normal_(self.linear3.weight, std=init_std)
        #print('done intializing cat pol')

    def forward(self, state):
        #print('in forward state', state)
        x = F.relu(self.linear1(state))
        #print('x after 1', x)
        x = F.relu(self.linear2(x))
        #print('x after 2', x)
        output = self.linear3(x)
        #print('output', output)
        return output

class CategoricalPolicyConvolutional(CategoricalPolicy):
   """
   Categorical policy using a convolutional network.
   """

   def __init__(self, num_actions,
                example_state_tensor,
                in_channels=1,
                out_channels1=16, kernel_size1=3, stride1=2, padding1=1,
                in_channels2=16, out_channels2=32, kernel_size2=2, stride2=1,
                padding2=1,
                out_features3=256):

       super().__init__(num_actions)

       example_state_tensor = self._reshape_state(example_state_tensor)

       self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size1,
                              stride1, padding1)
       self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size2,
                              stride2, padding2)

       in_features3 = self.conv2(
           self.conv1(example_state_tensor.to(torch.device('cpu')))).view(-1).shape[0]

       self.fc3 = nn.Linear(in_features3, out_features3, bias=True)
       self.head = nn.Linear(out_features3, num_actions, bias=True)

   def forward(self, x):
       x = self._reshape_state(x)
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = F.relu(self.fc3(x.view(x.size(0), -1)))  # flattens Conv2d output
       return self.head(x)

   def _reshape_state(self, x):
       num_elems = len(x) if len(x.shape) > 2 else 1
       dims = x.shape[-2:]
       return x.reshape(num_elems, 1, *dims)
