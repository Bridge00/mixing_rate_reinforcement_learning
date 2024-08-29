import numpy as np
import torch
import torch.nn.functional as F


def vstack_flattened(*args):

    flattened_args = [np.array(arg).flatten() for arg in args[0]]

    return np.vstack(flattened_args).T


def get_meshgrid(state_lows, state_highs, step):
    #print('get meshgrid')
    L = len(state_lows)
    grids = []
    #print(L)
    for i in range(L):
        #print('state low dim', i)
        grid = np.arange(state_lows[i], state_highs[i] + step, step)
        grids.append(grid)
    meshgrid = np.meshgrid(*grids, indexing='ij')
    return meshgrid

def multivariate_gaussian_entropy(std):
    k = len(std)
    #print('k', std)
    variances = np.square(std)
    cov_matrix = np.diag(variances)
    det_cov_matrix = np.linalg.det(cov_matrix)
    entropy = 0.5 * np.log((2*np.pi*np.e)**k * det_cov_matrix)
    return entropy

## if you have a state space where each state is a two-dimensional vector
## where the first index can go from -1 to 1 and the second index can go from
## 0 to 5, then state_lows = [-1, 0] and state_highs = [1, 5]  
def get_avg_entropy_neural(state_lows, state_highs, step, network, discrete_action = False):

    #print('neural entropy')
    network.eval()
    # if network.training:
    #     print("Network is in training mode.")
    # else:
    #     print("Network is in evaluation mode.")
    sampled_states = vstack_flattened(get_meshgrid(state_lows, state_highs, step))
    #print('sampled_states')
    entropies = []

    for s in sampled_states:
        #print(s)
        s_torch = torch.from_numpy(s)
        s_torch = s_torch.view(1, -1).to(torch.float32)
        #print(s_torch)
        #input_tensor = torch.randn(1, 2)
        #print(s_torch.view(1, -1).dtype, input_tensor.dtype)
        if not discrete_action:
            std = network(s_torch)[-1]
            #print('output std', std)
            #mu, std = network(s_torch.view(1, -1)).numpy()
            
            std = std[0].detach().numpy()
            #print('std', std)
            # std = std.detach().numpy()
            # print('numpy std', std)
            entropies.append(multivariate_gaussian_entropy(std))
        #print('got entropy')
        else:
            #print('discrete action neural entropy')
            action_dist, probs = network(s_torch)[:-1]
            #print(probs)
            #probs = F.softmax(action_dist, dim = 1).detach().numpy()
            entropies.append(-np.sum([p * np.log(p) for p in probs[0]]))

    network.train()
    m = np.mean(entropies)
    #print(m)
    return m



def map_entropy(init_entropy, init_rollout, alpha, current_entropy):
    #print('in map entropy')
    # Create the ranges
    # a = [0, init_entropy]
    # b = [ alpha, rollout]
    #print(init_entropy, rollout, max_traj, current_entropy)
    # if current_entropy > init_entropy:
    #     current_entropy = init_entropy
    # Map current_entropy from range a to range b
    #mapped_value = rollout + np.exp((current_entropy - init_entropy)/max_traj) #* (rollout - max_traj) + max_traj
    mapped_value = alpha - current_entropy/init_entropy * ( alpha - init_rollout)
    #mapped_value = np.exp(max_traj - current_entropy/init_entropy * (max_traj - rollout))
    #print(mapped_value)
    #mapped_value = init_rollout * np.exp(alpha* (init_entropy/current_entropy - 1))
    # Round the mapped value to the nearest integer
    mapped_value_rounded = round(mapped_value)

    # Return the rounded value
    return mapped_value_rounded

def get_avg_entropy(num_states, pi):
    entropy = 0
    pi_to_list = []

    for s in range(num_states):
        probs = pi._get_probs(s)
    
        pi_to_list.append(probs)

        #print(probs)
        entropy -= np.sum([p * np.log(p) for p in probs])

    entropy /= num_states

    return entropy, np.array(pi_to_list)

# def multiply_3d_array_2d_array(arr_3d, arr_2d):

#     #print(arr_2d.shape)
#     #arr_3d = torch.from_numpy(arr_3d).to(device="cuda")
#     #arr_2d = torch.from_numpy(arr_2d).to(device)
#     s, a, _ = arr_3d.shape
#     #print(_, a)
#     #return np.sum(arr_3d.reshape(s, s, a) * arr_2d.reshape(s, 1, a), axis=2)
#     return torch.sum(arr_3d.reshape(s, s, a) * arr_2d.reshape(s, 1, a), dim=2) #.cpu().numpy()

def nth_largest_eigenvalue(arr_3d, arr_2d, n):
    """
    Compute the nth largest eigenvalue of a matrix.
    Parameters:
    matrix (np.ndarray): Input matrix.
    Returns:
    float: The second largest eigenvalue of the matrix.
    """
    #arr_2d = torch.from_numpy(arr_2d).to(device)
    #matrix = multiply_3d_array_with_2d_array(arr_3d, arr_2d)

    s, a, _ = arr_3d.shape
    matrix = torch.sum(arr_3d.reshape(s, s, a) * arr_2d.reshape(s, 1, a), dim=2)

    #matrix = torch.from_numpy(matrix).to(device="cuda")
    #eigenvalues, _ = np.linalg.eig(matrix)
    eigenvalues, _ = torch.linalg.eig(matrix)
    del matrix
    #sorted_eigenvalues = np.sort(eigenvalues)
    sorted_eigenvalues = np.sort(eigenvalues.cpu().numpy())
    #print(sorted_eigenvalues)
    #print(sorted_eigenvalues)
    return sorted_eigenvalues[-n]