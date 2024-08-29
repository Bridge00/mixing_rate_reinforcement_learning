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
#from avg_agents_old import *
from ac_agents import *
from time import time
import torch
#import torch.multiprocessing as mp
#from trial import trial, train
from entropy_eig_utils import map_entropy, get_avg_entropy, nth_largest_eigenvalue
from env_utils import *
from save_utils import *
from utils import estimate_loss, advantage_estimation
import wandb


parser = argparse.ArgumentParser(description='Train')


parser.add_argument('--pg', type=str, default= 'AC', metavar='N',
                    help='type of pg')

parser.add_argument('--iterations_per_ep', '-i', type=int, default= 200, metavar='N',
                    help='number of iterations per episode')

parser.add_argument('--episodes', '-e', type=int, default= 350, metavar='N',
                    help='number of episodes')

parser.add_argument('--episode_save_freq', type=int, default= 1, metavar='N',
                    help='episode save freq')

parser.add_argument('--policy_lr', '-p', type=float, default= .01, metavar='N',
                    help='policy lr setup')

parser.add_argument('--value_lr', '-v', type=float, default= .01, metavar='N',
                    help='value lr setup')

parser.add_argument('--grid_length', type=int, default= 5, metavar='N',
                    help='grid_length')

parser.add_argument('--is_mlmc', '-mlmc', type=int, default= 1, metavar='N',
                    help='is mlmc')

parser.add_argument('--horizon_parameter', '-horizon', type=int, default= 4, metavar='N',
                    help='rollout length')

parser.add_argument('--adaptive_horizon', '-at', type=int, default= 0, metavar='N',
                    help='adatpive horizon')

parser.add_argument('--state_adaptive', '-sa', type=int, default= 1, metavar='N',
                    help='adaptive by episode')

parser.add_argument('--adaptive_horizon_sensitivity_parameter', '-ad_sens', type=float, default= 20, metavar='N',
                    help='horizon sensitivity parameter')

parser.add_argument('--env_name', '-env', type=str, default= '2D_grid', metavar='N',
                    help='environment')

parser.add_argument('--seed_num', '-seed',   type=int, default= 0, metavar='N',
                    help='seed number')

parser.add_argument('--reset_to_start', type=int, default= 1, metavar='N',
                    help='reset to start')

parser.add_argument('--keep_goal_same', type=int, default= 1, metavar='N',
                    help='keep goal same')

parser.add_argument('--estimated_tau_mix', type=int, default= 1, metavar='N',
                    help='estimated mixing time')

parser.add_argument('--save_locally', type=int, default= 1, metavar='N',
                    help='save locally')

parser.add_argument('--save_wandb', type=int, default= 1, metavar='N',
                    help='save on wandb')

args = parser.parse_args()


num_states = args.grid_length ** 2
start = 0 #args.grid_length + 1
goal = num_states - 1 # - args.grid_length - 2#(args.grid_length-1)**2 -1#- 2 - args.grid_length#args.goal
initial_horizon_parameter = args.horizon_parameter

print(args.env_name)

if args.save_wandb:

    wandb.init(
        # set the wandb project where this run will be logged
        project="mlmc",
        
        # track hyperparameters and run metadata
        config={
        "environment": args.env_name,
        "grid_length": args.grid_length,
        "start": start,
        "goal": goal,
        "episodes": args.episodes,
        "iters_per_ep": args.iterations_per_ep,
        "PG Alg": args.pg,
        "initial_horizon_parameter": args.horizon_parameter,
        "adaptive_horizon": args.adaptive_horizon,
        "state_adaptive": args.state_adaptive,
        "adaptive_horizon_sensitivity_parameter": args.adaptive_horizon_sensitivity_parameter,
        "is_mlmc": args.is_mlmc,
        "policy_learning_rate": args.policy_lr,
        "value_learning_rate": args.value_lr,
        "seed": args.seed_num,

        }
    )




device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if args.save_locally:
    results_folder = 'results'
    if not args.keep_goal_same:
        results_folder += '/goal_conditioned'
    else:
        results_folder += '/goal_constant'

    if args.reset_to_start:
        results_folder += '/reset_to_initial_start'
    else:
        results_folder += '/start_from_current_state'


    if args.is_mlmc:
        
        grad_estimation = f'MLMC'  
        
        horizon = f'log_tmax_{args.horizon_parameter}'

    else:
        grad_estimation = f'Vanilla'

        horizon = f'horizon_{args.horizon_parameter}'
    
    if args.pg == 'REINFORCE':
        alg = f'REINFORCE'
    elif args.pg == 'AC':
        alg = f'AC'
    else:
        alg = 'PPGAE'

    if args.adaptive_horizon:
        horizon_info = f'adaptive_horizon_param'


        if args.state_adaptive:
            horizon_info += '/conditional_state_entropy'
        else:
            horizon_info += '/avg_entropy'

        horizon_info += f'/adaptive_sensitivity_parameter_{args.adaptive_horizon_sensitivity_parameter}'

    else:

        horizon_info = f'fixed_horizon_param'


    folder_list = [
                   args.env_name,
                   f'{args.grid_length**2}_states', 
                   alg, 
                   grad_estimation,
                   horizon, 
                   f'plr_{args.policy_lr}_vlr_{args.value_lr}', 
                   f'{args.iterations_per_ep}_iters_per_ep'
                   ] #, f'{args.episodes}_ep']

    for f in folder_list:
        results_folder = os.path.join(results_folder, f)

    os.makedirs(results_folder, exist_ok=True)
    print(f'saving to {results_folder}')



def train(pg, env):


    print(f'trial {args.seed_num}')

    goal_sequence = [env.goal]
    largest_goal = env.goal
    while len(goal_sequence) < 1000:
        new_goal = np.random.randint(largest_goal) 
        if new_goal != goal_sequence[-1] and new_goal not in walls:
            goal_sequence.append(new_goal)

    print(env)

    start_episode = 1


    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.save_locally:
        episode_rewards_track = [] 
        episode_samples_track = []
        cumulative_samples_track = [] 
        horizon_parameter_track, pg_track, entropy_track = [], [], []
        pis, value_tables = [], []
        goal_reached_track = []
    # entropy_heatmaps = []
    #pis, value_tables = [], []

    #env.reset(seed = 0)

   
    with torch.no_grad():
        if not args.state_adaptive:
            entropies = []
            for state_index in range(num_states):

                two_d_rep = state_index_to_coord(state_index, args.grid_length) + state_index_to_coord(env.goal, args.grid_length)

                entropies.append([pg.pi.entropy(pg._torch_state(two_d_rep))])

            initial_entropy = np.mean(entropies)

        else:
            two_d_rep = state_index_to_coord(0, args.grid_length) + state_index_to_coord(env.goal, args.grid_length)
            initial_entropy = pg.pi.entropy(pg._torch_state(two_d_rep))

    pg.pi.train()
    if args.pg != 'REINFORCE':
        pg.v_c.train()

    initial_entropy = initial_entropy.item()

    current_entropy = initial_entropy
    cumulative_samples = 0

    goal_sequence_index = 0

    for e in range(start_episode, start_episode + args.episodes):


        if args.reset_to_start and args.keep_goal_same:
            env.reset(seed = 0)
        elif args.reset_to_start and not args.keep_goal_same:
            env.reset(seed = 0, goal = goal_sequence[goal_sequence_index])
        else:
            env.reset(goal = goal_sequence[goal_sequence_index])
        
        goal_sequence_index += 1

        episode_total_reward = 0
        num_episode_samples = 0
        done = False
        rollouts = 0
        while num_episode_samples < args.iterations_per_ep and not done: 
            states, torch_states, rewards, next_states, log_probs, actions, logits = [], [], [], [], [], [], []

            state = env.get_state()

            if args.is_mlmc:
                geom_draw = (np.random.geometric(p=0.50, size=1)[0])
           
                while geom_draw > args.horizon_parameter: #sample length is log 2 tmax
                    
                    geom_draw = (np.random.geometric(p=0.50, size=1)[0])
    
                horizon =  int(2**(geom_draw))

            else:
                if args.pg in ['AC', 'REINFORCE', 'PPO']:
                    horizon = args.horizon_parameter
                else:
                    horizon = args.iterations_per_ep

            r = 0
            done = False

            while r < horizon and num_episode_samples < args.iterations_per_ep and not done:
                states.append(state)
                torch_state = pg._torch_state(state)
                torch_states.append(torch_state)
                #print('state', torch_state)
                action, log_prob, logit  = pg.pi.sample(torch_state)
                logits.append(logit)
                actions.append(action)
                log_probs.append(log_prob)
                state, reward, done = env.step(action.item())[:3]

                next_states.append(pg._torch_state(state))
              
                rewards.append(reward)
                pg.number_of_samples += 1
                num_episode_samples += 1
                cumulative_samples += 1
                r += 1

            mean_reward  = np.mean(rewards)
            total_reward = np.sum(rewards)
        

            v_c_losses, pi_losses =  [], []

            pg.mu_r = (1 - pg.tau) * pg.mu_r + pg.tau * mean_reward 

            
            pi_loss  = 0

          
            v_c_loss = 0

            #print(states)
            for state, reward, next_state, log_prob, action in zip(
                torch_states, rewards, next_states, log_probs, actions):

                if args.pg == 'AC':

                    with torch.no_grad():
        
                        v_c_target = float(reward) - pg.mu_r + pg.v_c(next_state)
                    
                        td_c = v_c_target - pg.v_c(state)


                    v_c_losses.append((pg.v_c(state) - v_c_target) ** 2)

                    pi_losses.append(td_c * -log_prob)

                elif args.pg == 'REINFORCE':

                    pi_losses.append((float(reward) - pg.mu_r) * -log_prob)

                else:

                    N = 1
                    pi_loss += (advantage_estimation(torch_states, actions, rewards, N, state, action, logit) * -log_prob)

            
                   
            
            if args.pg in ['AC', 'REINFORCE']:
                
                pi_loss = estimate_loss(pi_losses, args.is_mlmc, args.horizon_parameter) #

                pg.pi_optim.zero_grad()

                pi_loss.backward()
        
                if pg.grad_clip_radius is not None:

                    torch.nn.utils.clip_grad_norm_(pg.pi.parameters(),
                                                pg.grad_clip_radius)
        
                pg.pi_optim.step()

                if args.pg == 'AC':
                    v_c_loss = estimate_loss(v_c_losses, args.is_mlmc, args.horizon_parameter)
                    pg.v_c_optim.zero_grad()
                    v_c_loss.backward()
                    if pg.grad_clip_radius is not None:
                        torch.nn.utils.clip_grad_norm_(pg.v_c.parameters(),
                                                    pg.grad_clip_radius)
                    pg.v_c_optim.step()

            else:
                pi_loss.backward()

                if pg.grad_clip_radius is not None:

                    torch.nn.utils.clip_grad_norm_(pg.pi.parameters(),
                                                pg.grad_clip_radius)
                pg.pi_optim.step()

            episode_total_reward += total_reward
        
            rollouts +=1
        
            goal_reached = int(done)

            episode_finished = (num_episode_samples == args.iterations_per_ep or done)

            if args.adaptive_horizon and ((args.state_adaptive) or (not args.state_adaptive and episode_finished)):
                with torch.no_grad():
                    if not args.state_adaptive:
                        entropies = []
                        for state_index in range(num_states):

                            two_d_rep = state_index_to_coord(state_index, args.grid_length) + state_index_to_coord(env.goal, args.grid_length) 
                            entropies.append([pg.pi.entropy(pg._torch_state(two_d_rep))])

                        current_entropy = np.mean(entropies)

                    else:
                        current_entropy = pg.pi.entropy(next_state)
                    current_entropy = current_entropy.item()

                x = map_entropy(initial_entropy, initial_horizon_parameter, args.adaptive_horizon_sensitivity_parameter, current_entropy)
            
                if x != args.horizon_parameter:
                    args.horizon_parameter = x
                    print('new rollout', x)
                # if args.horizon_parameter > r_max:
                #     r_max = args.horizon_parameter


        if e % args.episode_save_freq == 0:

            if args.save_wandb:
                wandb.log({"reward": episode_total_reward, 
                            "average_episode_reward": episode_total_reward/num_episode_samples,
                            "num_episode_samples": num_episode_samples,
                            "cumulative_samples": cumulative_samples,
                            "rollout_length": args.horizon_parameter,
                            "goal_reached" : goal_reached,
                            })

            if args.save_locally:
                episode_rewards_track.append(episode_total_reward)
                episode_samples_track.append(num_episode_samples)
                cumulative_samples_track.append(cumulative_samples)
                horizon_parameter_track.append(args.horizon_parameter)
                pg_track.append(pg)
                goal_reached_track.append(goal_reached)
                #entropy_heatmaps.append(entropy_heatmap_episode)
                #entropy_track.append(entropy)
        
                #print(trial_num, i, cumulative_samples, cumulative_reward, np.round(pg, 3), rl)
                titles_vals = { 
                    #'entropies': entropy_track, 
                    #'model': ac, 
                    'pi_gradients' : pg_track, 
                    'rollouts' : horizon_parameter_track, 
                    'total_rewards' : episode_rewards_track, 
                    'total_samples' : cumulative_samples_track,
                    'num_episode_samples' : episode_samples_track,
                    'goal_reached': goal_reached_track,
                    #'entropy_heatmaps': entropy_heatmaps
                #'average_reward': average_reward_track,
                }
            
                save_progress(results_folder, f'{e}_episodes', f'trial_{args.seed_num}', titles_vals)


        print(args.seed_num, e, num_episode_samples, cumulative_samples, episode_total_reward, goal_reached, horizon) #current_entropy

    return env




if __name__ == '__main__':

    print('goal', goal)
    a = 5
    if 'no_stay' in args.env_name:
        a = 4
    print(a)


    if 'walls_1' in args.env_name:
        walls = [2, 7, 17, 22]
    elif 'walls_2' in args.env_name:
        walls = wall_indices2
    elif 'walls_3' in args.env_name:
        walls = wall_indices3
    elif 'walls_even' in args.env_name:
        walls = wall_indices_even_9
    elif 'walls_uneven' in args.env_name:
        walls = wall_indices_uneven_9
    elif 'walls_snake' in args.env_name:
        walls = wall_indices_snake_9
    else:
        walls = []

    print(walls)
 
    env, state_dim, action_dim, state_lows, state_highs = choose_env(args.env_name, args.grid_length, goal = goal, walls = walls)[:5]
 
    pg = get_pg_alg(args.pg, state_dim, action_dim, args.policy_lr, args.value_lr, state_lows, state_highs, args.is_mlmc)
    env = train(pg, env) 
    print('trained')


    # wandb.finish()
    # test(pg, env)   
    # print('tested')

