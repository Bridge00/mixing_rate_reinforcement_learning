#import matplotlib.pyplot as plt
#import gym
import gymnasium as gym
#from gymnasium.envs.classic_control import MountainCar
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
from time import time
import torch
import torch.multiprocessing as mp
#from trial import trial, train
from entropy_eig_utils import map_entropy, get_avg_entropy, nth_largest_eigenvalue
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import math
from numpy import cos, sin

wall_grid  = [
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O'] ,
['O', 'O', 'O', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ]

wall_grid = [
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'X'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'X'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
]

# wall_grid = [
#     ['O', 'X'],
#     ['O', 'O']
# ]

# wall_grid = [
#     ['O', 'X', 'O', 'X'],
#     ['O', 'O', 'O', 'O'],
#     ['O', 'O', 'X', 'X'],
#     ['O', 'O', 'O', 'O']
# ]

# wall_grid = [
#     ['O', 'X', 'O', 'O', 'X', 'O', 'X' , 'X', 'O', 'X'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'O' , 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O']
# ]

# wall_grid = [
#     ['O', 'X', 'O', 'O', 'X', 'O', 'X' , 'X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'O', 'O', 'O', 'X' , 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O']
# ]


wall_grid  = [
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
]

wall_grid2 = [

['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 

]

wall_grid3 = [
    
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O'] 

]





f = np.array(wall_grid).flatten()
wall_indices = [i for i,_ in enumerate(f) if f[i] == 'X']

f = np.array(wall_grid2).flatten()
wall_indices2 = [i for i,_ in enumerate(f) if f[i] == 'X']

f = np.array(wall_grid3).flatten()
wall_indices3 = [i for i,_ in enumerate(f) if f[i] == 'X']


critical_states_grid1 = [
   
    ['O', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['X', 'O', 'O', 'O', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'O', 'O'],
  
]

f = np.array(critical_states_grid1).flatten()
cs_wall_indices1 = [i for i,_ in enumerate(f) if f[i] == 'X']


critical_states_grid2 = [
   
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['X', 'X', 'X', 'X', 'O', 'O', 'X'],
    ['X', 'X', 'X', 'X', 'O', 'O', 'X'],
    ['X', 'X', 'X', 'X', 'O', 'O', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
  
]

f = np.array(critical_states_grid2).flatten()
cs_wall_indices2 = [i for i,_ in enumerate(f) if f[i] == 'X']


critical_states_grid14 = [
   
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],

  
]

f = np.array(critical_states_grid14).flatten()
cs_wall_indices14 = [i for i,_ in enumerate(f) if f[i] == 'X']

# hole_grid = [
# ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O'] ,
# ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ]


# hole_grid =[
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'X', 'O', 'X' , 'O', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'O', 'O', 'O', 'O', 'O', 'O' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
#     ['O', 'X', 'O', 'X', 'X', 'O', 'X' , 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O']
# ]

# hole_grid =[
#     ['O', 'X'],
#     ['O', 'O']
# ]


hole25 = [
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
]

# hole_grid = [
    
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O']  


# ]


# hole_grid = [
    
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'] ,
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O'],
# ['O', 'O', 'O',  'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O']  


# ]


# hole_grid = [

#     ['O', 'O'],
#     ['X', 'O']
# ]





wall_grid_2_15 = [
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]


wall_grid_3_15 = [
['O', 'O', 'o', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'o', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['X', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'X', 'X'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'],
 ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['X', 'X', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'X', 'X'],
 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O'],
 ['O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O', 'O', 'O']]


f = np.array(wall_grid_2_15).flatten()
wall_indices2_15 = [i for i,_ in enumerate(f) if f[i] == 'X']

f = np.array(wall_grid_3_15).flatten()
wall_indices3_15 = [i for i,_ in enumerate(f) if f[i] == 'X']



wall_indices_even_9   = [4, 13, 22, 36, 37, 38, 42, 43, 44, 58, 67, 76]

wall_indices_uneven_9 = [3, 12, 18, 19, 23, 24, 25, 26, 48, 57, 66, 75]

wall_indices_snake_9  = [1, 5, 10, 14, 19, 23, 28, 30, 32, 34, 37, 39, 41, 43, 46, 48, 50, 52, 57, 61, 66, 70, 75, 79]


#this function takes in n as an input and return a transition probability dictionary 
#where each (state, action) pair is a key. Value is a list of length n**2. Each element represents
#the probabilty of going to a certain state based on the (state, action) key
#All entries are initally set to a list of n**2 length with all 0's. 
#Then in the double for loop it goes through each state, action pair and decides which element to turn into 1
#meaning that given a (state, action), with 100% certainty, you will go to a certain state
# def n_to_grid_trans(n):
    
#     tp = {}
    
#     for state in range(n**2):
#         for action in range(5):
#             tp[(state, action)] = [0] * (n**2)
#             if action == 0:
#                 tp[(state, action)][state] = 1
            
#             if action == 1:  #Up
#                 if state - n > -1:
#                     tp[(state, action)][state-n] = 1
#                 else:
#                     tp[(state, action)][state] = 1
                
#             if action == 2: # Down
#                 if state + n < n**2:
#                     tp[(state, action)][state+n] = 1
#                 else:
                    
#                     tp[(state, action)][state] = 1
                    
#             if action == 3: # Left
#                 if (state - 1) % n < n-1:
#                     tp[(state, action)][state-1] = 1
#                 else:
#                     tp[(state, action)][state] = 1
                
#             if action == 4: #Right
#                 if (state + 1) % n > 0:
#                     tp[(state, action)][state+1] = 1   
#                 else:
#                     tp[(state, action)][state] = 1
#     return tp


def n_to_grid_trans_no_stay(n, walls = []):
    
    tp = {}
    
    for state in range(n**2):
        #if state not in holes:
        for action in range(4):
            tp[(state, action)] = [0 for _ in range(n**2)] #[0] * (n**2)

            
            if action == 0:  #Up
                #print('Up')
                if state - n > -1 and state - n not in walls:
                    tp[(state, action)][state-n] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 1: # Down
                #print('Down')
                if state + n < n**2 and state + n not in walls:
                    tp[(state, action)][state+n] = 1
                else:
                    
                    tp[(state, action)][state] = 1
                    
            if action == 2: # Left
                #print('Left')
                if (state - 1) % n < n-1 and state - 1 not in walls:
                    tp[(state, action)][state-1] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 3: #Right
                #print('Right')
                if (state + 1) % n > 0 and state + 1 not in walls:

                    tp[(state, action)][state+1] = 1   
                else:
                    tp[(state, action)][state] = 1
    return tp

def n_to_grid_trans_no_stay_3D(n, walls = []):
    
    tp = {}
    
    for state in range(n**3):
        #if state not in holes:
        for action in range(4):
            tp[(state, action)] = [0 for _ in range(n**2)] #[0] * (n**2)

            
            if action == 0:  #Up
                #print('Up')
                if state - n**2 > -1 and state - n**2 not in walls:
                    tp[(state, action)][state-n**2] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 1: # Down
                #print('Down')
                if state + n < n**2 and state + n not in walls:
                    tp[(state, action)][state+n] = 1
                else:
                    
                    tp[(state, action)][state] = 1
                    
            if action == 2: # Left
                #print('Left')
                if (state - 1) % n < n-1 and state - 1 not in walls:
                    tp[(state, action)][state-1] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 3: #Right
                #print('Right')
                if (state + 1) % n > 0 and state + 1 not in walls:

                    tp[(state, action)][state+1] = 1   
                else:
                    tp[(state, action)][state] = 1
    return tp



def n_to_grid_trans(n, walls = wall_indices):
    
    tp = {}
    
    for state in range(n**2):
        #if state not in holes or state in holes:
        for action in range(5):
            tp[(state, action)] = [0] * (n**2)
            if action == 0:
                tp[(state, action)][state] = 1
            
            if action == 1:  #Up
                if state - n > -1 and state - n not in walls:
                    tp[(state, action)][state-n] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 2: # Down
                if state + n < n**2 and state + n not in walls:
                    tp[(state, action)][state+n] = 1
                else:
                    
                    tp[(state, action)][state] = 1
                    
            if action == 3: # Left
                if (state - 1) % n < n-1 and state - 1 not in walls:
                    tp[(state, action)][state-1] = 1
                else:
                    tp[(state, action)][state] = 1
                
            if action == 4: #Right
                if (state + 1) % n > 0 and state + 1 not in walls:

                    tp[(state, action)][state+1] = 1   
                else:
                    tp[(state, action)][state] = 1
    return tp




def n_to_rewards_4(n, g, value = 0, goal_value = 1):
    rewards = []
    for _ in range(n**2):
        rewards.append([value] * 4)

    for state in range(n**2):

        if state - n == g:
            #print('up')
            rewards[state][0] = goal_value #[value, 1, value, value, value]
        if state + n == g:
            #print('down')
            rewards[state][1] = goal_value #[value, value, 1, value, value]
        if state - 1 == g:
            #print('left')
            rewards[state][2] = goal_value #[value, value, value, 1, value]
        if state + 1 == g:
            #print('right')
            rewards[state][3] = goal_value #[value, value, value, value, 1]


        # if state - n > -1 and  state - n in holes:
        #     #print('up')
        #     rewards[state][0] = -1 #[value, -1, value, value, value]

        # if state + n < n**2 and state + n in holes:
        #     #print('down')
        #     rewards[state][1] = -1 #[value, value, -1, value, value]

        # if (state - 1) % n < n-1 and state - 1 in holes:
        #     #print('left')
        #     rewards[state][2] = -1 #[value, value, value, -1, value]

        # if (state + 1) % n > 0 and state + 1 in holes:
        #     #print('right')
        #     rewards[state][3] = -1 #[value, value, value, value, -1]  
    #print(rewards)
    return rewards



def n_to_rewards(n, g, value = 0):
    rewards = []
    for _ in range(n**2):
        rewards.append([value] * 5)


    for state in range(n**2):
        if state == g:
            #print('up')
            rewards[state][0] = 1 #[1, value, value, value, value]   
            
        if state - n == g:
            #print('up')
            rewards[state][1] = 1 #[value, 1, value, value, value]
        if state + n == g:
            #print('down')
            rewards[state][2] = 1 #[value, value, 1, value, value]
        if state - 1 == g:
            #print('left')
            rewards[state][3] = 1 #[value, value, value, 1, value]
        if state + 1 == g:
            #print('right')
            rewards[state][4] = 1 #[value, value, value, value, 1]

        # if state in holes:
        #     #print('stayed')
        #     rewards[state][0] = -1 #[-1, value, value, value, value] 

        # if state - n > -1 and  state - n in holes:
        #     #print('up')
        #     rewards[state][1] = -1 #[value, -1, value, value, value]

        # if state + n < n**2 and state + n in holes:
        #     #print('down')
        #     rewards[state][2] = -1 #[value, value, -1, value, value]

        # if (state - 1) % n < n-1 and state - 1 in holes:
        #     #print('left')
        #     rewards[state][3] = -1 #[value, value, value, -1, value]

        # if (state + 1) % n > 0 and state + 1 in holes:
        #     #print('right')
        #     rewards[state][4] = -1 #[value, value, value, value, -1]  
    #print(rewards)
    return rewards




def state_index_to_coord(state, grid_length):
    #print(state, grid_length)
    return [state // grid_length, state % grid_length]

def state_index_to_coord_3D(state, grid_length):
    #print(state, grid_length)
    return [state // grid_length ** 2, state // grid_length, state % grid_length]
    #[state // grid_length, state % grid_length]

def state_coord_to_index(state, grid_length):
    #print(state, grid_length)
    return state[0] * grid_length + state[1]#[state // grid_length, state % grid_length]



class MDPEnv(gym.Env):
    """
    A Gym-compatible environment that takes fully-specified average-reward MDPs.
    """
    
    def __init__(self, states, actions, transition_probabilities = {}, rewards = [[]], goal = 194, holes = []):
        """
        Parameters
        ----------
        states:                   a list of states 
        actions:                  a list of actions (*descriptions* of actions)
        transition_probabilities: a dictionary that returns a state distribution
                                  for a given (state, action) pair
        rewards:                  a numpy array of dimension num_states by num_actions
                                  specifying rewards for each state-action pair
        """
        #print('rewards matrix in evn', rewards)
        #print(states, actions)
        self.states                   = {s: i for i, s in enumerate(states)}
        #print('self.states', self.states)
        self.num_states               = len(states)
        #print(self.num_states)
        self.grid_length              = np.sqrt(len(states))
        self.actions                  = {a: i for i, a in enumerate(actions)}

        #print('self.actions', self.actions)
        self.rewards                  = self._array_to_fun(rewards)

        #print('self.rewards', self.rewards)
        self.transition_probabilities = self._dict_to_fun(transition_probabilities)
        #print('self.transprob', self.transition_probabilities)
        self.state             = 0
        self.goal              = goal#self.num_states - 1
        #print(self.goal)
      
        #print('self.state', self.state)
        self.observation_space = gym.spaces.Discrete(len(states))
        #print('self.obs', self.observation_space)
        self.action_space      = gym.spaces.Discrete(len(actions))

        self.holes = holes 
        #print(self.holes)
        #print('self.action_sapce', self.action_space)
        #print('initalized')
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
        #print('turning into function')
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

    def change_rewards(self, rewards):

        self.rewards = self._array_to_fun(rewards)

        for state in range(36):
            print([self.rewards(state, action) for action in range(5)])


    def get_state(self):

        return state_index_to_coord(self.state, self.grid_length) + state_index_to_coord(self.goal, self.grid_length)
        
        #print([self.state, self.goal])
        #return [self.state] #, self.goal]
        #two_d_rep = self.state_index_to_coord(self.state) + self.state_index_to_coord(self.goal)
        #print('in get state of mdp')
        #two_d_rep = state_index_to_coord(self.state, self.grid_length)
        #print(two_d_rep)
        #return two_d_rep


    def step(self, action):
        """
        Parameters
        ----------
        action: an element of the action space
        """
        #print('action in step argument', action)
        action = self.actions[action]
        #print('action from self.action', action)
        
        #reward = self.rewards(self.state, action)

        distribution = self.transition_probabilities(self.state, action)

        #print('distribution', distribution)
        x = self.state
        #print('dist in step', distribution)
        #print(self.observation_space.n)
        self.state = np.random.choice(self.observation_space.n, p=distribution)

        done = False #(self.state == self.goal) #or (self.state in self.holes)
        reward = 0
        #print('done in step', done)
        # if action == 0:
        #     reward -= 1
        # else:
        #     reward -= 2#self.move_penalty

        if self.state == self.goal:
            print('goal', self.state, self.goal)
            done = True
        
            reward = 1

        #return self.state, reward, done, {}
        return self.get_state(), reward, done

    def reset(self, seed = None, goal = None):
        """
        """
        #self.state = seed #self.observation_space.sample()
        if seed != None:
            self.state = seed
        if goal == None:
            self.goal = self.num_states - 1
        else:
            self.goal = goal

        return self.state








class PointMaze(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)

    def step(self, action):

 
        obs, reward, done = self.env.step([action])[:3]

        position_vel = obs['observation']
        goal = obs['desired_goal']

        state = np.array(position_vel + goal)
       
        return state, reward, done

class SparsePendulum(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
        print(self.env)
    def is_within_2_degrees(self, x, y):
        # Calculate the angle in radians between the line segment and the vertical line
        #print('in angle')
        angle = math.atan2(y, x)
        #print(angle)
        # Convert the angle to degrees
        angle_deg = math.degrees(angle)
        #print(angle_deg)
        # Check if the angle is within 2 degrees from the vertical line
        return int(abs(90 - angle_deg) <= 2)

    def get_state(self):
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)

    def step(self, action):

        #print('in step', action, type(action))
        #print(np.expand_dims(action, axis=0))
        state, reward, done = self.env.step([action])[:3]
        #print(state, reward, done)
        # Modify the reward here according to your requirements
        #print(state[0], state[1])
        modified_reward = self.is_within_2_degrees(state[0], state[1])
   
        #print(modified_reward)
        return state, modified_reward, done

class SparseCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
        print(self.env)
    def is_within_2_degrees(self, x, y):
        # Calculate the angle in radians between the line segment and the vertical line
        #print('in angle')
        angle = math.atan2(y, x)
        #print(angle)
        # Convert the angle to degrees
        angle_deg = math.degrees(angle)
        #print(angle_deg)
        # Check if the angle is within 2 degrees from the vertical line
        return int(abs(90 - angle_deg) <= 2)

    def get_state(self):
        return self.state



class SparseReacher(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
        print(self.env)


    def get_state(self):
        return self._get_obs()

    def step(self, a):

        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = np.linalg.norm(vec)
       
        reward = reward_dist != 0

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,

        )


class SparseHopper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)


    def step(self, action):

        #print('in step', action, type(action))
        #print(np.expand_dims(action, axis=0))
        state, reward, done = self.env.step([action])[:3]

        #print(modified_reward)
        return state, modified_reward, done


class SparseCheetah(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)


    def step(self, action):

        #print('in step', action, type(action))
        #print(np.expand_dims(action, axis=0))
        state, reward, done = self.env.step([action])[:3]

        #print(modified_reward)
        return state, modified_reward, done

class SparseAnt(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)


    def step(self, action):

        #print('in step', action, type(action))
        #print(np.expand_dims(action, axis=0))
        state, reward, done = self.env.step([action])[:3]

        #print(modified_reward)
        return state, modified_reward, done


class MountainCarDiscrete(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        return self.state  #np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)

    def step(self, action):

        state, reward, done = self.env.step(action)[:3]
        # print(state[0], action)
        
        # if state[0] > 0.3:
        #     reward = 0
        #     done = True

        return state, reward, done
        

class SparseAcrobot(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_state(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )



##def choose_env(env_name, states, actions, transition_probabilities, rewards, goal, holes):
def choose_env(env_name, n = 14, goal = 195, walls = [], num_tasks = 5, key = 72): #, states, actions, transition_probabilities, rewards, goal, holes):
    print(env_name)
    #if env_name in ['2D_cliff', '2D_cliff_no_stay', '2D_cliff_walls']:
    if '2D_grid' in env_name:
        print('MDP env')

        #print('holes', holes)
        #print(states, actions) #, transition_probabilities)
        if 'no_stay' in env_name:
            trans_prob = n_to_grid_trans_no_stay(n, walls = walls)
            action_dim = 4
            rewards = n_to_rewards_4(n, goal)
        else:
            trans_prob = n_to_grid_trans(n, walls = walls)
            action_dim = 5
            rewards = n_to_rewards(n, goal)

        env = MDPEnv(list(range(n**2)), list(range(action_dim)), trans_prob, rewards, goal)
        state_dim = 4
        
        state_lows = [0]
        state_highs = [n**2-1]
        action_lows = 0
        action_highs = action_dim - 1

        return env, state_dim, action_dim, state_lows, state_highs, action_lows, action_highs
    
    elif env_name == 'mountain_car_continuous':
        state_dim = 2
        action_dim = 1
        state_lows = [-1.2, -0.07]
        state_highs = [0.45, 0.07]
        min_action, max_action = -1, 1
        env = gym.make("MountainCarContinuous-v0")
        #return CustomMountainCar(), state_dim, action_dim, state_lows, state_highs
        return env, state_dim, action_dim, state_lows, state_highs, min_action, max_action

    elif env_name == 'mountain_car_discrete': # or env_name == 'mountain_car_discrete_zero':
        print('choosing env')
        print('discrete mountain car')
        state_dim = 2
        action_dim = 3
        state_lows = [-1.2, -0.07]
        state_highs = [0.5, 0.07]
        print(state_dim, action_dim, state_lows, state_highs)
        #return CustomMountainCar(), state_dim, action_dim, state_lows, state_highs
        env = gym.make("MountainCar-v0")

        env = MountainCarDiscrete(env)
        return env, state_dim, action_dim, state_lows, state_highs

    elif env_name == 'sparse_pendulum':
        print('choose sparse pendulum')
        state_dim = 3
        action_dim = 1
        state_lows = [-1, -1, -8]
        state_highs = [1, 1, 8]
        action_lows = [-2]
        action_highs = [2]

        env = gym.make('Pendulum-v1')

        # Wrap the environment with the custom reward wrapper
        env = SparsePendulum(env)

        print(env)
        return env, state_dim, action_dim, state_lows, state_highs

    elif env_name == 'sparse_cartpole':
        print('choose sparse cartpole')
        state_dim = 4
        action_dim = 2
        state_lows = [-4.8, -10000, -0.418, -100000]
        state_highs = [4.8, 10000, 0.418, 100000]


        env = gym.make('CartPole-v1')

        # Wrap the environment with the custom reward wrapper
        env = SparseCartPole(env)

        print(env)
        return env, state_dim, action_dim, state_lows, state_highs

    elif env_name == 'sparse_acrobot':
        print('choose sparse acrobot')
        state_dim = 6
        action_dim = 3
        state_lows = [-1, -1, -1, -1, -12.57, -28.27]
        state_highs = [1, 1, 1, 1, 12.57, 28.27]


        env = gym.make('Acrobot-v1')

        # Wrap the environment with the custom reward wrapper
        env = SparseAcrobot(env)

        print(env)
        return env, state_dim, action_dim, state_lows, state_highs

    elif env_name == 'sparse_reacher':
        print('choose sparse reacher')
        state_dim = 11
        action_dim = 2
        state_lows = [-1, -1, -8]
        state_highs = [1, 1, 8]
        action_lows = [-1]
        action_highs = [1]

        env = gym.make('Reacher-v4')

        # Wrap the environment with the custom reward wrapper
        env = SparseReacher(env)

        print(env)
        return env, state_dim, action_dim, state_lows, state_highs

    elif env_name == 'pointmaze':
        print('chose point maze')
        state_dim = 4
        action_dim = 2
        state_lows = [float('-inf'), float('-inf'), float('-inf'), float('-inf')]
        state_highs = [float('inf'), float('inf'), float('inf'), float('inf')]
        action_lows = [-1, -1]
        action_highs = [1, 1]
        env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100)
        env = PointMaze(env)

        return env, state_dim, action_dim, state_lows, state_highs
    
    elif env_name == 'pendulum':
        state_dim = 3
        action_dim = 1
        state_lows = [-1, -1, -8]
        state_highs = [1, 1, 8]
        env = gym.make('Pendulum-v1', g=9.81)
        min_action = -2
        max_action = 2
    
        return env, state_dim, action_dim, state_lows, state_highs
    else:
        print('choose inverted pendulum')

        # envs = gym.envs.registry.all()
        # env_ids = [env_spec.id for env_spec in envs]
        # print(env_ids)

        state_dim = 4
        action_dim = 1
        state_lows = [-1, -0.2, -1, -1]
        state_highs = [1, 0.2, 1, 1]
        #print('state')
        env = gym.make('InvertedPendulum-v4')
        min_action = -3
        max_action = 3
        #print('x', x)

    return env, state_dim, action_dim, state_lows, state_highs, action_lows, action_highs
