Repositories for the following papers:
<ul>
<li>Beyond exponentially fast mixing in average-reward reinforcement learning via multi-level Monte Carlo actor-critic(https://proceedings.mlr.press/v202/suttle23a.html)</li>
<li>CCE: Sample Efficient Sparse Reward Policy Learning for Robotic Navigation via Confidence-Controlled Exploration(https://arxiv.org/abs/2306.06192)</li>
<li>Towards Global Optimality for Practical Average Reward Reinforcement Learning without Mixing Time Oracles(https://icml.cc/virtual/2024/poster/34664)</li>
</ul>

Collection of agents and environments for testing mixing rate and MLMC algorithms. Adapted from wessle/inforatio.

To run on gridworld environments

```python3 run_gridworld_env.py```

List of arguments:

```--pg ``` default = "AC", the policy-gradient algorithm of choice. Right now it supports discrete version of AC and REINFORCE

```--iterations_per_ep``` default=200, the number of iterations or samples per episode

```--episodes``` default = 350, number of episodes

```--episode_save_freq```, default = 1, how often to save the data in terms of episodes

```--policy_lr```, default = 0.01, the policy learning rate, if using an adaptive optimizer like Adagrad or Adam, this is the initial value

```--value_lr```, default = 0.01, the critic learning rate, if using an adaptive optimizer like Adagrad or Adam, this is the initial value


```--grid_length```, default = 5, the length of one side of the 2D grid. So total number of states is ```grid_length ** 2```

```--horizon_parameter```, default = 4, if using MLMC gradient estimator, is the log Tmax. If using regular average, then this is trajectory length

```--is_mlmc```, default = 1, if 1 using MLMC gradient estimator. If 0, usign regular average.

```--adaptive_horizon```, default = 0, if 1 using adaptive horizon parameter based on policy entropy. If 0, using fixed

```--state_adaptive```, default = 1, only applies if adaptive horizon parameter is turned on. If 1, using the current state policy entropy to determine the horizon parameter of the next rollout. If 0, calculates the average policy entropy across all states at the end of the episode to determine the horizon parameter for the next episode.

```--adaptive_horizon_sensitivity_parameter```, default=20, determines how sensitive the horizon parameter is to policy change

```--env_name```, default = "2D_grid", the name of the environment. If "2D_grid_no_stay", the agent does not have the option to stay at current node.

```--seed_num```, default = 0, the random seed of the trial

```--reset_to_start```, default = 1, if 1, at the start of each episode, go back to top left. If 0, the episode starts at current node

```--keep_goal_same```, default = 1, if 1, the goal is always bottom right. If 0, new goal is randomly chosen based on ```seed_num```

```--save_locally```, default = 1, if 1, save the results of experiment locally in a folder. If 0, does not.

```--save_wandb```, default = 1, if 1, save the results of experiment to wandb account. If 0, does not.
