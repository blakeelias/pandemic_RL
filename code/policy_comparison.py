from pathlib import Path
from pdb import set_trace as b
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from policies import policy_fn_generator
from test import trajectory_generator
from utils import load_pickle, save_pickle


def compare_policies(env, gamma, default_policy_fns, custom_policies=(), load_cached=True, trial_num=0):
    file_path = Path(env.file_name_prefix) / f'policy_evaluations_trial_{trial_num}.pickle'
    try:
        if load_cached:
            print('Loading cached trajectory')
            return load_pickle(file_path)
        else:
            raise Exception() # i.e. skip to `except` clause
    except:
        custom_policy_fns = [(f'custom_policy_{i}', policy_fn_generator(policy)) for i, policy in enumerate(custom_policies)]
        policy_fns = default_policy_fns + custom_policy_fns
        policy_names = [policy_name for policy_name, policy_fn in policy_fns]
        policy_trajectories = [trajectory_generator(env, policy_fn, policy_name, gamma) for policy_name, policy_fn in policy_fns]

        # file_name = f'trial_{k}.png'
        
        #result = dict(list(zip(policy_names, policy_values)))
        result = (policy_names, policy_trajectories)
        save_pickle(result, file_path)
        return result


def evaluations_table(policy_names, policy_total_rewards, results_dir, param_dict_1, param_dict_2, constant_params, num_trials):
    param_1_name = list(param_dict_1.keys())[0]
    param_2_name = list(param_dict_2.keys())[0]

    param_1_values = param_dict_1[param_1_name]
    param_2_values = param_dict_2[param_2_name]

    # file_path = Path(results_dir) / f'policy_evaluations_{param_1_name}_{param_2_name}_{constant_params}.pickle'
    
    new_params = copy.deepcopy(constant_params)

    # num_policies = len(list(policy_evaluations.values())[0].values())

    results_table_raw = np.zeros((len(param_1_values), len(param_2_values), len(policy_names), num_trials))
    
    for i, param_1 in enumerate(param_1_values):
        for j, param_2 in enumerate(param_2_values):
            new_params[param_1_name] = param_1
            new_params[param_2_name] = param_2

            params_key = tuple(sorted(tuple(new_params.items())))
            try:
                trajectories_trials = policy_total_rewards[params_key]
            except:
                b()

            for trial_num, trials in enumerate(trajectories_trials):
                for k, policy_total_reward in enumerate(trials):
                    try:
                        results_table_raw[i, j, k, trial_num] = policy_total_reward
                    except:
                        b()

    results_table = np.average(results_table_raw, axis=-1)
                        
    return results_table
    
    
def visualize_evaluation(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params, num_trials):
    param_1_name = list(param_dict_1.keys())[0]
    param_2_name = list(param_dict_2.keys())[0]

    param_1_values = param_dict_1[param_1_name]
    param_2_values = param_dict_2[param_2_name]
    
    file_path = Path(results_dir) / f'policy_evaluations_{param_1_name,param_2_name}.png'  # _{constant_params

    table = evaluations_table(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params, num_trials)
    num_policies = table.shape[-1] # == len(policy_names)
    
    best_policies = table.argmax(axis=-1)

    '''best_policies = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 4, 4],
    ])'''

    best_policies_df = pd.DataFrame(best_policies)
    best_policies_df.columns = param_2_values
    best_policies_df.index = param_1_values
    
    sns.set(font_scale=0.7, rc={'figure.figsize':(25, 25)})

    ### Colors
    # White for 0-cases policy
    # Yellow / orange shades for N-cases policies
    # Red for do-nothing policy
    myColors = ((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 0.75, 0.0, 1.0), (1.0, 0.5, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0))
    # Green for optimized policy
    if num_policies > 5:
        myColors = myColors + ((0.0, 0.0, 0.8, 1.0),)
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    ### Heat map
    ax = sns.heatmap(best_policies_df, cmap=cmap, linewidths=.5, linecolor='lightgray', vmin=0, vmax=num_policies)
    
    # Manually specify colorbar labelling after it's been generated
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    tick_widths = [colorbar.vmin + 0.5 * r / (num_policies) + r * i / (num_policies) for i in range(num_policies)]
    colorbar.set_ticks(tick_widths)
    colorbar.set_ticklabels(policy_names)
    
    # X - Y axis labels
    ax.set_ylabel(param_1_name)
    ax.set_xlabel(param_2_name)

    plt.tight_layout()
    
    # Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
    _, labels = plt.yticks()
    plt.setp(labels, rotation=0)
    plt.savefig(file_path)
    
    return table
