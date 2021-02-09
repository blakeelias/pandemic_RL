from pathlib import Path
from pdb import set_trace as b
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from policies import policy_fn_generator
from test import trajectory_value
from utils import load_pickle, save_pickle


def compare_policies(env, gamma, default_policy_fns, custom_policies=()):
    file_path = Path(env.file_name_prefix) / 'policy_evaluations.pickle'
    try:
        return load_pickle(file_path)
    except:
        custom_policy_fns = [(f'custom_policy_{i}', policy_fn_generator(policy)) for i, policy in enumerate(custom_policies)]
        policy_fns = default_policy_fns + custom_policy_fns
        policy_names = [policy_name for policy_name, policy_fn in policy_fns]
        policy_values = [trajectory_value(env, policy_fn, policy_name, gamma) for policy_name, policy_fn in policy_fns]

        #result = dict(list(zip(policy_names, policy_values)))
        result = (policy_names, policy_values)
        save_pickle(result, file_path)
        return result


def evaluations_table(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params):
    param_1_name = list(param_dict_1.keys())[0]
    param_2_name = list(param_dict_2.keys())[0]

    param_1_values = param_dict_1[param_1_name]
    param_2_values = param_dict_2[param_2_name]

    # file_path = Path(results_dir) / f'policy_evaluations_{param_1_name}_{param_2_name}_{constant_params}.pickle'
    
    new_params = copy.deepcopy(constant_params)

    # num_policies = len(list(policy_evaluations.values())[0].values())

    results_table = np.zeros((len(param_1_values), len(param_2_values), len(policy_names)))
    
    for i, param_1 in enumerate(param_1_values):
        for j, param_2 in enumerate(param_2_values):
            new_params[param_1_name] = param_1
            new_params[param_2_name] = param_2

            params_key = tuple(sorted(tuple(new_params.items())))
            try:
                values = policy_evaluations[params_key]
            except:
                b()
                
            for k, policy_value in enumerate(values):
                try:
                    results_table[i, j, k] = policy_value
                except:
                    b()

    return results_table
    
    
def visualize_evaluation(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params):
    param_1_name = list(param_dict_1.keys())[0]
    param_2_name = list(param_dict_2.keys())[0]

    param_1_values = param_dict_1[param_1_name]
    param_2_values = param_dict_2[param_2_name]
    
    file_path = Path(results_dir) / f'policy_evaluations_{param_1_name,param_2_name}.png'  # _{constant_params

    table = evaluations_table(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params)
    
    best_policies = table.argmax(axis=-1)
        
    sns.set(font_scale=0.8, rc={'figure.figsize':(10, 6)})
    # dataFrame = pandas.read_csv('LUH2_trans_matrix.csv').set_index(['Unnamed: 0'])
    
    
    # For only three colors, it's easier to choose them yourself.
    # If you still really want to generate a colormap with cubehelix_palette instead,
    # add a cbar_kws={"boundaries": linspace(-1, 1, 4)} to the heatmap invocation
    # to have it generate a discrete colorbar instead of a continous one.
    myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
    ax = sns.heatmap(best_policies, cmap=cmap, linewidths=.5, linecolor='lightgray')
    
    # Manually specify colorbar labelling after it's been generated
    colorbar = ax.collections[0].colorbar
    # colorbar.set_ticks([0.9, 1.7, 2.5, 3.3, 4.1])

    r = colorbar.vmax - colorbar.vmin
    n = table.shape[-1]
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
    colorbar.set_ticklabels(policy_names)
    
    # X - Y axis labels
    ax.set_ylabel(param_2_name)
    ax.set_xlabel(param_1_name)
    
    # Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
    _, labels = plt.yticks()
    plt.setp(labels, rotation=0)
    
    plt.savefig(file_path)
    
    # save to file_path
    
    return table
