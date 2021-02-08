from pathlib import Path
from pdb import set_trace as b
import copy

import numpy as np

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
    
    file_path = Path(results_dir) / f'policy_evaluations_{param_1_name}_{param_2_name}_{constant_params}.png'

    table = evaluations_table(policy_names, policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params)

    b()

    # save to file_path
    
    return table
