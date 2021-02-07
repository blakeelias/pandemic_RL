from pathlib import Path

from policies import policy_fn_generator, default_policy_fns
from test import trajectory_value


def compare_policies(env, gamma, custom_policies=()):
    b()
    custom_policy_fns = [(f'custom_policy_{i}', policy_fn_generator(policy)) for i, policy in enumerate(custom_policies)]
    policy_fns = default_policy_fns + custom_policy_fns
    return {policy_name: trajectory_value(env, policy_fn, policy_name, gamma) for policy_name, policy_fn in policy_fns}


def visualize_evaluation(policy_evaluations, results_dir, param_dict_1, param_dict_2, constant_params):
    file_path = Path(results_dir) / f'policy_evaluations_{vertical_param}_{horizontal_param}_{constant_params}.png'

    param_1_name = list(param_dict_1.keys())[0]
    param_2_name = list(param_dict_2.keys())[0]

    param_1_values = param_dict_1[param_1_name]
    param_2_values = param_dict_2[param_2_name]
    
    new_params = copy.deepcopy(constant_params)

    num_policies = len(list(policy_evaluations.values())[0].values())

    results_table = np.array((len(param_1_values), len(param_2_values), num_policies))
    
    for i, param_1 in enumerate(param_1_values):
        for j, param_2 in enumerate(param_2_values):
            new_params[param_1_name] = param_1
            new_params[param_2_name] = param_2

            results_table[i, j] = policy_evaluations[tuple(new_params.items())]
    b()

    
