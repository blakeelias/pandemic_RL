from pathlib import Path

from policies import policy_fn_generator, default_policy_fns
from test import trajectory_value


def compare_policies(env, gamma, custom_policies=()):
    b()
    custom_policy_fns = [(f'custom_policy_{i}', policy_fn_generator(policy)) for i, policy in enumerate(custom_policies)]
    policy_fns = default_policy_fns + custom_policy_fns
    return {policy_name: trajectory_value(env, policy_fn, policy_name, gamma) for policy_name, policy_fn in policy_fns}


def visualize_evaluation(policy_evaluations, results_dir, vertical_param='extra_scale', horizontal_param='cost_per_case', constant_params={}):
    file_path = Path(results_dir) / f'policy_evaluations_{vertical_param}_{horizontal_param}_{constant_params}.png'

    b()

    
