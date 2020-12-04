from pdb import set_trace as b

import gym
import numpy as np

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


DISCOUNT_FACTOR=0.99

def train_environment(env, theta=0.0001, discount_factor=DISCOUNT_FACTOR):
    reward_param_str = env.reward_param_str + f',theta={theta},discount_factor={discount_factor}'
    file_name_prefix = f'../results/{env.dynamics_param_str}/policy_reward=({reward_param_str}),env=({env.dynamics_param_str})'
    file_name = file_name_prefix + '.pickle'

    try:
        policy, V = load_pickle(file_name)
        print('Loaded policy')
        return policy, V, file_name_prefix
    except:
        if not env.P:
            env._set_transition_probabilities()
        policy, V = value_iteration(env,
                                    theta=theta,
                                    discount_factor=discount_factor,
                                    initial_value=0)
        save_pickle((policy, V), file_name)
        return policy, V, file_name_prefix



if __name__ == '__main__':
    env = gym.make('pandemic-v0')
    policy, V = train_environment(env)
    test_environment(env, policy)
    

# python main.py --power_scale_factors 0.25 1.0 --imported_cases_per_step_range 0 0.5 --powers 0.1 0.25 0.5 1.0 1.5

# python main.py --power_scale_factors 0.25 1.0 --imported_cases_per_step_range 1.0 5.0 10.0 --powers 0.1 0.25 0.5 1.0 1.5
