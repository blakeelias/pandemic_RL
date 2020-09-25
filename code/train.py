from pdb import set_trace as b

import gym
import numpy as np

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


def train_environment(env, theta=0.0001, discount_factor=0.99):
    reward_param_str = env.reward_param_str + f',theta={theta},discount_factor={discount_factor}'
    file_name = f'../lookup_tables/{env.dynamics_param_str}/policy_reward=({reward_param_str}),env=({env.dynamics_param_str}).pickle'

    try:
        policy, V = load_pickle(file_name)
        print('Loaded policy')
        return policy, V
    except:
        if not env.P:
            env._set_transition_probabilities()
        b()
        policy, V = value_iteration(env,
                                    theta=theta,
                                    discount_factor=discount_factor,
                                    initial_value=-np.inf)
        save_pickle((policy, V), file_name)
        return policy, V



if __name__ == '__main__':
    env = gym.make('pandemic-v0')
    policy, V = train_environment(env)
    test_environment(env, policy)
    

# python main.py --power_scale_factors 0.25 1.0 --imported_cases_per_step_range 0 0.5 --powers 0.1 0.25 0.5 1.0 1.5

# python main.py --power_scale_factors 0.25 1.0 --imported_cases_per_step_range 1.0 5.0 10.0 --powers 0.1 0.25 0.5 1.0 1.5
