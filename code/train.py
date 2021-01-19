from pdb import set_trace as b
from math import ceil

import gym
import numpy as np

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


def train_environment(env, discount_factor, convergence_threshold=0.0001):
    reward_param_str = env.reward_param_str + f',convergence_threshold={convergence_threshold},discount_factor={discount_factor}'
    file_name_prefix = f'../results/env=({env.dynamics_param_str})/reward=({reward_param_str})/'
    file_name = file_name_prefix + 'policy.pickle'

    try:
        policy, V = load_pickle(file_name)
        print('Loaded policy')
        return policy, V, file_name_prefix
    except:
        #if not env.P:
        #    env._set_transition_probabilities()
        policy, V = value_iteration(env,
                                    theta=convergence_threshold,
                                    discount_factor=discount_factor ** env.action_frequency,
                                    initial_value=0,
                                    horizon=env.horizon_effective)
        save_pickle((policy, V), file_name)
        return policy, V, file_name_prefix



if __name__ == '__main__':
    env = gym.make('pandemic-v0')
    policy, V = train_environment(env)
    test_environment(env, policy)
    
