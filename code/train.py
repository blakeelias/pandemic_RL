from pdb import set_trace as b
from math import ceil

import gym
import numpy as np

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


def train_environment(env, discount_factor, planning_horizon, convergence_threshold=0.0001):
    file_name = env.file_name_prefix + 'policy.pickle'

    total_horizon = env.horizon_effective
    
    try:
        policy, V = load_pickle(file_name)
        print('Loaded policy')
        return policy, V
    except:
        #if not env.P:
        #    env._set_transition_probabilities()
        policy, V = value_iteration(env,
                                    theta=convergence_threshold,
                                    discount_factor=discount_factor ** env.action_frequency,
                                    initial_value=0,
                                    horizon=planning_horizon)
        save_pickle((policy, V), file_name)
        return policy, V



if __name__ == '__main__':
    env = gym.make('pandemic-v0')
    policy, V = train_environment(env)
    test_environment(env, policy)
    
