from itertools import product
from collections import namedtuple
from pdb import set_trace as b

import gym
from tqdm import tqdm
import argparse
import replicate
import numpy as np

from train import train_environment
from test import test_environment
from gym_pandemic.envs.pandemic_env import PandemicEnv
from gym_pandemic.envs.pandemic_immunity_env import PandemicImmunityEnv
from utils import combine_dicts


Params = namedtuple('Params', ['num_population', 'imported_cases_per_step', 'power', 'extra_scale', 'dynamics', 'distr_family', 'horizon', 'action_frequency', 'tags'])


def parse_args():
    parser = argparse.ArgumentParser(description='Boolean command-line')

    
    parser.add_argument('--num_population',
                        metavar='num_population',
                        type=int,
                        nargs='+',
                        default=[1000], help='')
    
    parser.add_argument('--imported_cases_per_step_range',
                        metavar='imported_cases_per_step_range',
                        type=float,
                        nargs='+',
                        default=[0.0, 0.5, 1.0, 5.0, 10.0], help='')

    parser.add_argument('--powers',
                        metavar='powers',
                        type=float,
                        nargs='+',
                        default=[1.0],
                        help='')

    parser.add_argument('--extra_scale',
                        metavar='extra_scale',
                        type=float,
                        nargs='+',
                        default=[1.0],
                        help='')

    parser.add_argument('--dynamics',
                        type=str,
                        nargs='+',
                        default=['SIS'],
                        help='"SIR" or "SIS"')

    parser.add_argument('--distr_family',
                        type=str,
                        nargs='+',
                        default=['nbinom'],
                        help='"nbinom", "poisson", or "deterministic"')

    parser.add_argument('--horizon',
                        type=float,
                        nargs='+',
                        default=[np.inf],
                        help='Time horizon over which to optimize.')

    parser.add_argument('--action_frequency',
                        type=float,
                        nargs='+',
                        default=[1],
                        help='Frequency (in time steps) to allow agent to set a new action')
    
    parser.add_argument('--tags',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Custom argument to be recorded in output directory name')
    
    
    return parser.parse_args()


power_scale_factor = {
    .1: 2000,
    .25: 700,
    .5: 275,
    1.0: 70,
    1.5: 22,
}


def main(args):
    experiment_parameters = {
        'time_lumping': False,
        #'num_population': args['num_population'],
        'initial_fraction_infected': 0.1,
        'R_0': 2.5
    }
    
    # experiment = replicate.init(combine_dicts(args, experiment_parameters))
    
    parameters_sweep = [
        Params(*parameters) for parameters in product(
            args.num_population,
            args.imported_cases_per_step_range,
            args.powers,
            args.extra_scale,
            args.dynamics,
            args.distr_family,
            args.horizon,
            args.action_frequency,
            args.tags,
        )
    ]

    policies = {}
    Vs = {}
    
    for particular_parameters in tqdm(parameters_sweep):
        parameters = combine_dicts(particular_parameters._asdict(), experiment_parameters)
        if parameters['dynamics'] == 'SIR':
            env = PandemicImmunityEnv(**parameters)
        else:
            env = PandemicEnv(**parameters)
        policy, V, file_name_prefix = train_environment(env)
        policies[particular_parameters] = policy
        Vs[particular_parameters] = V

        print(particular_parameters)
        test_environment(env, policy, V, file_name_prefix)

    # experiment.checkpoint(path="lookup_tables")

if __name__ == '__main__':
    args = parse_args()
    main(args)

    #try:
    #main(args={
    #    'imported_cases_per_step_range': [0.0],
    #    'powers': [1.0],
    #    'extra_scale': [10.0/7],
    #    'num_population': [100]
    #})
    #except:
    #    pass
    #    # b()

