from itertools import product
from collections import namedtuple
from pdb import set_trace as b

import gym
from tqdm import tqdm
import argparse
import replicate

from train import train_environment
from test import test_environment
from gym_pandemic.envs.pandemic_env import PandemicEnv
from utils import combine_dicts


Params = namedtuple('Params', ['imported_cases_per_step', 'power', 'extra_scale'])


def parse_args():
    parser = argparse.ArgumentParser(description='Boolean command-line')

    parser.add_argument('--imported_cases_per_step_range',
                        metavar='imported_cases_per_step_range',
                        type=float,
                        nargs='+',
                        default=[0.0, 0.5, 1.0, 5.0, 10.0], help='')

    parser.add_argument('--powers',
                        metavar='powers',
                        type=float,
                        nargs='+',
                        help='')

    parser.add_argument('--extra_scale',
                        metavar='extra_scale',
                        type=float,
                        nargs='+',
                        help='')

    return parser.parse_args()


power_scale_factor = {
    .1: 2000,
    .25: 700,
    .5: 275,
    1.0: 70,
    1.5: 22,
}


def main(args={
          'imported_cases_per_step_range': [0.0, 0.5, 1.0, 5.0, 10.0],
          'powers': [0.1, 0.25, 0.5, 1.0, 1.5],
          'extra_scale': [0.25, 1.0]
         }):

    experiment_parameters = {
        'distr_family': 'poisson',
        'dynamics': 'SIS',
        'time_lumping': False,
        'num_population': 1000,
        'initial_num_cases': 100,
        'R_0': 2.5
    }
    
    experiment = replicate.init(combine_dicts(args, experiment_parameters))
    
    parameters_sweep = [
        Params(*parameters) for parameters in product(
            args['imported_cases_per_step_range'],
            args['powers'],
            args['extra_scale']
        )
    ]

    policies = {}
    Vs = {}
    
    for particular_parameters in tqdm(parameters_sweep):
        parameters = combine_dicts(particular_parameters._asdict(), experiment_parameters)
        env = PandemicEnv(**parameters)
        policy, V = train_environment(env)
        policies[particular_parameters] = policy
        Vs[particular_parameters] = V

        print(particular_parameters)
        test_environment(env, policy, V)

    experiment.checkpoint(path="lookup_tables")

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
