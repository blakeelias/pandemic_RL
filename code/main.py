from itertools import product
from collections import namedtuple

import gym
from tqdm import tqdm
import argparse

from train import train_environment
from test import test_environment
from gym_pandemic.envs.pandemic_env import PandemicEnv


Params = namedtuple('Params', ['imported_cases_per_step', 'power', 'extra_scale'])


def parse_args():
    parser = argparse.ArgumentParser(description='Boolean command-line')

    parser.add_argument('--imported_cases_per_step',
                        metavar='imported_cases_per_step',
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


def main(imported_cases_per_step_range=[0.0, 0.5, 1.0, 5.0, 10.0],
         powers=[0.1, 0.25, 0.5, 1.0, 1.5],
         extra_scale=[0.25, 1.0]):

    params_sweep = [Params(*params)
                    for params in product(imported_cases_per_step_range, powers, extra_scale)]

    policies = {}
    Vs = {}
    
    for params in tqdm(params_sweep):
        env = make_environment(params)
        policy, V = train_environment(env)
        policies[params] = policy
        Vs[params] = V

    for params in params_sweep:
        env = make_environment(params)
        print('params', params)
        test_environment(env, policies[params], Vs[params])
        
    
def make_environment(params: 'Params'):
    return PandemicEnv(imported_cases_per_step=params.imported_cases_per_step,
                       power=params.power,
                       scale_factor=power_scale_factor[params.power] * params.extra_scale)
    

if __name__ == '__main__':
    args = parse_args()

    main(imported_cases_per_step_range=args.imported_cases_per_step,
         powers=args.powers,
         extra_scale=args.extra_scale)
