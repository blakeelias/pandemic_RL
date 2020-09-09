import gym
from tqdm import tqdm
import argparse

from train import train_environment
from test import test_environment
from gym_pandemic.envs.pandemic_env import PandemicEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Boolean command-line')

    parser.add_argument('--imported_cases_per_step', metavar='imported_cases_per_step', type=float, nargs='+', default=[0.0, 0.5, 1.0, 5.0, 10.0], help='')

    parser.add_argument('--powers', metavar='powers', type=float, nargs='+', help='')

    parser.add_argument('--extra_scale', metavar='extra_scale', type=float, nargs='+', help='')

    return parser.parse_args()

def main(imported_cases_per_step_range=[0.0, 0.5, 1.0, 5.0, 10.0],
         powers=[0.1, 0.25, 0.5, 1.0, 1.5],
         extra_scale=[0.25, 1.0]):
    # imported_cases_per_step_range = [0, 0.5, 1.0, 5.0, 10.0])
    # power_range = np.arange(1.0, 3.0, 0.5)
    # scale_factor_range = [100, 500, 1000]

    # power_scale_factor_range = [0.25, 1]

    power_scale_factor = {
        .1: 2000,
        .25: 700,
        .5: 275,
        1.0: 70,
        1.5: 22
    }

    print(imported_cases_per_step_range)
    print(powers)
    print(extra_scale)
    
    for imported_cases_per_step in tqdm(imported_cases_per_step_range):
        for power in tqdm(powers):
            for extra in tqdm(extra_scale):
                env = PandemicEnv(imported_cases_per_step=imported_cases_per_step,
                                  power=power,
                                  scale_factor=power_scale_factor[power]*extra)
                policy, V = train_environment(env)
                test_environment(env, policy)
                
    #    for power in tqdm(power_range):
    #        for scale_factor in tqdm(scale_factor_range):


if __name__ == '__main__':
    args = parse_args()

    main(imported_cases_per_step_range=args.imported_cases_per_step,
         powers=args.powers,
         extra_scale=args.extra_scale)
