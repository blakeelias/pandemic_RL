from itertools import product
from collections import namedtuple
from pdb import set_trace as b
import traceback

import gym
from tqdm import tqdm
import argparse
import replicate
import numpy as np

from train import train_environment
from test import test_environment, compare_policies
from gym_pandemic.envs.pandemic_env import PandemicEnv
from gym_pandemic.envs.pandemic_immunity_env import PandemicImmunityEnv
from utils import combine_dicts


Params = namedtuple('Params', ['num_population', 'hospital_capacity_proportion', 'R_0', 'imported_cases_per_step', 'power', 'extra_scale', 'dynamics', 'distr_family', 'horizon', 'planning_horizon', 'action_frequency', 'vaccine_start', 'vaccine_final_susceptible', 'vaccine_schedule', 'initial_fraction_infected', 'tags'])


def parse_args():
    parser = argparse.ArgumentParser(description='Boolean command-line')


    parser.add_argument('--results_dir',
                        metavar='results_dir',
                        type=str,
                        default='../results',
                        help='Where to save results, relative to code directory')
    
    parser.add_argument('--num_population',
                        metavar='num_population',
                        type=int,
                        nargs='+',
                        default=[1000], help='Size of total population')

    parser.add_argument('--hospital_capacity_proportion',
                        metavar='hospital_capacity_proportion',
                        type=float,
                        nargs='+',
                        default=[0.01], help='Maximum fraction of population to allow infected at one time step')
    
    parser.add_argument('--R_0',
                        metavar='R_0',
                        type=float,
                        nargs='+',
                        default=[2.5], help='Initial reproductive number')
    
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
                        help='Time horizon over which problem is defined.')
    
    parser.add_argument('--planning_horizon',
                        type=float,
                        nargs='+',
                        default=[np.inf],
                        help='Time horizon over which to optimize.')

    parser.add_argument('--action_frequency',
                        type=int,
                        nargs='+',
                        default=[1],
                        help='Frequency (in time steps) to allow agent to set a new action')

    parser.add_argument('--vaccine_schedule',
                        type=str,
                        nargs='+',
                        default=['none'])
    
    parser.add_argument('--vaccine_start',
                        type=float,
                        nargs='+',
                        default=[0])

    parser.add_argument('--vaccine_final_susceptible',
                        type=float,
                        nargs='+',
                        default=[1])

    parser.add_argument('--initial_fraction_infected',
                        type=float,
                        nargs='+',
                        default=[0.001])
    
    parser.add_argument('--tags',
                        type=str,
                        nargs='+',
                        default=[None],
                        help='Custom argument to be recorded in output directory name')

    parser.add_argument('--policy-comparison', dest='policy_comparison', action='store_true')
    parser.add_argument('--no-policy-comparison', dest='policy_comparison', action='store_false')
    parser.set_defaults(policy_comparison=True)

    parser.add_argument('--policy-optimization', dest='policy_optimization', action='store_true')
    parser.add_argument('--no-policy-optimization', dest='policy_optimization', action='store_false')
    parser.set_defaults(policy_optimization=False)
    
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
    }
    
    # experiment = replicate.init(combine_dicts(args, experiment_parameters))
    
    parameters_sweep = [
        Params(*parameters) for parameters in product(
            args.num_population,
            args.hospital_capacity_proportion,
            args.R_0,
            args.imported_cases_per_step_range,
            args.powers,
            args.extra_scale,
            args.dynamics,
            args.distr_family,
            args.horizon,
            args.planning_horizon,
            args.action_frequency,
            args.vaccine_start,
            args.vaccine_final_susceptible,
            args.vaccine_schedule,
            args.initial_fraction_infected,
            args.tags,
        )
    ]

    policies_by_param = {}
    Vs_by_param = {}

    
    
    discount_factor = 1.0

    for i, particular_parameters in enumerate(parameters_sweep):
        try:
            parameters = combine_dicts(particular_parameters._asdict(), experiment_parameters)
            print(f'Experiment {i}: {parameters}')
        
            env = PandemicEnv(**parameters, results_dir=args.results_dir)
            
            policy = None
            if args.policy_optimization:
                planning_horizon = int(parameters['planning_horizon'])
                optimized_policies, optimized_Vs = train_environment(env, discount_factor, planning_horizon)

                policy = optimized_policies[-1]
                V = optimized_Vs[-1]
                
                policies_by_param[particular_parameters] = policy
                Vs_by_param[particular_parameters] = V
                
                print(particular_parameters)
                
                # For finite time horizon, these tests are less appropriate
                # Because the policy is time-varying

                for i, (policy, V) in enumerate(zip(optimized_policies, optimized_Vs)):
                    max_time = (i + 1) * planning_horizon
                    test_environment(env, policy, V, discount_factor, max_time)

                # TODO: test environment with all the partial policies
                #   (1) display policy
                #   (2) follow the policy half-way [solid line]
                #   (3) [optional: dotted line following remainder of policy]
                #   (4) extended/lengthen policy
                #   (5) repeat (back to (1))
                #   ...
                
            if args.policy_comparison:
                if args.policy_optimization:
                    values = compare_policies(env, discount_factor, custom_policies=[policy])
                else:
                    values = compare_policies(env, discount_factor)

                print('Policy Comparison:')
                print(values)

            del env
        except:
            try:
                print(f'Exception for parameters={parameters}:')
                print('-' * 60)
                traceback.print_exc()
                print('-' * 60)
                print('Continuing with next set of parameters...')
                print('-' * 60)
            except:
                print('Exception (could not print...)')
            continue
        
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

