from itertools import product
from collections import namedtuple
from pdb import set_trace as b
import traceback

import gym
from tqdm import tqdm
import argparse
import replicate
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from train import train_environment
from test import test_environment, plot_policy_trajectory, cost_of_trajectory
from policy_comparison import compare_policies, visualize_evaluation
from policies import default_policy_fns
from gym_pandemic.envs.pandemic_env import PandemicEnv
from gym_pandemic.envs.pandemic_immunity_env import PandemicImmunityEnv
from utils import combine_dicts


Params = namedtuple('Params', ['num_population', 'hospital_capacity_proportion', 'R_0', 'imported_cases_per_step', 'power', 'cost_of_R_1_over_cost_per_case', 'dynamics', 'distr_family', 'horizon', 'planning_horizon', 'action_frequency', 'vaccine_start', 'final_vaccinated', 'vaccine_schedule', 'initial_fraction_infected', 'tags'])


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

    parser.add_argument('--power',
                        metavar='power',
                        type=float,
                        nargs='+',
                        default=[1.0],
                        help='')

    parser.add_argument('--cost_of_R_1_over_cost_per_case',
                        metavar='cost_of_R_1_over_cost_per_case',
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

    parser.add_argument('--num_trials',
                        type=int,
                        default=1)

    
    parser.add_argument('--policy-comparison', dest='policy_comparison', action='store_true')
    parser.add_argument('--no-policy-comparison', dest='policy_comparison', action='store_false')
    parser.set_defaults(policy_comparison=True)

    parser.add_argument('--policy-comparison-time', dest='policy_comparison_time', action='store_true')
    parser.add_argument('--no-policy-comparison-time', dest='policy_comparison_time', action='store_false')
    parser.set_defaults(policy_comparison_time=True)

    
    parser.add_argument('--policy-optimization', dest='policy_optimization', action='store_true')
    parser.add_argument('--no-policy-optimization', dest='policy_optimization', action='store_false')
    parser.set_defaults(policy_optimization=False)
    
    return parser.parse_args()



def plot(args, comparisons):
    pass



def sample_trajectories(args, env):
    trials_policy_trajectories = []
    
    # Generate and plot trajectories in canonical environment
    print('Running trials in the canonical environment')
    # for k in tqdm(list(range(args.num_trials))):
    for k in range(args.num_trials):
        # Can run each policy just once, in a single environment, then evaluate the trajectory's cost
        # in all other environments, because they have the same state dynamics (just different reward function)
        policy_names, trajectories = compare_policies(env, args.discount_factor, default_policy_fns, load_cached=True, trial_num=k)
        trials_policy_trajectories.append((policy_names, trajectories))

        print(f'Running trial {k}')
        # for policy_name, trajectory in tqdm(zip(policy_names, trajectories)):
        for policy_name, trajectory in zip(policy_names, trajectories):
            extra_str = f'{policy_name}_trial_{k}'
            # plot_trajectory(trajectory, file_name)
            policy = None
            # put this back?
            # was this plotting with policy==None?
            # and no policy name included?

            
            # TODO: Put back if want to plot particular trajectories:
            # TODO: Should the center just be 0.5? 0.75?
            # plot_policy_trajectory(env, policy, trajectory, 'contact_rate', center=1.0 / env.R_0, extra_str=extra_str)

    return trials_policy_trajectories

                

def policy_comparison(args, experiment_parameters, parameters_sweep):
    # TODO: Make a command-line arg
    contact_factor_resolution_comparison = 0.01

    policy_evaluations = {}
    
    # sample_trajectories(args, envs[0])
    
    for i, particular_parameters in tqdm(list(enumerate(parameters_sweep))):
        print('Combining param dicts... ', end='')
        parameters = combine_dicts(particular_parameters._asdict(), experiment_parameters)
        print('Done')
        # print(f'Experiment {i}: {parameters}')
        
        #parameters['cost_per_case'] = env.cost_per_case
        #parameters['cost_of_R=1_lockdown'] = env._cost_of_contact_factor(env.actions_r.index(1.0))
        # TODO: put these back in -- better to have the actual cost rather than a multiplier
        
        optimized_policy = None
        policy_names = []

        print('After "else" block')
        print(f'policy_comparison: {args.policy_comparison}')
        
        if args.policy_comparison:
            print('Initializing environment... ', end='')
            env = PandemicEnv(
                **parameters,
                results_dir=args.results_dir,
                cap_infected_hospital_capacity=False,
                contact_factor_resolution=contact_factor_resolution_comparison
            )
            print('Done.')
            results = []

            
            trials_policy_trajectories = sample_trajectories(args, env)
            
            
            for k in range(args.num_trials):
                policy_names, trajectories = trials_policy_trajectories[k]
                print(f'Trial {k}')

                # TODO: make this work in the new format

                '''
                if args.policy_optimization:
                    optimized_policy_names, optimized_trajectories = compare_policies(env, discount_factor, [], custom_policies=[optimized_policy], load_cached=True, trial_num=k)
                    policy_names += optimized_policy_names
                    trajectories += optimized_trajectories
                '''

                
                trajectory_total_rewards = [cost_of_trajectory(trajectory, env, args.discount_factor) for trajectory in trajectories]

                # results.append((policy_names, values))
                params_key = tuple(sorted(tuple(parameters.items())))
                policy_evaluations.setdefault(params_key, [])
                policy_evaluations[params_key].append(trajectory_total_rewards)
            
        del env


        
    # Plot results
    if args.policy_comparison:
        constant_params = parameters # last parameters that were set
        
        # variable_params = ['cost_per_case', 'cost_of_R=1_lockdown']
        # variable_params = ['cost_per_case_scale_factor', 'extra_scale']

        # variable_params = ['power', 'cost_of_R_1_over_cost_per_case']
        variable_params = ['R_0', 'cost_of_R_1_over_cost_per_case']
        
        for param in variable_params:
            del constant_params[param]
            
        args_dict = vars(args)
        
        try:
            visualize_evaluation(
                policy_names,
                policy_evaluations,
                args.results_dir,
                {variable_params[0]: args_dict[variable_params[0]]},
                {variable_params[1]: args_dict[variable_params[1]]},
                constant_params,
                args.num_trials
            )
        except:
            b()

            
# TODO: Test this
def policy_optimization(args, experiment_parameters, parameters_sweep):
    # Main loop
    
    envs = [
        PandemicEnv(
            **combine_dicts(particular_parameters._asdict(), experiment_parameters),
            results_dir=args.results_dir,
            cap_infected_hospital_capacity=False,
            contact_factor_resolution=contact_factor_resolution_comparison, # TODO: This was for policy comparison. Turn back to 0.05 or 0.1 when doing optimization
        ) for particular_parameters in parameters_sweep
    ]

    
    print('Evaluating policy cost with respect to each reward function')
    for i, particular_parameters in tqdm(list(enumerate(parameters_sweep))):
        print('Combining param dicts... ', end='')
        parameters = combine_dicts(particular_parameters._asdict(), experiment_parameters)
        print('Done')
        # print(f'Experiment {i}: {parameters}')
        
        #parameters['cost_per_case'] = env.cost_per_case
        #parameters['cost_of_R=1_lockdown'] = env._cost_of_contact_factor(env.actions_r.index(1.0))
        # TODO: put these back in -- better to have the actual cost rather than a multiplier
        
        optimized_policy = None
        policy_names = []
        if args.policy_optimization:
            env = PandemicEnv(**parameters, results_dir=args.results_dir)
            
            optimized_policies, optimized_Vs = train_environment(env, args.discount_factor, parameters['planning_horizon'])
            
            optimized_policy = optimized_policies[-1]
            optimized_V = optimized_Vs[-1]
            
            policies[particular_parameters] = optimized_policy
            Vs[particular_parameters] = optimized_V
            
            print(particular_parameters)
            # For finite time horizon, these tests are less appropriate
            # Because the policy is time-varying
            test_environment(env, optimized_policy, optimized_V, args.discount_factor)
            
            # TODO: test environment with all the partial policies
            #   (1) display policy
            #   (2) follow the policy half-way [solid line]
            #   (3) [optional: dotted line following remainder of policy]
            #   (4) extended/lengthen policy
            #   (5) repeat (back to (1))
            #   ...

        del env


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
            args.power,
            args.cost_of_R_1_over_cost_per_case,
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

    policies = {}
    Vs = {}

    policy_evaluations = {}

    # plot_cost_curves(envs, f'{args.results_dir}/cost_of_lockdown.png')
    
    args.discount_factor = 1.0
    

    if args.policy_comparison:
        comparisons = policy_comparison(args, experiment_parameters, parameters_sweep)

    if args.policy_optimization:
        optimizations = policy_optimization(args, experiment_parameters, parameters_sweep)
        comparisons += optimizations


    plot(comparisons)
    
    
def plot_cost_curves(envs, filename):
    fig, ax = plt.subplots()
    
    xs = np.arange(0.05, 1, 0.01)
    for env in envs:
        ys = []
        for x in xs:
            y = env._cost_of_contact_factor(x)
            ys.append(y)
            
        ax.plot(xs, ys)

    ax.set_xlabel('Contact Factor', fontsize=14)
    ax.set_ylabel('Cost (Millions of Dollars)', fontsize=14)
    scale_y = 1e6
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    fig.savefig(filename)
    fig.clf()
    
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

