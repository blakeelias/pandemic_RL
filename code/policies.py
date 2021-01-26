import numpy as np

from pdb import set_trace as b


def policy_fn_generator(policy):
    def policy_fn(env, state_idx, time_idx):
        b()
        return policy[state_idx].argmax()
    return policy_fn


def policy_fn_R(env, state_idx, time_idx, R_t):
    '''
    `R_t`: Target R value for all time steps. 
    Chooses the max value of R less than this R_t.
    '''
    
    state = env.states[state_idx]
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in range(env.nA)])
    # Choose action with largest R_t such that R_t <= 1
    # If none satisfy this, pick action with smallest R_t
    
    valid_actions = np.where(R_ts <= R_t)[0]
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_actions]
        action = valid_actions[valid_Rts.argmax()]
    else:
        action = R_ts.argmin()
        
    return action


def policy_fn_R_generator(R):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_R(env, state_idx, time_idx, R)
    return policy_fn


def policy_fn_cases(env, state_idx, time_idx, target_cases):
    '''
    `R_t`: Target R value for all time steps. 
    Chooses the max value of R less than this R_t.
    '''
    
    state = env.states[state_idx]
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in range(env.nA)])
    # Choose action with largest R_t such that R_t <= 1
    # If none satisfy this, pick action with smallest R_t

    possible_new_infected = num_infected * R_ts
    valid_actions = np.where(possible_new_infected <= target_cases)[0]
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_actions]
        action = valid_actions[valid_Rts.argmax()]
    else:
        action = R_ts.argmin()

    print(f'{num_susceptible}, {num_infected}, {R_ts[action]}, {num_infected * R_ts[action]}')
        
    return action


def policy_fn_cases_generator(target_cases):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_cases(env, state_idx, time_idx, target_cases)
    return policy_fn


possible_Rs = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5]
policy_fns_R = [policy_fn_R_generator(R) for R in possible_Rs]

possible_case_levels = [5, 10, 20, 50, 100, 200, 500, 1000]
policy_fns_case_level = [policy_fn_cases_generator(target_cases) for target_cases in possible_case_levels]


# default_policy_fns = policy_fns_R + policy_fns_case_level
default_policy_fns = policy_fns_case_level
