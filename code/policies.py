import numpy as np

from pdb import set_trace as b


def policy_fn_generator(policy):
    def policy_fn(env, state_idx, time_idx):
        try:
            return policy[time_idx, state_idx].argmax()
        except:
            b()
    return policy_fn


def policy_fn_do_nothing(env, state_idx, time_idx):
    # Pick the action which applies no restrictions
    actions = list(range(env.nA))
    return actions[-1]


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
    `target_cases`: Target number of cases for all time steps
    Chooses the max value of R such that expected number of cases is less than {target_cases}.
    '''
    
    state = env.states[state_idx]
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    min_action_idx = 4   # min R = 0.8 (with no immunity)
    
    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in range(min_action_idx, env.nA)])
    
    # Choose action with largest R_t such that R_t <= 1
    # If none satisfy this, pick action with smallest R_t
    possible_new_infected = num_infected * R_ts
    valid_actions = np.where(possible_new_infected <= target_cases)[0]
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_actions]
        action = min_action_idx + valid_actions[valid_Rts.argmax()]
    else:
        action = min_action_idx + R_ts.argmin()

    return action


def policy_fn_cases_generator(target_cases):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_cases(env, state_idx, time_idx, target_cases)
    return policy_fn


possible_Rs = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5]
policy_fns_R = [(f'R={R}', policy_fn_R_generator(R)) for R in possible_Rs]

# possible_case_levels = [5, 10, 20, 50, 100, 200, 500, 1000]
possible_case_levels = [0, 5, 10, 20, 50, 100]
# possible_case_levels = [10]
policy_fns_case_level = [(f'target_cases={target_cases}', policy_fn_cases_generator(target_cases)) for target_cases in possible_case_levels]


# default_policy_fns = policy_fns_R + policy_fns_case_level
default_policy_fns = policy_fns_case_level + [('do nothing', policy_fn_do_nothing)]
