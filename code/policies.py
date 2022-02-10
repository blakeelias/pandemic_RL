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
    
    state = env.state_idx_to_obj(state_idx)
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
    
    state = env.state_idx_to_obj(state_idx)
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    min_action_idx = 4   # min R = 0.8 (with no immunity)
    
    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in range(min_action_idx, env.nA)])
    
    # Choose action with largest R_t such that num_cases <= target_cases
    # If none satisfy this, pick action with smallest R_t
    possible_new_infected = num_infected * R_ts
    valid_actions = np.where(possible_new_infected <= target_cases)[0]
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_actions]
        action = min_action_idx + valid_actions[valid_Rts.argmax()]
    else:
        action = min_action_idx + R_ts.argmin()

    return action



def policy_fn_cases_simple(env, state_idx, time_idx, target_cases):
    '''
    `target_cases`: Target number of cases for all time steps
    Chooses the max value of R such that expected number of cases is less than {target_cases}.
    '''
    
    state = env.state_idx_to_obj(state_idx)
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    min_action_idx = 0   # min R = 0.8 (with no immunity)
    max_action_idx = env.nA - 1
    mid_action_idx_1 = (min_action_idx + max_action_idx) // 4
    mid_action_idx_2 = (min_action_idx + max_action_idx) // 2
    mid_action_idx_3 = (3 * (min_action_idx + max_action_idx)) // 4

    # actions = [min_action_idx, mid_action_idx_1, mid_action_idx_2, mid_action_idx_3, max_action_idx]
    actions = list(range(0, env.nA))
    # actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]   

    
    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in actions])


    # General case:
    R_target = 0


    if num_infected < 1:
        R_target = np.inf
    elif num_infected < target_cases / 10:
        R_target = 2.0
    elif num_infected < target_cases / 5:
        R_target = 1.5
    elif num_infected < target_cases / 3:
        R_target = 1.25
    elif num_infected < 2 * target_cases / 3:
        R_target = 1.1
    elif num_infected < target_cases:
        R_target = 1.0
    elif num_infected < 1.1 * target_cases:
        R_target = 0.9
    elif num_infected < 1.5 * target_cases:
        R_target = 0.8
    elif num_infected < 2 * target_cases:
        R_target = 0.7
    else:
        R_target = 0.6


    # Give the virus a "head start":
    if time_idx < 8:
        R_target = 1.5

    if time_idx < 4:
        R_target = 2.0

    if time_idx < 2:
        R_target = np.inf




    
    # Set policy based on R:
        
    valid_actions = np.where(R_ts <= R_target)[0]
        
    # Choose action with largest R_t such that R_t <= R_target (if possible)
    # If none satisfy this, pick action with smallest R_t available
    
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_actions]
        action = min_action_idx + valid_actions[valid_Rts.argmax()]
    else:
        action = min_action_idx + R_ts.argmin()

    return action



def policy_fn_time(env, state_idx, time_idx, lockdown_duration, init_cases):
    '''
    `lockdown_duration`: How long to push cases down for, before switching to an R=1 strategy. If cases are 0, it will relax to R>1 as long as cases stay at 0, otherwise .
    Chooses the max value of R such that expected number of cases is less than {target_cases}.

    
    Note: `lockdown_duration` never needs to be longer than the expected amount of time to get to 0 cases. If it's longer, we will let it do that lockdown anyway, as a minimum.
    '''

    state = env.state_idx_to_obj(state_idx)
    num_susceptible, num_infected = state
    # susceptible_fraction = num_susceptible / env.num_population
    
    # TODO: allow continuous action space?

    min_action_idx = 4   # min R = 0.8 (with no immunity)

    if time_idx < lockdown_duration:
        return min_action_idx
    
    R_ts = np.array([env.R_t(action, time_idx, num_susceptible) for action in range(min_action_idx, env.nA)])
    

    lockdown_Rt = R_ts[0]
    target_cases = init_cases * (lockdown_Rt ** duration)
    
    
    return policy_fn_cases(env, state_idx, target_cases)
    


def policy_fn_cases_generator(target_cases):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_cases(env, state_idx, time_idx, target_cases)
    return policy_fn


def policy_fn_cases_simple_generator(target_cases):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_cases_simple(env, state_idx, time_idx, target_cases)
    return policy_fn


def policy_fn_time_generator(init_cases, lockdown_duration):
    def policy_fn(env, state_idx, time_idx):
        return policy_fn_time(env, state_idx, time_idx, lockdown_duration, init_cases)
    return policy_fn





possible_Rs = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5]
policy_fns_R = [(f'R={R}', policy_fn_R_generator(R)) for R in possible_Rs]

# possible_case_levels = [5, 10, 20, 50, 100, 200, 500, 1000]
# possible_case_levels = [0, 5, 10, 20, 50, 100, 1000, 10000]


# This one!
possible_case_levels = [0, 20, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000] # [20000, 30000, 40000, 50000]

policy_fns_case_level = [(f'target_cases={target_cases}', policy_fn_cases_generator(target_cases)) for target_cases in possible_case_levels]


policy_fns_case_level_simple = [(f'target_cases={target_cases}', policy_fn_cases_simple_generator(target_cases)) for target_cases in possible_case_levels]

# default_policy_fns = policy_fns_R + policy_fns_case_level
# default_policy_fns = policy_fns_case_level + [('do nothing', policy_fn_do_nothing)]

default_policy_fns = policy_fns_case_level_simple


def time_policy_fns(init_cases, max_duration):
    return [policy_fns_time_generator(init_cases, lockdown_duration) for lockdown_duration in range(max_duration)]
