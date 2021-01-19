import numpy as np


def policy_fn_generator(policy):
    def policy_fn(env, state_idx, time_idx):
        return policy[state_idx].argmax()
    return policy_fn


def policy_fn_R_eq_1(env, state_idx, time_idx):
    state = env.states[state_idx]
    num_susceptible, num_infected = state

    # TODO: allow continuous action space?

    R_ts = np.array([env.R_t(action, time_idx) for action in range(env.nA)])
    # Choose action with largest R_t such that R_t <= 1
    # If none satisfy this, pick action with smallest R_t
    
    valid_actions = np.where(R_ts <= 1)[0]
    if valid_actions.shape[0] > 0:
        valid_Rts = R_ts[valid_idxs]
        action = valid_actions[valid_Rts.argmax()]
    else:
        action = R_ts.argmin()
        
    print(f'Action index chosen: {action}')
    print(f'R_t: {R_ts[action]}')
    
    return action
