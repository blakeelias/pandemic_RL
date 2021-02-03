# Courtesy of https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

from pdb import set_trace as b

import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")


def value_iteration(env, theta=0.0001, discount_factor=1.0, initial_value=0, horizon=np.inf, end_time=np.inf):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        horizon: Number of time steps over which we are interested in maximizing
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    n_decisions = horizon
    n_steps = horizon + 1 if horizon < np.inf else 1  # there's 1 more step at the end which has 0 cost
    
    V = np.zeros([n_steps, env.nS]) * initial_value
    policy = np.zeros([n_steps, env.nS, env.nA])

    start_time = end_time - n_steps
    
    time_step = end_time
    while time_step > start_time:
        print(f'time_step = {time_step}')
        # Stopping condition
        delta = 0

        V_new = np.zeros(env.nS) * initial_value
        policy_new = np.zeros([env.nS, env.nA])
        
        time_idx = time_step if horizon < np.inf else 0
        array_idx = time_step - start_time if horizon < np.inf else 0
        
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(env, s, V[array_idx, :], discount_factor, time_idx)
            best_action_value = np.max(A)
            best_action = np.argmax(A)
            #if np.isnan(best_action_value) or best_action_value == -np.inf:
            #    b()

            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[array_idx, s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V_new[s] = best_action_value
            # Update the policy to take the best action
            policy_new[s, best_action] = 1.0

        new_array_idx = array_idx - 1 if horizon < np.inf else 0
        V[new_array_idx, :] = V_new
        policy[new_array_idx, :, :] = policy_new

        # Check if we can stop 
        if horizon == np.inf and delta < theta:
            break

        # Advance time step
        time_step -= 1
        
    return policy, V


def value_iteration_overlapping_horizons(
        env,
        theta=0.0001,
        discount_factor=1.0,
        initial_value=0,
        total_horizon=np.inf,
        planning_horizon=np.inf):

    if total_horizon == np.inf or planning_horizon == np.inf:
        return value_iteration(env, theta, discount_factor, initial_value, horizon=np.inf)

    ### Else: finite horizons on both

    total_horizon = int(total_horizon)
    planning_horizon = int(planning_horizon)
    
    if total_horizon % planning_horizon != 0:
        raise Exception('total_horizon must be a multiple of planning_horizon')
    
    n_steps = total_horizon + 1
    V_final = np.ones([n_steps, env.nS]) * initial_value
    policy_final = np.zeros([n_steps, env.nS, env.nA])

    action_period = int(planning_horizon / 2)
    start_idx = 0

    while start_idx + planning_horizon <= total_horizon:
        end_time = start_idx + planning_horizon

        policy, V = value_iteration(
            env,
            theta,
            discount_factor,
            initial_value,
            horizon=planning_horizon,
            end_time=end_time
        )
        V_final[start_idx : start_idx + action_period, :] = V[:action_period, :]
        policy_final[start_idx : start_idx + action_period, :, :] = policy[start_idx : start_idx + action_period, :, :]
        start_idx += planning_horizon
    
    

def invalid_number(x):
    #return (np.isnan(x) or x == -np.inf)
    return np.isnan(x)


def one_step_lookahead(env, state, V, discount_factor, time_idx):
    """
    Helper function to calculate the value for all action in a given state.
        
    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS
        
    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.transitions(state, a, time_idx):
            if invalid_number(V[next_state]):
                b()
            if invalid_number(reward):
                b()
            
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A
