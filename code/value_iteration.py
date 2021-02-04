# Courtesy of https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

from pdb import set_trace as b

import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")


def value_iteration(env, start_time, horizon, theta=0.0001, discount_factor=1.0, initial_value=0, max_steps=100):
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
    n_steps = horizon if horizon < np.inf else 1  # there's 1 more step at the end which has 0 cost
    
    V = np.zeros([n_steps + 1, env.nS]) * initial_value
    policy = np.zeros([n_steps, env.nS, env.nA])
    
    end_time = start_time + horizon
    time_step = end_time

    accum_steps = 0
    
    while time_step > start_time:
        print(f'time_step = {time_step}')
        # Stopping condition
        delta = 0
        accum_steps += 1

        V_new = np.zeros(env.nS) * initial_value
        policy_new = np.zeros([env.nS, env.nA])
        
        time_idx = time_step if horizon < np.inf else 0
        array_idx = time_step - start_time if horizon < np.inf else 0
        
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            try:
                A = one_step_lookahead(env, s, V[array_idx, :], discount_factor, time_idx)
            except:
                b()
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

        # Advance time step
        time_step -= 1
        new_array_idx = array_idx - 1 if horizon < np.inf else 0

        # Create new value and policy entries 
        V[new_array_idx, :] = V_new
        policy[new_array_idx, :, :] = policy_new

        # Check if we can stop 
        if horizon == np.inf and (delta < theta or accum_steps > max_steps):
            break

        
    return policy, V


def value_iteration_overlapping_horizons(
        env,
        theta=0.0001,
        discount_factor=1.0,
        initial_value=0,
        total_horizon=np.inf,
        planning_horizon=np.inf):

    if total_horizon == np.inf or planning_horizon == np.inf:
        start_time = 0
        horizon = np.inf
        policy, V = value_iteration(env, start_time, horizon, theta, discount_factor, initial_value)
        return ([policy], [V])

    ### Else: finite horizons on both

    total_horizon = int(total_horizon)
    planning_horizon = int(planning_horizon)
    
    if total_horizon % planning_horizon != 0:
        raise Exception('total_horizon must be a multiple of planning_horizon')
    
    n_steps = total_horizon + 1
    V_final = np.ones([total_horizon + 1, env.nS]) * initial_value
    policy_final = np.zeros([total_horizon, env.nS, env.nA])

    action_period = int(planning_horizon / 2)
    start_idx = 0

    policies = []
    Vs = []
    
    while start_idx + planning_horizon <= total_horizon:
        end_time = start_idx + planning_horizon

        policy, V = value_iteration(
            env,
            start_idx,
            planning_horizon,
            theta,
            discount_factor,
            initial_value
        )

        policies.append(policy)
        Vs.append(V)

        is_final_round = (start_idx + planning_horizon)
        if is_final_round:
            duration = planning_horizon
        else:
            duration = action_period
        # Could even have done this with `duration = planning_horizon` always, since this will get overwritten when needed
            
        V_final[start_idx : start_idx + duration, :] = V[:duration, :]
        policy_final[start_idx : start_idx + duration, :, :] = policy[:duration, :, :]
        start_idx += action_period

    return (policies + [policy_final], Vs + [V_final])


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
            try:
                A[a] += prob * (reward + discount_factor * V[next_state])
            except:
                b()
    return A
