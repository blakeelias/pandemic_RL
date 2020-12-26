# Courtesy of https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

from pdb import set_trace as b

import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")


def value_iteration(env, theta=0.0001, discount_factor=1.0, initial_value=0, horizon=np.inf):
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

    time_step = n_decisions
    while time_step > 0:
        print(f'time_step = {time_step}')
        # Stopping condition
        delta = 0

        V_new = np.zeros(env.nS) * initial_value
        policy_new = np.zeros([env.nS, env.nA])
        
        time_idx = time_step if horizon < np.inf else 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(env, s, V[time_idx, :], discount_factor, time_idx)
            best_action_value = np.max(A)
            best_action = np.argmax(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[time_idx, s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V_new[s] = best_action_value
            # Update the policy to take the best action
            policy_new[s, best_action] = 1.0

        new_time_idx = time_idx - 1 if horizon < np.inf else 0
        V[new_time_idx, :] = V_new
        policy[new_time_idx, :, :] = policy_new

        # Check if we can stop 
        if horizon == np.inf and delta < theta:
            break

        # Advance time step
        time_step -= 1
        
    return policy, V


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
        for prob, next_state, reward, done in env.transitions(state, a):
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A
