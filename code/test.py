from pdb import set_trace as b
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import axes3d
from einops import rearrange

from value_iteration import one_step_lookahead
from policies import policy_fn_generator, default_policy_fns


Trajectory = namedtuple('Trajectory', ['times', 'num_susceptible_t', 'num_infected_t', 'action_taken_t', 'cost_t', 'total_reward'])

def test_environment(env, policy, V, discount_factor):
    # policy: (time, env.nS, env.nA)
    # V: (time, env.nS)
    
    ### Extract policy
    policy_idxs = policy.argmax(axis=-1)
    # (time, env.nS)
    policy_rs = np.array([[env.actions_r[policy_idxs[i, j]] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])
    # (time, env.nS)
    policy_contact_rates = np.array([[env.contact_factor[policy_idxs[i, j]] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])
    # (time, env.nS)

    ### Generate Trajectory
    policy_fn = policy_fn_generator(policy)
    trajectory = trajectory_value(env, policy_fn, 'optimized_policy', discount_factor)

    ### Save CSVs
    save_policy_csvs(env, policy_rs, policy_contact_rates)
    save_value_csv(env, V)
    save_trajectory_csv(env, trajectory)
    
    ### Plot full policy
    plot_policy_trajectory(env, policy_rs, trajectory, 'R', center=1.0)
    plot_policy_trajectory(env, policy_contact_rates, trajectory, 'contact_rate', center=1.0/2.5)
    
    env.close()

    
def save_policy_csvs(env, policy_rs, policy_contact_rates):
    # Policy with respect to R ideally achieved at the given contact rate (with no influence of vaccination or other factors)
    policy_dict = {f't={t}': policy_rs[t, :] for t in range(policy_rs.shape[0])}
    policy_dict['state'] = range(env.nS)
    df_policy = pd.DataFrame(policy_dict)
    df_policy.to_csv(env.file_name_prefix + 'policy_R.txt')

    # Policy with respect to contact rate achieved
    policy_dict = {f't={t}': policy_contact_rates[t, :] for t in range(policy_contact_rates.shape[0])}
    policy_dict['state'] = range(env.nS)
    df_policy = pd.DataFrame(policy_dict)
    df_policy.to_csv(env.file_name_prefix + 'policy_contact_rate.txt')


def save_value_csv(env, V):
    # V: (time, env.nS)
    value_dict = {f't={t}': V[t, :] for t in range(V.shape[0])}
    value_dict['state'] = range(env.nS)
    df_value = pd.DataFrame(value_dict)
    df_value.to_csv(env.file_name_prefix + 'value.txt')


def save_trajectory_csv(env, trajectory):
    df = pd.DataFrame(trajectory._asdict())
    df.to_csv(env.file_name_prefix + 'trajectory.txt')

    
def plot_policy_trajectory(env, policy, trajectory, policy_type_str, center=1.0):
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('lockdown', [(0.0, 'red'), (0.25, 'red'), (0.5, 'white'), (1, 'green')])
    
    # With respect to R
    # ax = sns.heatmap(policy_rs[:-1, :].T, linewidths=0.5, center=1.0, cmap='RdYlGn')
    # To show policy values: use `annot=True`
    # Round to integer: `fmt='d'` (gives error for floats)
    # To hide x axis ticks: `xticklabels=False`
    # TODO: label x and y axes
    # TODO: better color scheme
    # ax.invert_yaxis()
    # ax.get_figure().savefig(file_name_prefix + 'policy_R.png')
    # color_map = matplotlib.colors.LinearSegmentedColormap.from_list('lockdown', [(0.0, 'red'), (0.5/env.R_0, 'red'), (1.0/env.R_0, 'white'), (1, 'green')])

    # color_map = sns.color_palette("vlag_r", as_cmap=True)
    f, ax = plt.subplots(figsize=(11, 9))
    plt.tick_params(bottom='on')
    
    ax = sns.heatmap(policy[:-1, :].T, center=center, cmap=color_map) # 'RdYlGn')
    ax.invert_yaxis()

    if trajectory:
        # Add trajectory plot on to heat map
        ax2 = ax # .twinx().twiny()
        # sns.lineplot(data=trajectory.num_infected_t, linewidth=2, ax=ax2)
        sns.lineplot(x=list(range(len(trajectory.num_infected_t))), y=trajectory.num_infected_t, linewidth=2, ax=ax2)  # x=trajectory.times
        ax.axis('tight')

    ax.get_figure().savefig(env.file_name_prefix + f'policy_{policy_type_str}.png')


def plot_policy(best_action_idx):
    ax = sns.barplot(list(range(num_states)),
                [actions[int(best_action_idx[n])] for n in range(num_states)])
    # ax.set(xlabel='common xlabel', ylabel='common ylabel')
    ax.set_xticks(range(int(len(states) / 1)))
    ax.set_xticklabels([state for i, state in enumerate(states) if i % 1 == 0])
    plt.show()
    

def plot_value_function(env, policy, V):
    state_to_value = {}
    state_to_action = {}

    for packed_state in env.states:
        idx = env.state_to_idx[packed_state]
        unpacked_state = env._unpack_state(packed_state)
        state_to_value[unpacked_state] = V[idx]
        state_to_action[unpacked_state] = env.actions_r[policy[idx].argmax()]

    def reshape(array):
        h = env.observation_space.high[0] - env.observation_space.low[0] + 1
        relevant_array = array[:-1] # remove last entry, for 'saturated state'
        return rearrange(relevant_array, '(h w) -> h w', h=h)

    unpacked_states = [env._unpack_state(packed_state) for packed_state in env.states]

    X = reshape(np.array([state[0] for state in unpacked_states]))
    Y = reshape(np.array([state[1] for state in unpacked_states]))
    Z_value = reshape(np.array([state_to_value[state] for state in unpacked_states]))
    Z_policy = reshape(np.array([state_to_action[state] for state in unpacked_states]))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #print('Value function')
    
    #ax.bar_wireframe(X, Y, Z_value, rstride=10, cstride=10)
    #plt.show()

        
    #fig = plt.figure()
    print('Policy')
    ax.plot_wireframe(X, Y, Z_policy, rstride=10, cstride=10)
    plt.show()
    
    
def trajectory_value(env, policy_fn, policy_name, gamma):
    file_name_prefix = env.file_name_prefix + f'/policy={policy_name}/'
    Path(file_name_prefix).mkdir(parents=True, exist_ok=True)
    
    ### Trajectory
    # Step through a trajectory
    # TODO: put this back and actually plot a few trajectories
    # Must change to use time-varying policy

    # Time Series to Save
    num_susceptible_t = []
    num_infected_t = []
    actions_taken_t = []
    R_t = []
    cost_t = []
    
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1

    # Track trajectory as it's generated (no longer needed if plots work)
    # print(f'initial num infected: {env.state}')
    # print('{num_susceptible}, {num_infected}, {R_ts[action]}, {num_infected * R_ts[action]}')
    # plot_value_function(env, policy, V)

    
    t = 0
    while t < min(env.horizon, 100):
        try:
            new_state = env.states[observation]
        except:
            b()
        if type(new_state) == tuple:
            num_susceptible, num_infected = new_state
            num_infected_t.append(num_infected)
            num_susceptible_t.append(num_susceptible)
        else:
            num_infected_t.append(new_state)

        if t % env.action_frequency == 0:
            # Get best action
            # Allowed to take a new action once every {env.action_frequency} steps
            # observation = min(observation, env.nS - 1) # max number infected
            #b()
            state_idx = observation
            action = policy_fn(env, state_idx, t)
            
        actions_taken_t.append(env.contact_factor[action])
        observation, reward, done, info = env.step(action)
        cost_t.append(reward)
        R_t.append(env.R_t(action, t, num_susceptible))
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma

        t += 1

    # Duration of each time step:
    time_step_days = 4
    
    # TODO: put back these plots?
    
    times = [time_step_days * t for t in range(len(num_susceptible_t))]
    trajectory = Trajectory(
        times,
        num_susceptible_t,
        num_infected_t,
        actions_taken_t,
        cost_t,
        total_reward
    )
    
    return trajectory
