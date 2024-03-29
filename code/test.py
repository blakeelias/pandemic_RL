from pdb import set_trace as b
import os
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import axes3d
from einops import rearrange
from tqdm import tqdm

from value_iteration import one_step_lookahead
from policies import policy_fn_generator, default_policy_fns, policy_fn_cases_generator


Trajectory = namedtuple('Trajectory', ['time_idxs', 'time_days', 'states_t', 'actions_t', 'num_susceptible_t', 'num_infected_t', 'contact_factor_t', 'cost_t', 'total_reward'])

def test_environment(env, policy, V=None, discount_factor=1.0, policy_switch_times=(0, 8, 16, 32, 64, 96, 128, 134, 160)):
    Path(env.file_name_prefix).mkdir(parents=True, exist_ok=True)
    
    # policy: (time, env.nS, env.nA)
    # V: (time, env.nS)
    
    time, env.nS, env.nA = policy.shape
    if V is None:
        V = -1 * np.ones((time, env.nS))
    
    ### Extract policy
    policy_idxs = policy.argmax(axis=-1)
    # (time, env.nS)
    policy_rs = np.array([[env.actions_r[policy_idxs[i, j]] * env.transmissibility_schedule[i] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])
    # (time, env.nS)
    policy_contact_rates = np.array([[env.contact_factor[policy_idxs[i, j]] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])
    # (time, env.nS)

    ### Save CSVs
    save_policy_csvs(env, policy_rs, policy_contact_rates)
    save_value_csv(env, V)

    save_transmissibility_csv(env)
    save_vaccinated_csv(env)
    # save_cost_per_case_csv(env)
    
    # vaccine schedule
    plot_transmissibility(env)
    # plot_vaccinated(env)
    # plot_cost_per_case_csv(env)
    
    new_policy_fn = policy_fn_generator(policy)
    original_policy_fn = policy_fn_cases_generator(60)

    ### Generate Trajectories
    for policy_switch_time in policy_switch_times:
        print(f'policy_switch_time: {policy_switch_time}')
        trajectory = trajectory_generator(env, new_policy_fn, 'optimized_policy', discount_factor, original_policy_fn, policy_switch_time)
        plot_policy_trajectory(env, policy_rs, trajectory, 'R', center=1.0)
        plot_policy_trajectory(env, policy_contact_rates, trajectory, 'contact_rate', center=1.0/env.R_0)
        
    env.close()


def test_environment_default_policy(env, discount_factor):
    trajectory = trajectory_generator(env, policy_fn, 'optimized_policy', discount_factor, original_policy_fn, policy_switch_time)

    save_trajectory_csv(env, trajectory)
    ### Plots
    # policy + trajectory
    plot_policy_trajectory(env, policy_rs, trajectory, 'R', center=1.0)
    plot_policy_trajectory(env, policy_contact_rates, trajectory, 'contact_rate', center=1.0/env.R_0)
    
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

    
def save_transmissibility_csv(env):
    df = pd.DataFrame(env.transmissibility_schedule)
    df.to_csv(env.file_name_prefix + 'transmissibility_schedule.txt')

    
def save_vaccinated_csv(env):
    df = pd.DataFrame(env.vaccinated)
    df.to_csv(env.file_name_prefix + 'vaccine_schedule.txt')

    
def save_cost_per_case_csv(env):
    raise Exception('Not yet implemented')
    df = pd.DataFrame(env.cost_per_case)
    df.to_csv(env.file_name_prefix + 'transmissibility_schedule.txt')

    
def plot_policy_trajectory(env, policy, trajectory, policy_type_str, center=1.0, extra_str=''):
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('lockdown', [(0.0, 'red'), (0.25, 'red'), (0.5, 'yellow'), (1, 'green')])
    
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

    left = 0.125
    bottom = 0.1
    
    width_0 = 0.85
    height_0 = 0.2

    width_1 = width_0
    height_1 = 0.55

    rect_0 = [left, bottom, width_0, height_0]
    rect_1 = [left, bottom + height_0 + 0.1, width_1, height_1]
    
    fig = plt.figure()

    ax_0 = plt.axes(rect_0)
    ax_1 = plt.axes(rect_1)
    
    axs = [ax_0, ax_1]

    T = len(trajectory.contact_factor_t)
    
    # color_map = sns.color_palette("vlag_r", as_cmap=True)
    # fig, axs = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [1, 3]})
    plt.tick_params(bottom='on')

    if policy is not None:
        # Plot full policy heatmap
        axs[1] = sns.heatmap(policy[:-1, :].T, center=center, cmap=color_map) # 'RdYlGn')
        axs[1].invert_yaxis()
                
    if trajectory:
        # Plot infection trajectory

        # plot = sns.lineplot(x=list(range(len(trajectory.num_infected_t))), y=trajectory.num_infected_t, linewidth=2, ax=ax_1, color='black')  # x=trajectory.times
        times = trajectory.time_days
        plot = sns.lineplot(x=times, y=trajectory.num_infected_t, linewidth=2, ax=ax_1, color='black')  # x=trajectory.times
        # ax_1.axis('tight')

        ax_1.set_xlim(0, times[-1])
        ax_1.set_ylim(0, env.max_infected * 1.1)
        
        plot.set_ylabel('Number of New Infections')
        plot.set_title('New Infections Over Time')
        
        # Plot actions taken
        for t in range(T):
            contact_factor = trajectory.contact_factor_t[t]
            color = color_map(contact_factor)
            ax_1.axvspan(times[t], times[t] + env.time_step_days, facecolor=color, alpha=0.5, zorder=-100)        
        
        axs[0] = plot_vaccinated(env, trajectory, ax=axs[0])

    directory = Path(env.file_name_prefix) / f'trajectories_(policy={policy_type_str})_(switch_time={trajectory.policy_switch_time})'
    file_path = str(Path(directory) / f'{extra_str}.png')
    Path(directory).mkdir(parents=True, exist_ok=True)
    print('Created directory')
    fig.savefig(file_path)
    plt.clf()
    fig.clf()

def plot_policy(best_action_idx):
    plt.clf()
    ax = sns.barplot(list(range(num_states)),
                [actions[int(best_action_idx[n])] for n in range(num_states)])
    # ax.set(xlabel='common xlabel', ylabel='common ylabel')
    ax.set_xticks(range(int(len(states) / 1)))
    ax.set_xticklabels([state for i, state in enumerate(states) if i % 1 == 0])
    plt.show()


def plot_transmissibility(env):
    plt.clf()
    transmissibility = env.transmissibility_schedule
    times = list(range(len(transmissibility)))
    sns.lineplot(x=times, y=transmissibility)
    plt.savefig(env.file_name_prefix + f'transmissibility.png')

    
def plot_vaccinated(env, trajectory, ax=None):
    vaccinated = env.vaccinated[:-1]  # one extra entry here...
    times = trajectory.time_days
        
    ax_new = sns.lineplot(x=times, y=vaccinated, ax=ax, color='black')
    ax_new.set_xlim(0, times[-1])
    ax_new.set_ylim(-0.05, 1.05)
    ax_new.set_xlabel('Time (Days)')
    ax_new.set_ylabel('Portion Vaccinated')

    return ax_new
        
    
def plot_value_function(env, policy, V):
    plt.clf()
    state_to_value = {}
    state_to_action = {}

    for idx in range(env.nS):
        idx = env.state_obj_to_idx(packed_state)
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

def cost_of_trajectory(trajectory, env, discount_factor):
    '''
    Re-evaluate the cost of a trajectory in a different environment than where the trajectory was generated.
    Assumes the new environment's "(state, action) -> new_state" dynamics are the same, and that just the reward function R(state, action) is different.
    '''
    
    total_reward = 0
    
    for state, action, time_idx in zip(trajectory.states_t, trajectory.actions_t, trajectory.time_idxs):
        
        reward = env._reward(state, action, time_idx)

        total_reward *= discount_factor
        total_reward += reward
        
    return total_reward

    
def trajectory_generator(env, policy_fn, policy_name, gamma):
    # file_name_prefix = env.file_name_prefix + f'/policy={policy_name}/'
    # Path(file_name_prefix).mkdir(parents=True, exist_ok=True)
    
    ### Trajectory
    # Step through a trajectory
    # TODO: put this back and actually plot a few trajectories
    # Must change to use time-varying policy

    # Time Series to Save
    states_t = []
    actions_t = []
    num_susceptible_t = []
    num_infected_t = []
    contact_factor_t = []
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
    horizon = env.horizon
    if horizon == np.inf:
        horizon = 100

    print('Starting trial:')
    for t in tqdm(list(range(horizon))):
        try:
            new_state = env.state_idx_to_obj(observation)
        except:
            b()
        if type(new_state) == tuple:
            num_susceptible, num_infected = new_state
            num_infected_t.append(num_infected)
            num_susceptible_t.append(num_susceptible)
        else:
            num_infected_t.append(new_state)

        states_t.append(observation)
        
        if t % env.action_frequency == 0:
            # Get best action
            # Allowed to take a new action once every {env.action_frequency} steps
            # observation = min(observation, env.nS - 1) # max number infected
            #b()
            state_idx = observation
            action = policy_fn(env, state_idx, t)

        actions_t.append(action)
        contact_factor_t.append(env.contact_factor[action])
        observation, reward, done, info = env.step(action)
        cost_t.append(reward)
        R_t.append(env.R_t(action, t, num_susceptible))
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma


    # Duration of each time step:
    time_step_days = 4
    
    # TODO: put back these plots?
    
    time_days = [env.time_step_days * t for t in range(len(num_susceptible_t))]
    time_idxs = list(range(len(time_days)))
    
    trajectory = Trajectory(
        time_idxs,
        time_days,
        states_t,
        actions_t,
        num_susceptible_t,
        num_infected_t,
        contact_factor_t,
        cost_t,
        total_reward
    )
    
    return trajectory
