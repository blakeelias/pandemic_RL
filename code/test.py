from pdb import set_trace as b

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from einops import rearrange

from value_iteration import one_step_lookahead


def test_environment(env, policy, V, file_name_prefix):
    # policy: (time, env.nS, env.nA)
    # V: (time, env.nS)
    
    ### Save full policy
    # (time, env.nS)
    policy_idxs = policy.argmax(axis=-1)

    # (time, env.nS)
    policy_rs = np.array([[env.actions_r[policy_idxs[i, j]] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])
    policy_contact_rates = np.array([[env.contact_factor[policy_idxs[i, j]] for j in range(policy_idxs.shape[1])] for i in range(policy_idxs.shape[0])])

    # Policy with respect to R ideally achieved at the given contact rate (with no influence of vaccination or other factors)
    policy_dict = {f't={t}': policy_rs[t, :] for t in range(policy_rs.shape[0])}
    policy_dict['state'] = range(env.nS)
    df_policy = pd.DataFrame(policy_dict)
    df_policy.to_csv(file_name_prefix + 'policy_R.txt')

    # Policy with respect to contact rate achieved
    policy_dict = {f't={t}': policy_contact_rates[t, :] for t in range(policy_contact_rates.shape[0])}
    policy_dict['state'] = range(env.nS)
    df_policy = pd.DataFrame(policy_dict)
    df_policy.to_csv(file_name_prefix + 'policy_contact_rate.txt')

    ### Plot full policy
    ax = sns.heatmap(policy_rs[:-1, :].T, linewidths=0.5, center=1.0, cmap='RdYlGn')
    # To show policy values: use `annot=True`
    # Round to integer: `fmt='d'` (gives error for floats)
    # To hide x axis ticks: `xticklabels=False`
    # TODO: label x and y axes
    # TODO: better color scheme
    ax.invert_yaxis()
    ax.get_figure().savefig(file_name_prefix + 'policy.png')
    
    ### Save full value function
    # V: (time, env.nS)
    value_dict = {f't={t}': V[t, :] for t in range(V.shape[0])}
    value_dict['state'] = range(env.nS)
    df_value = pd.DataFrame(value_dict)
    df_value.to_csv(file_name_prefix + 'value.txt')


    ### Plot start policy
    policy_start = policy[0, :, :] # (env.nS, env.nA)
    actions_policy = [env.actions_r[policy_start[i].argmax()] for i in range(env.nS)]
    df_policy_start = pd.DataFrame({'state': range(env.nS), 'action': actions_policy})
    df_policy_start.to_csv(file_name_prefix + 'policy_start.txt')
    print('policy')
    fig = plt.figure()
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Action (R)')
    ax1.set_xlabel('Number of Individuals Infectious ($I_t$)')
    ax1.set_title('Policy')
    ax1.bar(range(env.nS), actions_policy)
    plt.savefig(file_name_prefix + 'policy_start.png')

    
    ### Plot start value function
    V_start = V[0, :] # (env.nS,)
    df_value_start = pd.DataFrame({'state': range(env.nS), 'value': V_start})
    df_value_start.to_csv(file_name_prefix + 'value_start.txt')
    print('value function')
    fig = plt.figure()
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Value (Arbitrary Units)')
    ax1.set_xlabel('Number of Individuals Infectious ($I_t$)')
    ax1.set_title('Value Function')
    ax1.bar(range(env.nS), V_start)
    plt.savefig(file_name_prefix + 'value_start.png')


    ### Trajectory
    # Step through a trajectory
    # TODO: put this back and actually plot a few trajectories
    # Must change to use time-varying policy
    
    '''
    gamma = 0.99
    num_susceptible_t = []
    num_infected_t = []
    actions_taken_t = []
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1

    print(f'initial num infected: {env.state}')

    # plot_value_function(env, policy, V)

    print('Best action?')
    env._set_transition_probabilities()
    state_idx = 0
    packed_state = env.states[state_idx]
    unpacked_state = env._unpack_state(packed_state)
    print(f'State: {unpacked_state}')
    expected_action_values = one_step_lookahead(env, state_idx, V)
    print(f'expected_action_values: {expected_action_values}')
    print(f'best action: {expected_action_values.argmax()}')

    t = 0
    while t < min(env.horizon, 100):
        new_state = env._unpack_state(observation)
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
            state_idx = env.state_to_idx[observation]
            action = policy[state_idx].argmax()
            
        actions_taken_t.append(env.actions_r[action])
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma

        t += 1
    '''

    # Duration of each time step:
    time_step_days = 4

    '''
    # print('num susceptible')
    times = [time_step_days * t for t in range(len(num_susceptible_t))]
    df_susceptible = pd.DataFrame({'time': times, 'num_susceptible': num_susceptible_t})
    df_susceptible.to_csv(file_name_prefix + 'susceptible.txt')
    fig = plt.figure()
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Number Susceptible')
    ax1.set_xlabel('Time (days)')
    ax1.set_title('Number of Individuals Susceptible ($S_t$)')
    ax1.bar(times, num_susceptible_t)
    fig.savefig(file_name_prefix + 'susceptible.png')
    
    # print('num infected')
    times = [time_step_days * t for t in range(len(num_infected_t))]
    df_infected = pd.DataFrame({'time': times, 'num_infected': num_infected_t})
    df_infected.to_csv(file_name_prefix + 'infected.txt')
    fig = plt.figure()
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Number Infected')
    ax1.set_xlabel('Time (days)')
    ax1.set_title('Number of Individuals Infected ($I_t$)')
    ax1.bar(times, num_infected_t)
    fig.savefig(file_name_prefix + 'infected.png')
    # Try with Seaborn
    ax2 = sns.barplot(times, num_infected_t)
    ax2.set_xticks(times)
    ax2.set_xticklabels(times)
    fig2 = ax2.get_figure()
    fig2.savefig(file_name_prefix + 'infected_sns.png')
    
    # print('action taken')
    times = [time_step_days * t for t in range(len(actions_taken_t))]
    df_actions = pd.DataFrame({'time': times, 'action_taken': actions_taken_t})
    df_actions.to_csv(file_name_prefix + 'actions.txt')
    fig = plt.figure()
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Level of Lockdown, $R_t$')
    ax1.set_xlabel('Time (days)')
    ax1.set_title('Intervention Taken ($R_t$)')
    ax1.bar(times, actions_taken_t)
    fig.savefig(file_name_prefix + 'actions.png')
    
    print(f'total reward: {total_reward}')
    '''
    
    env.close()


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

    
