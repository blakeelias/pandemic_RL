from pdb import set_trace as b

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from einops import rearrange


def test_environment(env, policy, V=None):
    # Plot policy
    actions_policy = [env.actions_r[policy[i].argmax()] for i in range(env.nS)]
    print('policy')
    plt.plot(range(env.nS), actions_policy)
    plt.show()

    # Plot value function
    print('value function')
    plt.plot(range(env.nS), V)
    plt.show()
    
    gamma = 0.99
    num_susceptible_t = []
    num_infected_t = []
    actions_taken_t = []
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1

    print(f'initial num infected: {env.state}')
    b()
    
    '''for t in range(100):
        # Get best action
        # observation = min(observation, env.nS - 1) # max number infected
        state_idx = env.state_to_idx[observation]
        action = policy[state_idx].argmax()
        actions_taken_t.append(env.actions_r[action])
        num_susceptible, num_infected = env._unpack_state(observation)
        num_infected_t.append(num_infected)
        num_susceptible_t.append(num_susceptible)
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma'''


    
    print('num susceptible')
    plt.plot(range(len(num_susceptible_t)), num_susceptible_t)
    plt.show()
    
    print('num infected')
    plt.plot(range(len(num_infected_t)), num_infected_t)
    plt.show()

    print('action taken')
    plt.plot(range(len(actions_taken_t)), actions_taken_t)
    plt.show()

    print(f'total reward: {total_reward}')

    plot_value_function(env, policy, V)
    
    env.close()


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
    # Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, Z_value, rstride=10, cstride=10)
    #plt.show()

    print('Policy')
    ax.plot_wireframe(X, Y, Z_policy, rstride=10, cstride=10)
    plt.show()

    
