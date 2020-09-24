from pdb import set_trace as b

from matplotlib import pyplot as plt


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
    num_infected = []
    actions_taken = []
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1

    print(f'initial num infected: {env.state}')
    b()
    
    for t in range(100):
        # Get best action
        # observation = min(observation, env.nS - 1) # max number infected
        action = policy[observation].argmax()
        actions_taken.append(env.actions_r[action])
        num_infected.append(observation)
        observation, reward, done, info = env.step(action)

        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma

    
    
    print('num infected')
    plt.plot(range(len(num_infected)), num_infected)
    plt.show()

    print('action taken')
    plt.plot(range(len(actions_taken)), actions_taken)
    plt.show()

    print(f'total reward: {total_reward}')
    
    env.close()
