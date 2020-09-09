from matplotlib import pyplot as plt


def test_environment(env, policy, V=None):
    env.reset()
    gamma = 0.99

    num_infected = []
    actions_taken = []

    actions_policy = [env.actions_r[policy[i].argmax()] for i in range(env.nS)]

    print('policy')
    plt.plot(range(env.nS), actions_policy)
    plt.show()

    print('value function')
    plt.plot(range(env.nS), V)
    plt.show()
    
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1

    print(f'initial num infected: {env.state[0]}')
    
    for t in range(100):
        # Get best action
        action = policy[observation].argmax()
        actions_taken.append(env.actions_r[action])
        observation, reward, done, info = env.step(action)
        num_infected.append(observation)
        
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
