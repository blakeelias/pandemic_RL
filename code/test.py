from matplotlib import pyplot as plt


def test_environment(env, policy):
    env.reset()
    gamma = 0.99

    num_infected = []
    actions_r = []
    
    print('R \t total reward')
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1
    for t in range(100):
        # Get best action
        action = policy[observation].argmax()
        actions_r.append(env.actions_r[action])
        observation, reward, done, info = env.step(action)
        num_infected.append(observation)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma
    print(f'{env.actions_r[action]} \t {total_reward}')
    plt.plot(range(len(num_infected)), num_infected)
    plt.show()
    plt.plot(range(len(actions_r)), actions_r)
    plt.show()
    env.close()
