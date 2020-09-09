import gym

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


def train_environment(env):
    policy, V = value_iteration(env)
    save_pickle(policy)
    
def test_environment(env, policy=None):
    env.reset()
    gamma = 0.99

    print('R \t total reward')
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1
    for t in range(100):
        # action = env.action_space.sample()
        if not policy:
            action = i_episode
        else:
            # get best action
            action = policy[observation].argmax()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total_reward += gamma_cum * reward
        gamma_cum *= gamma
    print(f'{env.actions_r[action]} \t {total_reward}')
    env.close()


if __name__ == '__main__':
    env = gym.make('pandemic-v0')
    policy, V = value_iteration(env)
    test_environment(env)
