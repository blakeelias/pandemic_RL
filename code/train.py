import gym

import gym_pandemic
from value_iteration import value_iteration
from utils import save_pickle, load_pickle


def train_environment(env, theta=0.0001, discount_factor=0.99):
    reward_param_str = env.reward_param_str + f',theta={theta},discount_factor={discount_factor}'
    file_name = f'../lookup_tables/{env.dynamics_param_str}/policy_reward=({reward_param_str}),env=({env.dynamics_param_str}).pickle'

    try:
        policy, V = load_pickle(file_name)
        print('Loaded policy')
        return policy, V
    except:
        policy, V = value_iteration(env, theta=theta, discount_factor=discount_factor)
        save_pickle((policy, V))
        return policy, V


def test_environment(env, policy):
    env.reset()
    gamma = 0.99

    print('R \t total reward')
    observation = env.reset()
    total_reward = 0
    gamma_cum = 1
    for t in range(100):
        # Get best action
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
