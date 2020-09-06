import gym

import gym_pandemic


env = gym.make('pandemic-v0')
env.reset()

gamma = 0.99

print('R \t total reward')
for i_episode in range(env.num_actions):
  observation = env.reset()
  total_reward = 0
  gamma_cum = 1
  for t in range(100):
    # action = env.action_space.sample()
    action = i_episode
    observation, reward, done, info = env.step(action)
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
    total_reward += gamma_cum * reward
    gamma_cum *= gamma
  print(f'{env.actions_r[action]} \t {total_reward}')
    
env.close()
