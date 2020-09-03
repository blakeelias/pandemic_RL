import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PandemicEnvironment(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}
  NUM_POPULATION = 1000
  N_DISCRETE_ACTIONS = 14

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0,
                                        high=NUM_POPULATION,

                                        shape=(1,), dtype=np.uint16)  # maximum infected = 2**16 == 65536

  def step(self, action):
    # Execute one time step within the environment
    ...
    
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...
