import numpy as np
from scipy.stats import poisson, nbinom
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PandemicEnvironment(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  rs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5])

  def __init__(self, num_population=1000, initial_num_cases=100):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions
    num_discrete_actions = rs.shape[0]
    self.action_space = spaces.Discrete(num_discrete_actions)
    
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0,
                                        high=num_population,
                                        shape=(1,), dtype=np.uint16)  # maximum infected = 2**16 == 65536

    self.state = [initial_num_cases]
    self.done = 0
    self.reward = 0

    # assumes num_population = 1000
    self.states_num_infected = sum([list(range(100*i, 100*(i+1), i+1)) for i in range(10)], []) + [1_000]

    
    
  def step(self, action):
    # Execute one time step within the environment
    prev_cases = self.state[-1]
    r = rs[action]
    expected_new_cases = r * prev_cases
    new_cases = new_state_distribution(expected_new_cases)
    
    
  def reset(self):
    # Reset the state of the environment to an initial state
      self.state = [INITIAL_NUM_CASES]
      
      return self._next_observation()
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(self.state)




def new_state_distribution(state, action, distr_family='poisson', **kwargs):
    # distr_family: 'poisson' or 'nbinom'
    current_num_infected, current_fraction_susceptible = state
    expected_new_cases, expected_num_recovered = expected_change_in_state(state, action, **kwargs)
    
    if distr_family == 'poisson':
        num_infected = poisson(current_num_infected + expected_new_cases - expected_num_recovered)
        current_fraction_susceptible = (current_fraction_susceptible * num_population + expected_num_recovered) / num_population
    elif distr_family == 'nbinom':
        r = 100000000000000.0
        p = lam / (r + lam)
        return nbinom(r, 1-p)
    
# mean = pr/(1-p)
# = (r(lam)/(r+lam))/(r/(r+lam))
# = r*lam/r
# = lam
    
