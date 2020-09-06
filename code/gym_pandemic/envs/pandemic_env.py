import numpy as np
from scipy.stats import poisson, nbinom
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class PandemicEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self,
               num_population=1000,
               initial_num_cases=100,
               R_0=2.5,
               imported_cases_per_step=0.5,
               power=2,
               scale_factor=100):
    super(PandemicEnv, self).__init__()
    self.num_population = num_population
    self.initial_num_cases = initial_num_cases
    self.R_0 = R_0
    self.imported_cases_per_step = imported_cases_per_step
    self.power = power
    self.scale_factor = scale_factor

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions
    self.actions_r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5])
    self.num_actions = self.actions_r.shape[0]
    self.action_space = spaces.Discrete(self.num_actions)
    
    # assumes num_population = 1000
    self.states_num_infected = sum([list(range(100*i, 100*(i+1), i+1)) for i in range(10)], []) + [1_000]
    self.num_states = len(self.states_num_infected)
    self.lookup = dict([(x, i) for i in range(self.num_states - 1) for x in range(self.states_num_infected[i], self.states_num_infected[i+1])] + [(num_population, self.num_states - 1)])

    # Use entire state space
    #self.observation_space = spaces.Box(low=0,
    #                                    high=num_population,
    #                                    shape=(1,), dtype=np.uint16)  # maximum infected = 2**16 == 65536
    self.observation_space = spaces.Discrete(self.num_states)

    self.state = [self._bucket_state_index(self.initial_num_cases)]
    self.done = 0
    self.reward = 0
    
  def step(self, action):
    # Execute one time step within the environment
    prev_cases = self.state[-1]
    r = self.actions_r[action]

    distr = self._new_state_distribution(prev_cases, r, imported_cases_per_step=self.imported_cases_per_step)
    new_cases = distr.rvs()
    new_cases = min(new_cases, self.num_population)
    new_state = self._bucket_state_index(new_cases)

    reward = self._reward(self.states_num_infected[new_state], self.actions_r[action])

    # Add new observation to state array
    self.state.append(new_state)
    # Remove oldest observation from state array
    self.state.pop(0)

    obs = self.state
    done = self.done
    
    return obs, reward, done, {}
    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.state = [self._bucket_state_index(self.initial_num_cases)]
    obs = self.state

    return obs
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(self.state)

  def _reward(self, num_infected, r, **kwargs):
    return -self._cost_of_n(num_infected, **kwargs) - self._cost_of_r(r, **kwargs)
    
  def _cost_of_r(self, r, **kwargs):
    baseline = 1/(self.R_0 ** self.power)
    actual = 1/(r ** self.power)
    
    # cost_to_keep_half_home / (1/((num_population/4)**power) - 1/(R_0 ** power))
    if r >= self.R_0:
        return 0
    else:
        return (actual - baseline) * self.scale_factor  # (actual - baseline)
    #return actual

  def _cost_of_n(self, n, **kwargs):
    if n <= 0:
      return 0
    else:
      return n

  def _expected_new_state(self, num_cases, r, **kwargs):    
    fraction_susceptible = 1 # (num_population - current_cases) / num_population
    # TODO: may need better way to bound susceptible population,
    # to account for immunity
    # One option: fraction_susceptible = 1 always, and just bound new_state by num_population

    # Better solution: keep track of how many people are susceptible NOW, based on some immunity time
    expected_new_cases = (num_cases * r) * fraction_susceptible + self.imported_cases_per_step

    return expected_new_cases
  
  def _bucket_state_index(self, new_num_cases):
    if new_num_cases < 0:
        raise Exception('Cannot have negative case count')
    elif new_num_cases in self.lookup:
        return self.lookup[new_num_cases]
    else:
        return self.lookup[self.num_population]

  def _new_state_distribution(self, state, action, distr_family='poisson', **kwargs):
    # distr_family: 'poisson' or 'nbinom'
    lam = self._expected_new_state(state, action, **kwargs)
    
    if distr_family == 'poisson':
        return poisson(lam)
    elif distr_family == 'nbinom':
        r = 100000000000000.0
        p = lam / (r + lam)
        return nbinom(r, 1-p)

  def _bucketed_distribution(self, distribution):
    num_states = len(self.states_num_infected)
    probs = np.zeros_like(self.states_num_infected, dtype=float)
    for state in range(num_states - 1):
      low_state = self.states_num_infected[state]
      high_state = self.states_num_infected[state + 1]
      probs[state] = (distribution.cdf(high_state) - distribution.cdf(low_state))
    probs[num_states - 1] = 1 - distribution.cdf(states[num_states - 1])
    return probs                                                  


def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]
