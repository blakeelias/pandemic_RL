from math import floor, ceil

import numpy as np
from scipy.stats import poisson, nbinom
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm

from utils import save_pickle, load_pickle


class PandemicEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
               num_population=1000,
               initial_num_cases=100,
               R_0=2.5,
               imported_cases_per_step=0.5,
               power=2,
               scale_factor=100,
               distr_family='nbinom',
               dynamics='SIS',
               time_lumping=False):
        super(PandemicEnv, self).__init__()
        self.num_population = num_population
        self.initial_num_cases = initial_num_cases
        self.R_0 = R_0
        self.imported_cases_per_step = imported_cases_per_step
        self.power = power
        self.scale_factor = scale_factor
        self.distr_family = distr_family
        self.dynamics = dynamics
        self.time_lumping = time_lumping
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions
        self.actions_r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5])
        self.nA = self.actions_r.shape[0]
        self.action_space = spaces.Discrete(self.nA)

        self.nS = num_population
        # Use entire state space
        self.observation_space = spaces.Box(low=0,
                                            high=num_population,
                                            shape=(1,), dtype=np.uint16)  # maximum infected = 2**16 == 65536
        # self.observation_space = spaces.Discrete(self.nS)

        self.P = self._transition_probabilities()
        
        self.state = [self.initial_num_cases]
        self.done = 0
        self.reward = 0
    
    def step(self, action):
        # Execute one time step within the environment
        prev_cases = self.state[-1]
        r = self.actions_r[action]

        distr = self._new_state_distribution(prev_cases, r, imported_cases_per_step=self.imported_cases_per_step)
        new_cases = distr.rvs()
        new_cases = min(new_cases, self.num_population)
        new_state = new_cases

        reward = self._reward(new_state, self.actions_r[action])

        # Add new observation to state array
        self.state.append(new_state)
        # Remove oldest observation from state array
        self.state.pop(0)

        obs = self.state
        done = self.done

        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = [self.initial_num_cases]
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

    def _expected_new_state(self, num_infected, r, **kwargs):    
        fraction_susceptible = 1 # (num_population - current_cases) / num_population
        # TODO: may need better way to bound susceptible population,
        # to account for immunity
        # One option: fraction_susceptible = 1 always, and just bound new_state by num_population

        # Better solution: keep track of how many people are susceptible NOW, based on some immunity time
        expected_new_cases = (num_infected * r) * fraction_susceptible + self.imported_cases_per_step

        return expected_new_cases

    def _new_state_distribution(self, num_infected, r, **kwargs):
        # distr_family: 'poisson' or 'nbinom'
        lam = self._expected_new_state(num_infected, r, **kwargs)

        if self.distr_family == 'poisson':
            return poisson(lam)
        elif self.distr_family == 'nbinom':
            r = 100000000000000.0
            p = lam / (r + lam)
            return nbinom(r, 1-p)

    def _transition_probabilities(self, **kwargs):
        file_name = f'lookup_tables/transitions/transition_probs_distr={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},num_states={self.nS},num_actions={self.nA},dynamics={self.dynamics},time_lumping={self.time_lumping}.pickle'
        try:
            P = load_pickle(file_name)
            print('Loaded transition_probs')
            return P
        except:
            P = []
        
        P = [[ [] for j in range(self.nA)] for i in range(self.nS)]

        for state in tqdm(range(self.nS)):
            for action in range(self.nA):
                distr = self._new_state_distribution(state, self.actions_r[action], **kwargs)
                
                k = 3
                low = distr.mean() - k * distr.std()
                high = distr.mean() + k * distr.std()
                feasible_range = range(floor(low), ceil(high))
                
                for new_state in feasible_range: # range(self.nS):
                    prob = distr.pmf(new_state)
                    done = False
                    reward = self._reward(new_state, self.actions_r[action])

                    outcome = (prob, new_state, reward, done)
                    P[state][action].append(outcome)

        save_pickle(P, file_name)
        return P
