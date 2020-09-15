from math import floor, ceil
import itertools

import numpy as np
from scipy.stats import poisson, nbinom
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm

from utils import save_pickle, load_pickle


class PandemicImmunityEnv(gym.Env):
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
               dynamics='SIR',
               time_lumping=False,
               init_transition_probs=False,
               **kwargs):
        super(PandemicImmunityEnv, self).__init__()
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

        self.nS = (num_population + 1) * (num_population + 1)
        # Use entire state space
        self.observation_space = spaces.Box(low=0,
                                            high=num_population,
                                            shape=(2,), dtype=np.uint16)  # maximum infected = 2**16 == 65536
        # self.observation_space = spaces.Discrete(self.nS)

        self.dynamics_param_str = f'distr_family={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},num_states={self.nS},num_actions={self.nA},dynamics={self.dynamics},time_lumping={self.time_lumping}'

        self.reward_param_str = f'power={self.power},scale_factor={self.scale_factor}'
        
        self.P = None
        if init_transition_probs:
            self._set_transition_probabilities()
        
        self.done = 0
        self.reward = 0

        self.state = self.reset()
        
    def step(self, action):
        # Execute one time step within the environment

        prev_num_infected = self.state[-1]
        prev_num_immune = sum(self.state[:-1])
        num_susceptible = self.num_population - prev_num_immune
        
        r = self.actions_r[action]

        distr = self._new_infected_distribution(self.state, r, imported_cases_per_step=self.imported_cases_per_step)
        new_num_infected = distr.rvs()
        new_num_infected = min(new_num_infected, num_susceptible)

        new_num_immune = prev_num_immune + prev_num_infected
        
        new_state = [new_num_immune, new_num_infected]

        reward = self._reward(self.state, self.actions_r[action])

        # Add new observation to state array
        self.state = new_state

        obs = self.state
        done = self.done

        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the environment to an initial state
        initial_num_immune = 0
        self.state = [initial_num_immune, self.initial_num_cases]
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

    def _expected_new_infected(self, num_infected, num_immune, r, **kwargs):    
        fraction_susceptible = 1 # (num_population - current_cases) / num_population
        # TODO: may need better way to bound susceptible population,
        # to account for immunity
        # One option: fraction_susceptible = 1 always, and just bound new_state by num_population

        # Better solution: keep track of how many people are susceptible NOW, based on some immunity time
        fraction_susceptible = 1.0 - (num_immune * 1.0 / self.num_population)
        expected_new_cases = (num_infected * r) * fraction_susceptible + self.imported_cases_per_step

        return expected_new_cases

    def _new_infected_distribution(self, state, r, **kwargs):
        # distr_family: 'poisson' or 'nbinom'

        prev_num_infected = state[-1]
        prev_num_immune = sum(state[:-1])
        
        lam = self._expected_new_infected(prev_num_infected, prev_num_immune, r, **kwargs)

        if self.distr_family == 'poisson':
            return poisson(lam)
        elif self.distr_family == 'nbinom':
            r = 100000000000000.0
            p = lam / (r + lam)
            return nbinom(r, 1-p)

    def _set_transition_probabilities(self, **kwargs):
        file_name = f'../lookup_tables/{self.dynamics_param_str}/transition_dynamics_{self.dynamics_param_str}.pickle'
        try:
            self.P = load_pickle(file_name)
            print('Loaded transition_probs')
            return self.P
        except:
            self.P = []

        states = itertools.product(range(self.observation_space.low[0],
                                         self.observation_space.high[0] + 1),
                                   range(self.observation_space.low[1],
                                         self.observation_space.high[1] + 1))
        states_list = list(states)
        state_to_idx = {states_list[idx]: idx for idx in range(len(states_list))}
        self.P = [ [[] for action in range(self.nA)] for state in range(self.nS)]

        for state_idx in tqdm(range(self.nS)):
            state = states_list[state_idx]
            prev_num_infected = state[-1]
            prev_num_immune = sum(state[:-1])

            #if prev_num_immune + prev_num_infected > self.num_population:
            #    continue

            num_susceptible = self.num_population - prev_num_immune
            
            new_num_immune = prev_num_immune + prev_num_infected
            
            for action in range(self.nA):
                distr = self._new_infected_distribution(state, self.actions_r[action], **kwargs)
                
                k = 3
                low = max(floor(distr.mean() - k * distr.std()), 0)
                high = min(ceil(distr.mean() + k * distr.std()), num_susceptible)
                feasible_range = range(low, high + 1)
                
                for new_num_infected in feasible_range: # range(self.nS):
                    new_state = (new_num_immune, new_num_infected)
                    
                    prob = 0
                    if new_num_infected == self.num_population - 1:
                        # probability of landing on new_state or anything above
                        prob = 1 - distr.cdf(new_num_infected - 1)
                    else:
                        prob = distr.pmf(new_num_infected)
                    done = False
                    reward = self._reward(new_num_infected, self.actions_r[action])

                    new_state_idx = state_to_idx[new_state]
                    outcome = (prob, new_state_idx, reward, done)
                    self.P[state_idx][action].append(outcome)

        save_pickle(self.P, file_name)
        return self.P
