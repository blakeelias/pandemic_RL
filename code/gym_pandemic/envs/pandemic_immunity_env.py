from math import floor, ceil, sqrt
import itertools

import numpy as np
from scipy.stats import poisson, nbinom, rv_discrete
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
               max_infected_desired=100,
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
        self.max_infected_desired = max_infected_desired
        self.num_stdevs = 3
        self.susceptible_increment = int(self.num_population * 0.01)
        
        # Action: Setting contact rate, R
        self.actions_r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5])
        self.nA = self.actions_r.shape[0]
        self.action_space = spaces.Discrete(self.nA)

        
        # State: (num_susceptible, num_infected)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([
                                                int(self.num_population / self.susceptible_increment),
                                                self.max_infected_desired
                                            ]),
                                            shape=(2,), dtype=np.uint16)  # maximum infected = 2**16 == 65536
        # Observation: (num_susceptible, num_infected)
        self.nS = (self.observation_space.high[0] + 1) * (self.observation_space.high[1] + 1)
        
        self.dynamics_param_str = f'distr_family={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},num_states={self.nS},num_actions={self.nA},dynamics={self.dynamics},time_lumping={self.time_lumping}'

        self.reward_param_str = f'power={self.power},scale_factor={self.scale_factor}'
        
        self.P = None
        if init_transition_probs:
            self._set_transition_probabilities()
        
        self.done = 0
        self.reward = 0

        self.state = self.reset()
        
    def step(self, action):
        '''Execute one time step within the environment'''
        prev_num_susceptible, prev_num_infected = self._unpack_state(self.state)
        r = self.actions_r[action]

        # calculate new number of infected
        distr = self._new_infected_distribution(self.state, r, imported_cases_per_step=self.imported_cases_per_step)
        new_num_infected = distr.rvs()
        new_num_infected = min(new_num_infected, num_susceptible)

        # calculate new number susceptible
        expected_new_num_susceptible = prev_num_susceptible - new_num_infected
        new_num_susceptible_distr = self._new_num_susceptible_distr(new_num_susceptible)
        new_num_susceptible = new_num_susceptible_distr.rvs()
        
        # new state
        new_unpacked_state = (new_num_susceptible, new_num_infected)

        # reward
        reward = self._reward(new_num_infected, self.actions_r[action])

        
        # Add new observation to state array
        self.state = self._pack_state(new_state)
        obs = self.state
        done = self.done

        return obs, reward, done, {}

    
    def _unpack_state(self, state):
        prev_num_susceptible = state[0] * self.susceptible_increment
        prev_num_infected = state[1]
        return (prev_num_susceptible, prev_num_infected)

    def _pack_state(self, state):
        prev_num_susceptible = state[0] // self.susceptible_increment
        prev_num_infected = self.state[1]
        return (prev_num_susceptible, prev_num_infected)

        
    def _new_num_susceptible_distr(self, new_num_susceptible):
        num_increments_low = new_num_susceptible // self.susceptible_increment
        remainder = new_num_susceptible % self.susceptible_increment

        probs = (
            (num_increments_low * self.susceptible_increment, (num_increments_low + 1) * self.susceptible_increment),
            (1 - (remainder / self.susceptible_increment), remainder / self.susceptible_increment),
        )

        return rv_discrete(values=probs)
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        initial_num_immune = 0
        self.state = (initial_num_immune, self.initial_num_cases)
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

    def _expected_new_infected(self, num_infected, num_susceptible, r, **kwargs):
        fraction_susceptible = 1 # (num_population - current_cases) / num_population
        # TODO: may need better way to bound susceptible population,
        # to account for immunity
        # One option: fraction_susceptible = 1 always, and just bound new_state by num_population

        # Better solution: keep track of how many people are susceptible NOW, based on some immunity time
        fraction_susceptible = num_susceptible / self.num_population
        expected_new_cases = (num_infected * r) * fraction_susceptible + self.imported_cases_per_step

        return expected_new_cases
    
    def _new_infected_distribution(self, unpacked_state, r, **kwargs):
        # distr_family: 'poisson' or 'nbinom'

        prev_num_susceptible = unpacked_state[0]
        prev_num_infected = unpacked_state[1]
        
        lam = self._expected_new_infected(prev_num_infected, prev_num_susceptible, r, **kwargs)

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

        assert len(states_list) == self.nS
        
        state_to_idx = {states_list[idx]: idx for idx in range(len(states_list))}
        self.P = [ [[] for action in self._allowed_rs(self._unpack_state(packed_state))] for packed_state in states_list]

        for state_idx in tqdm(range(self.nS)):
            packed_state = states_list[state_idx]
            unpacked_state = self._unpack_state(packed_state)
            prev_num_susceptible = unpacked_state[0]
            prev_num_infected = unpacked_state[1]
            
            if prev_num_susceptible + prev_num_infected > self.num_population:
                continue

            for action_idx, action_r in enumerate(self._allowed_rs(unpacked_state)):
                distr = self._new_infected_distribution(unpacked_state, action_r, **kwargs)
                k = self.num_stdevs
                low = max(floor(distr.mean() - k * distr.std()), 0)
                high = min(ceil(distr.mean() + k * distr.std()), prev_num_susceptible)
                feasible_range = range(low, high + 1)
                
                for new_num_infected in feasible_range: # range(self.nS):
                    
                    expected_new_num_susceptible = prev_num_susceptible - new_num_infected
                    new_num_susceptible_distr = self._new_num_susceptible_distr(expected_new_num_susceptible)
                    
                    for new_num_susceptible in new_num_susceptible_distr.support():
                        prob_new_num_susceptible = new_num_susceptible_distr.pmf(new_num_susceptible)
                        if prob_new_num_susceptible == 0:
                            continue
                        
                        new_unpacked_state = (new_num_susceptible, new_num_infected)
                        new_packed_state = self._pack_state(new_unpacked_state)
                        
                        prob = 0
                        if new_num_infected == prev_num_susceptible:
                            # probability of landing on new_state or anything above
                            prob = 1 - distr.cdf(new_num_infected - 1)
                        else:
                            prob = distr.pmf(new_num_infected)
                        done = False
                        reward = self._reward(new_num_infected, action_r)

                        new_state_idx = state_to_idx[new_packed_state]
                        outcome = (prob * prob_new_num_susceptible, new_state_idx, reward, done)
                        self.P[state_idx][action_idx].append(outcome)

        save_pickle(self.P, file_name)
        return self.P

    
    def _max_allowed_r(self, state):
        prev_num_infected = state[-1]
        new_num_immune = sum(state)
        
        # E[new_num_infected] = (num_population - new_num_immune) / num_population * prev_num_infected * r
        # max_new_num_infected = E[new_num_infected] + k * sqrt(E[new_num_infected])
        # max_infected_desired = 100
        # E[new_num_infected] + k * sqrt(E[new_num_infected]) < max_infected_desired
        # x + k * sqrt(x) < max_infected_desired (== M)
        # sqrt(x) * (sqrt(x) + k) < M
        # a * (a + k) < M
        # a**2 + k*a < M
        # a**2 + k*a - M < 0
        # a = (-k +/- sqrt(k^2 + 4*1*M))/(2)
        # x = a ** 2

        k = self.num_stdevs
        a = (-k + sqrt(k * k + 4 * self.max_infected_desired))/2
        target_expected_new_num_infected = a * a

        if prev_num_infected > 0 and (self.num_population - new_num_immune > 0):
            r = target_expected_new_num_infected * self.num_population / ((self.num_population - new_num_immune) * prev_num_infected)
        else:
            r = np.inf
        
        return r

    def _allowed_rs(self, state):
        max_r = self._max_allowed_r(state)
        allowed = [r for r in self.actions_r if r <= max_r]
        # Note: the indices of `allowed` can also be used to index into self.actions_r
        return allowed
