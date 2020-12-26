from math import floor, ceil, prod
from pdb import set_trace as b
import itertools
import copy

import numpy as np
from scipy.stats import poisson, nbinom, rv_discrete
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm

from utils import save_pickle, load_pickle
from scenarios import US, Test

class PandemicEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    days_per_step = 4
    
    def __init__(self,
               num_population=1000,
               initial_fraction_infected=0.1,
               R_0=2.5,
               imported_cases_per_step=0.5,
               power=2,
               scale_factor=100,
               distr_family='nbinom',
               dynamics='SIS',
               init_transition_probs=False,
               horizon=np.inf,
               action_frequency=1,
               scenario=Test,
               **kwargs):
        super(PandemicEnv, self).__init__()
        self.num_population = num_population
        self.initial_num_infected = int(initial_fraction_infected * num_population)
        self.R_0 = R_0
        self.imported_cases_per_step = imported_cases_per_step
        self.power = power
        self.scale_factor = scale_factor
        self.distr_family = distr_family
        self.dynamics = dynamics
        self.horizon = horizon
        self.action_frequency = action_frequency
        self.horizon_effective = ceil(horizon / action_frequency) if horizon < np.inf else horizon
        self.scenario = scenario
        self.kwargs = kwargs
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions
        self.actions_r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5])
        self.nA = self.actions_r.shape[0]
        self.action_space = spaces.Discrete(self.nA)

        # Use entire state space
        self.observation_space = spaces.Box(low=0,
                                            high=num_population,
                                            shape=(1,), dtype=np.uint16)  # maximum infected = 2**16 == 65536
        # self.observation_space = spaces.Discrete(self.nS)
        self.states = list(range(self.observation_space.low[0],
                                 self.observation_space.high[0] + 1))
        self.state_to_idx = {self.states[idx]: idx for idx in range(len(self.states))}
        self.nS = len(self.states)

        self.dynamics_param_str = self._param_string(self.action_frequency, **self.kwargs)

        self.reward_param_str = f'power={self.power},scale_factor={self.scale_factor},horizon={self.horizon}'
        
        self.P = None
        if init_transition_probs:
            self._set_transition_probabilities()
            
        self.state = self.initial_num_infected
        self.done = 0
        self.reward = 0

        self.reset()
        
    def step(self, action):
        # Execute one time step within the environment
        prev_cases = self.state
        r = self.actions_r[action]
        distr = self._new_state_distribution(prev_cases, r, imported_cases_per_step=self.imported_cases_per_step)
        new_cases = distr.rvs()
        new_cases = min(new_cases, self.num_population)
        new_state = new_cases

        reward = self._reward(new_state, self.actions_r[action])

        # Add new observation to state array
        self.state = new_state

        obs = self.state
        done = self.done

        return obs, reward, done, {}

    def step_macro(self, action):
        for i in range(self.action_frequency):
            result = self.step(action)
        # TODO: accumulate results of each individual step into result object
        # i.e. list of observations, sum of rewards, any done, list of info?
        return result
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self.initial_num_infected
        obs = self.state

        return obs
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.state)

    def _unpack_state(self, packed_state):
        num_infected = packed_state
        num_susceptible = self.num_population
        return (num_susceptible, num_infected)
        
    def _reward(self, num_infected, r, **kwargs):
        return -self._cost_of_n(num_infected, **kwargs) \
               -self._cost_of_r_linear(r, self.R_0, self.R_0, **kwargs)
    
    def _cost_of_r(self, r, R_0, **kwargs):
        baseline = 1/(R_0 ** self.power)
        actual = 1/(r ** self.power)

        # cost_to_keep_half_home / (1/((num_population/4)**power) - 1/(R_0 ** power))
        if r >= self.R_0:
            return 0
        else:
            return (actual - baseline) * self.scale_factor  # (actual - baseline)
        #return actual

    def _cost_of_r_linear(self, r, R_0_new, R_0_orig, **kwargs):
        '''
>>> from gym_pandemic.envs.pandemic_env import PandemicEnv
>>> env = PandemicEnv()
>>> env
<gym_pandemic.envs.pandemic_env.PandemicEnv object at 0x7fd5157e1130>
>>> env._cost_of_r_linear(1.0, 4.0, 4.0, 1e6)
750000.0
>>> env._cost_of_r_linear(2.0, 4.0, 4.0, 1e6)
500000.0
>>> env._cost_of_r_linear(4.0, 4.0, 4.0, 1e6)
0.0
>>> env._cost_of_r_linear(4.0, 3.0, 4.0, 1e6)
-250000.0
>>> env._cost_of_r_linear(3.0, 3.0, 4.0, 1e6)
0.0
>>> env._cost_of_r_linear(2.0, 3.0, 4.0, 1e6)
250000.0
>>> env._cost_of_r_linear(1.0, 3.0, 4.0, 1e6)
500000.0
>>> env._cost_of_r_linear(0.0, 3.0, 4.0, 1e6)
750000.0
        '''
        cost_of_full_lockdown = self.days_per_step * self.scenario.gdp_per_day * self.scenario.fraction_gdp_lost
        r = max(r, 0) # cannot physically make r < 0
        fraction_locked_down = (R_0_orig - r) / R_0_orig
        fraction_for_free = (R_0_orig - R_0_new) / R_0_orig
        net_cost = cost_of_full_lockdown * (fraction_locked_down - fraction_for_free)
        return max(net_cost, 0) # cannot incur negative cost (i.e. make money) by making r > R_0_new

    def _cost_of_n(self, n, **kwargs):
        if n <= 0:
            return 0
        else:
            return n * self.scenario.cost_per_case

    def _expected_new_state(self, num_infected, r, **kwargs):
        fraction_susceptible = 1 # (num_population - current_cases) / num_population
        # TODO: may need better way to bound susceptible population,
        # to account for immunity
        # One option: fraction_susceptible = 1 always, and just bound new_state by num_population

        # Better solution: keep track of how many people are susceptible NOW, based on some immunity time
        expected_new_cases = (num_infected * r) * fraction_susceptible + self.imported_cases_per_step

        return expected_new_cases

    def _new_state_distribution(self, num_infected, r, **kwargs):
        # distr_family: 'poisson' or 'nbinom' or 'deterministic'
        lam = self._expected_new_state(num_infected, r, **kwargs)

        if self.distr_family == 'poisson':
            return poisson(lam)
        elif self.distr_family == 'nbinom':
            r = 100000000000000.0
            # r = 0.17
            p = lam / (r + lam)
            return nbinom(r, 1 - p)
        elif self.distr_family == 'deterministic':
            return rv_discrete(values=([lam], [1.0]))
        
    def _set_transition_probabilities_1_step(self, **kwargs):
        file_name = self._dynamics_file_name(iterations=1)
        file_name_lookup = self._dynamics_file_name(iterations=1, lookup=True)
        try:
            self.P_1_step = load_pickle(file_name)
            self.P_lookup_1_step = load_pickle(file_name_lookup)
            print('Loaded transition_probs')
            return self.P_1_step
        except:
            self.P_1_step = []
        
        self.P_1_step = [[ [] for j in range(self.nA)] for i in range(self.nS)]
        self.P_lookup_1_step = [[[None for k in range(self.nS)] for j in range(self.nA)] for i in range(self.nS)]

        for state in tqdm(range(self.nS)):
            for action in range(self.nA):
                distr = self._new_state_distribution(state, self.actions_r[action], **kwargs)
                clipping = False
                if clipping:
                    k = 3
                    low = min(
                        max(distr.mean() - k * distr.std(), 0),
                        self.nS - 1
                    )
                    high = max(
                        min(distr.mean() + k * distr.std(), self.nS),
                        1
                    )
                    feasible_range = range(floor(low), ceil(high))
                else:
                    feasible_range = range(self.nS)
                for new_state in feasible_range:
                    prob = 0
                    if new_state == self.nS - 1:
                        # probability of landing on new_state or anything above
                        prob = 1 - distr.cdf(new_state - 1)
                    else:
                        prob = distr.pmf(new_state)
                    done = False
                    reward = self._reward(state, self.actions_r[action])

                    outcome = (prob, new_state, reward, done)
                    self.P_1_step[state][action].append(outcome)
                    self.P_lookup_1_step[state][action][new_state] = (prob, reward)
                    
        save_pickle(self.P_1_step, file_name)
        save_pickle(self.P_lookup_1_step, file_name_lookup)
        return self.P_1_step

    def _set_transition_probabilities(self):
        self._set_transition_probabilities_1_step(**self.kwargs)
        iterations = self.action_frequency
        file_name = self._dynamics_file_name(iterations=iterations, **self.kwargs)
        try:
            self.P = load_pickle(file_name)
            print('Loaded transition_probs')
            return self.P
        except:
            self.P = []

        self.P = [[ [] for j in range(self.nA)] for i in range(self.nS)]
        
        self.P_lookup_prev = copy.deepcopy(self.P_lookup_1_step)
        self.P_lookups_next = [[[set() for k in range(self.nS)] for j in range(self.nA)] for i in range(self.nS)]
        
        for iteration in range(iterations - 1):
            self.P_lookups_next = [[[set() for k in range(self.nS)] for j in range(self.nA)] for i in range(self.nS)]
            for start_state in range(self.nS):
                for action in range(self.nA):
                    for intermediate_state in range(self.nS):
                        for new_state in range(self.nS):
                            prob_start_intermediate = self.P_lookup_prev[start_state][action][intermediate_state][0]
                            prob_intermediate_new = self.P_lookup_1_step[intermediate_state][action][new_state][0]
                            reward_start_intermediate = self.P_lookup_prev[start_state][action][intermediate_state][1]
                            reward_intermediate_new = self.P_lookup_1_step[intermediate_state][action][new_state][1]
                            self.P_lookups_next[start_state][action][new_state].add(
                                (prob_start_intermediate * prob_intermediate_new,
                                 reward_start_intermediate + reward_intermediate_new)
                            )

            # sum up over all intermediate states
            self.P_lookup_prev = [[[None for k in range(self.nS)] for j in range(self.nA)] for i in range(self.nS)]
            for start_state in range(self.nS):
                for action in range(self.nA):
                    for new_state in range(self.nS):
                        outcomes = self.P_lookups_next[start_state][action][new_state] # {(prob, reward)}
                        prob = sum([outcome[0] for outcome in outcomes])
                        if prob > 0:
                            reward = sum(outcome[0] * outcome[1] for outcome in outcomes) / prob
                        else:
                            reward = 0
                        self.P_lookup_prev[start_state][action][new_state] = (prob, reward)

        for state in range(self.nS):
            for action in range(self.nA):
                for new_state in range(self.nS):
                    prob, reward = self.P_lookup_prev[state][action][new_state]
                    done = False
                    outcome = (prob, new_state, reward, done)
                    self.P[state][action].append(outcome)
                    
        save_pickle(self.P, file_name)
        return self.P

    def create_iterated_env(self, iterations=4):
        self._set_iterated_probabilities(iterations=iterations)
        new_env = copy.copy(self)
        new_env.P = new_env.P_iterated
        new_env._single_step = new_env.step
        def macro_step(env, action):
            for i in range(iterations):
                result = env._single_step(action)
            return result
        new_env.macro_step = macro_step

    def _param_string(self, action_frequency, **kwargs):
        return f'distr_family={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},num_states={self.nS},num_actions={self.nA},dynamics={self.dynamics},action_frequency={action_frequency},custom={self.kwargs}'
        
    def _dynamics_file_name(self, iterations, lookup=False, **kwargs):
        param_str = self._param_string(iterations, **kwargs)
        file_name = f'../results/env=({param_str})/transition_dynamics{"_lookup" if lookup else ""}.pickle'
        return file_name
