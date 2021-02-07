from math import floor, ceil, prod
from pdb import set_trace as b
import itertools
import copy
import sys

import numpy as np
from scipy.stats import poisson, nbinom, rv_discrete
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm

from utils import save_pickle, load_pickle, cap_distribution
from scenarios import US, Test, Test2
from vaccine_schedule import schedule_even_delay, schedule_custom_delay, schedule_sigmoid, schedule_none

class PandemicEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    days_per_step = 4
    
    def __init__(self,
                 num_population=10000,
                 hospital_capacity_proportion=0.01,
                 initial_fraction_infected=0.001,
                 R_0=2.5,
                 imported_cases_per_step=0.5,
                 power=2,
                 scale_factor=100,
                 cost_per_case_scale_factor=1.0,
                 distr_family='nbinom',
                 dynamics='SIS',
                 init_transition_probs=False,
                 horizon=np.inf,
                 action_frequency=1,
                 scenario=US,
                 vaccine_start=0,
                 vaccine_final_susceptible=0,
                 vaccine_schedule='none',
                 results_dir='../results',
                 **kwargs):
        super(PandemicEnv, self).__init__()
        
        # States
        self.num_population = num_population
        self.max_infected = int(num_population * hospital_capacity_proportion)
        self.initial_num_infected = int(num_population * initial_fraction_infected)

        # Transitions
        self.R_0 = R_0
        self.imported_cases_per_step = imported_cases_per_step
        self.distr_family = distr_family
        self.dynamics = dynamics
        self.vaccine_schedule = vaccine_schedule
        
        # Actions
        self.action_frequency = action_frequency
        
        # Reward
        ## Cost of Lockdown
        self.power = power
        self.scale_factor = scale_factor

        ## Cost of cases
        self.scenario = scenario
        self.cost_per_case = cost_per_case_scale_factor * self.scenario.cost_per_case
        
        # Horizon
        self.horizon = int(horizon) if horizon < np.inf else horizon
        self.horizon_effective = ceil(horizon / action_frequency) if horizon < np.inf else horizon


        self.kwargs = kwargs
        
        self.results_dir = results_dir
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Here using discrete actions

        
        ### Action space
        #   in increments of 0.5 up to R_0
        self.actions_r = np.array(
            # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25] + \
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25] + \
            list(np.arange(1.5, self.R_0, 0.5)) + \
            [self.R_0]
        )
        self.contact_factor = self.actions_r / self.R_0
        self.nA = self.actions_r.shape[0]
        self.action_space = spaces.Discrete(self.nA)

        
        ### State space
        if self.track_immunity():  # SIR
            # State: (num_susceptible, num_infected)
            self.observation_space = spaces.Box(low=np.array([0, 0]),
                                                high=np.array([
                                                    self.num_population,
                                                    self.max_infected
                                                ]),
                                                shape=(2,), dtype=np.uint16)
        else:  # SIS
            self.observation_space = spaces.Box(low=np.array([self.num_population, 0]),
                                                high=np.array([
                                                    self.num_population,
                                                    self.max_infected
                                                ]),
                                                shape=(2,), dtype=np.uint16)

        # Observation: (num_susceptible, num_infected)
        self.states = list(
            itertools.product(
                range(self.observation_space.low[0], self.observation_space.high[0] + 1),
                range(self.observation_space.low[1], self.observation_space.high[1] + 1),
            ))

        self._state_to_idx = None
        
        self.nS = len(self.states)

        ### Vaccination / Transmissibility:
        #   Transmissibility goes down over time due to vaccinations
        self.vaccine_final_susceptible = vaccine_final_susceptible
        # self.vaccine_start_idx = round(self.horizon_effective * vaccine_start)
        time_til_half_vaccinated = 32
        vaccination_rate = 0.1

        num_steps = 4
        # vaccine_schedule = schedule_custom_delay(self.horizon_effective + 1, self.vaccine_start_idx)   # TODO: make this horizon, not horizon + 1
        '''vaccine_schedule = schedule_even_delay(
            self.horizon_effective + 1,
            self.vaccine_start_idx,
            8,
            self.vaccine_final_susceptible)   # TODO: make this horizon, not horizon + 1'''

        # Vaccine roll-out. Then, 100% immune after horizon is over
        if self.vaccine_schedule == 'sigmoid':
            self.transmissibility_schedule = schedule_sigmoid(
                self.horizon_effective + 1,
                time_til_half_vaccinated,
                vaccination_rate,
                1 - self.vaccine_final_susceptible)
            
        elif self.vaccine_schedule == 'none':
            self.transmissibility_schedule = schedule_none(self.horizon_effective + 1)
                
        ### Infectiousness:
        #   can go down over time due to better treatments or vaccines
        #   This could go down due to vaccination
        #   Right now, being optimistic with the vaccination and assuming those people just never become infectious at all
        #   To be less optimistic, move vaccination from transmissibility and into infectiousness
        #    (i.e., those people can still get infected, they're just a bit less infectious)
        self.infectious_schedule = [1 for time_idx in range(self.horizon_effective + 1)] if self.horizon < np.inf else None
        
        ### Contact rate:
        #   can go down over time: people independently learn to limit contact in low-cost ways
        #   (e.g. adoption of masks, safer business practices, etc.)
        #   Gets multiplied by the contact reduction the policymaker sets, but policymaker does not get charged for it 
        self.contact_rate_schedule = [1 for time_idx in range(self.horizon_effective + 1)] if self.horizon < np.inf else None
        
        self.P = None
        #if init_transition_probs:
        #    self._set_transition_probabilities()
            
        self.state = self.initial_num_infected
        self.done = 0
        self.reward = 0

        self.reset()

        # File name
        self.dynamics_param_str = self._param_string(self.action_frequency, **self.kwargs)
        self.reward_param_str = f'power={self.power},scale_factor={self.scale_factor},horizon={self.horizon}'
        self.file_name_prefix = f'{self.results_dir}/env=({self.dynamics_param_str})/reward=({self.reward_param_str})/'
        
        
    def track_immunity(self):
        return self.dynamics == 'SIR'

    
    def step(self, action):
        outcomes = self.transitions(self.state, action, self.time_idx)
        # outcomes = [(probs[i], feasible_range[i], reward, done) for i in range(len(feasible_range))]
        outcome_idxs = list(range(len(outcomes)))
        probs = [outcome[0] for outcome in outcomes]
        outcome_distr = rv_discrete(values=(outcome_idxs, probs))
        outcome_idx = outcome_distr.rvs()
        outcome = outcomes[outcome_idx]

        self.state = outcome[1]
        reward = outcome[2]
        self.done = outcome[3]
        
        obs = self.state
        done = self.done
        self.time_idx += 1

        return obs, reward, done, {}

    def step_macro(self, action):
        for i in range(self.action_frequency):
            result = self.step(action)
        # TODO: accumulate results of each individual step into result object
        # i.e. list of observations, sum of rewards, any done, list of info?
        return result
    
    def reset(self):
        # Reset the state of the environment to an initial state
        num_infected = self.initial_num_infected
        if self.track_immunity():
            num_susceptible = self.num_population - self.initial_num_infected
        else:
            num_susceptible = self.num_population

        state_vector = (num_susceptible, num_infected)
        self.time_idx = 0
        
        state = self.state_obj_to_idx(state_vector)
        obs = state
        self.state = state
        
        return obs
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.state)

    def _unpack_state(self, packed_state):
        raise Exception('Not implemented')
        # num_infected = packed_state
        # num_susceptible = self.max_infected
        # return (num_susceptible, num_infected)
        
    def _reward(self, state, action, time_idx=None, **kwargs):        
        # Do not allow exceeding hospital capacity
        expected_new_infected = self._expected_new_infected(state, action)
        if expected_new_infected > self.max_infected:
            return -sys.float_info.max / 1e50  # use instead of -np.inf, to avoid `nan` issue when multiplying by 0
        # TODO: replace with:
        #  if Prob(actual_new_cases > self.max_infected) > .05:  return -np.inf
        
        return -self._cost_of_infections(state, **kwargs) \
               -self._cost_of_contact_factor(action, **kwargs)
             # -self._cost_of_r_linear(r, self.R_0, self.R_0, **kwargs)
    
    def _cost_of_contact_factor(self, action, **kwargs):
        '''
        `action` in [0..self.nA]
        '''
        # `factor_contact` \in (0, 1]
        factor_contact = self.contact_factor[action] # What percentage of contact are we allowing
        
        baseline = 1/(1 ** self.power)
        actual = 1/(factor_contact ** self.power)

        cost_of_full_lockdown = self.days_per_step * self.scenario.gdp_per_day * self.scenario.fraction_gdp_lost
        
        # cost_to_keep_half_home / (1/((num_population/4)**power) - 1/(R_0 ** power))
        if factor_contact >= 1:
            return 0
        else:
            return (actual - baseline) * self.scale_factor / (self.R_0 ** self.power) * cost_of_full_lockdown
            # put back in the factor of self.R_0 ** self.power that's been divided out
            # by dividing the denominator of both baseline and actual by R_0
            # (was previously measured on the scale of 0 to R_0; now on the scale of 0 to 1
    
    def _cost_of_r(self, r, **kwargs):
        baseline = 1/(self.R_0 ** self.power)
        actual = 1/(r ** self.power)
        base_population = 1000 # Population the cost function was originally designed for
        scaling = self.scale_factor * self.num_population / base_population

        # cost_to_keep_half_home / (1/((max_infected/4)**power) - 1/(R_0 ** power))
        if r >= self.R_0:
            return 0
        else:
            return (actual - baseline) * self.scale_factor / (self.R_0 ** self.power)
            # put back in the factor of self.R_0 ** self.power that's been divided out
            # by dividing the denominator of both baseline and actual by R_0
            # (was previously measured on the scale of 0 to R_0; now on the scale of 0 to 1

            
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
        raise Exception('Needs to be re-implemented to take `action` argument')
        cost_of_full_lockdown = self.days_per_step * self.scenario.gdp_per_day * self.scenario.fraction_gdp_lost
        r = max(r, 0) # cannot physically make r < 0
        fraction_locked_down = (R_0_orig - r) / R_0_orig
        fraction_for_free = (R_0_orig - R_0_new) / R_0_orig
        net_cost = cost_of_full_lockdown * (fraction_locked_down - fraction_for_free)
        return max(net_cost, 0) # cannot incur negative cost (i.e. make money) by making r > R_0_new

    def _cost_of_infections(self, state, **kwargs):
        num_susceptible, num_infected = self.states[state]
        return max(num_infected, 0) * self.scenario.cost_per_case

    def _expected_new_infected(self, state, action, time_idx=None, **kwargs):
        num_susceptible, num_infected = self.states[state]
        R_t = self.R_t(action, time_idx, num_susceptible)
        expected_new_infected = (num_infected * R_t) + self.imported_cases_per_step
        return expected_new_infected

    def _new_infected_distribution(self, state, action, time_idx=None, **kwargs):
        # distr_family: 'poisson' or 'nbinom' or 'deterministic'
        lam = self._expected_new_infected(state, action, time_idx, **kwargs)
        num_susceptible, num_infected = self.states[state]
        
        if self.distr_family == 'poisson':
            distr = poisson(lam)
        elif self.distr_family == 'nbinom':
            r = 100000000000000.0
            # r = 0.17
            p = lam / (r + lam)
            distr = nbinom(r, 1 - p)
        elif self.distr_family == 'deterministic':
            distr = rv_discrete(values=([int(lam)], [1.0]))

        max_infectable = min(num_susceptible, self.max_infected)
        feasible_range = range(max_infectable + 1)
        return cap_distribution(distr, feasible_range)

    
    def R_t(self, action, time_idx, num_susceptible):
        factor_transmissibility = self.transmissibility_schedule[time_idx] if time_idx else 1
        factor_contact = (self.contact_rate_schedule[time_idx] if time_idx else 1) * self.contact_factor[action]
        factor_infectious_period = self.infectious_schedule[time_idx] if time_idx else 1
        fraction_susceptible = num_susceptible / self.num_population
        
        R_t = self.R_0 * factor_transmissibility * factor_contact * factor_infectious_period * fraction_susceptible
        return R_t

    
    def transitions(self, state, action, time_idx=None):
        
        reward = self._reward(state, action, time_idx)
        distr = self._new_infected_distribution(state, action, time_idx)
        num_susceptible, num_infected = self.states[state]
        max_infected = min(self.max_infected, num_susceptible)
        feasible_num_infected_range = list(range(max_infected + 1))
        probs = distr.pmf(feasible_num_infected_range)
        done = False

        #if np.isnan(reward) or reward == -np.inf:
        #    b()
                    
        if self.track_immunity():
            outcomes = [(
                probs[i],
                self.state_obj_to_idx((num_susceptible - new_num_infected, new_num_infected)), # Reduce number susceptible by number new infected
                reward,
                done
            ) for i, new_num_infected in enumerate(feasible_num_infected_range)]
        else:
            outcomes = [(
                probs[i],
                self.state_obj_to_idx((num_susceptible, new_num_infected)), # Keep same number susceptible as before
                reward,
                done
            ) for i, new_num_infected in enumerate(feasible_num_infected_range)]
        return outcomes

    
    def _set_transition_probabilities_1_step(self, **kwargs):
        # TODO: switch to contact-rate actions
        raise Exception('Not implemented (_set_transition_probabilities_1_step() must be updated to use contact-rate actions rather than R actions)')

        file_name = self._dynamics_file_name(iterations=1)
        file_name_lookup = self._dynamics_file_name(iterations=1, lookup=True)
        try:
            print('Loading 1-step transition probs...')
            self.P_1_step = load_pickle(file_name)
            self.P_lookup_1_step = load_pickle(file_name_lookup)
            print('Loaded 1-step transition_probs')
            return self.P_1_step
        except:
            self.P_1_step = []

        print('Generating 1-step transition probabilities:')
        self.P_1_step = np.empty((self.nS, self.nA), dtype=list)
        self.P_lookup_1_step = np.empty((self.nS, self.nA, self.nS), dtype=list)

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
                    prob = distr.pmf(new_state)
                    done = False
                    reward = self._reward(state, self.actions_r[action])

                    outcome = (prob, new_state, reward, done)
                    if not self.P_1_step[state, action]:
                        self.P_1_step[state, action] = []
                    self.P_1_step[state, action].append(outcome)
                    self.P_lookup_1_step[state, action, new_state] = (prob, reward)
        print('Saving 1-step transition probabilities...')
        save_pickle(self.P_1_step, file_name)
        save_pickle(self.P_lookup_1_step, file_name_lookup)
        print('Saved')
        return self.P_1_step

    def _set_transition_probabilities(self):
        self._set_transition_probabilities_1_step(**self.kwargs)
        iterations = self.action_frequency
        file_name = self._dynamics_file_name(iterations=iterations, **self.kwargs)
        try:
            print('Loading multi-step transition probs')
            self.P = load_pickle(file_name)
            print(f'Loaded multi-step transition_probs ({iterations}-step)')
            return self.P
        except:
            self.P = []

        print('Allocating multi-step transition table (empty)...')
        self.P = np.empty((self.nS, self.nA), dtype=list)

        print('Copying 1-step lookup table...')
        self.P_lookup_prev = copy.deepcopy(self.P_lookup_1_step)

        print('Allocating iterative lookup table...')
        self.P_lookups_next = np.empty((self.nS, self.nA, self.nS), dtype=set)

        print(f'Iterating multi-step transitions ({iterations}-step)')
        for iteration in range(iterations - 1):
            print(f'Iteration {iteration}. Branching out.')
            self.P_lookups_next.fill(None)
            for start_state in tqdm(range(self.nS)):
                for action in range(self.nA):
                    for intermediate_state in range(self.nS):
                        for new_state in range(self.nS):
                            prob_start_intermediate = self.P_lookup_prev[start_state, action, intermediate_state][0]
                            prob_intermediate_new = self.P_lookup_1_step[intermediate_state, action, new_state][0]
                            reward_start_intermediate = self.P_lookup_prev[start_state, action, intermediate_state][1]
                            reward_intermediate_new = self.P_lookup_1_step[intermediate_state, action, new_state][1]
                            if not self.P_lookups_next[start_state, action, new_state]:
                                self.P_lookups_next[start_state, action, new_state] = set()
                            self.P_lookups_next[start_state, action, new_state].add(
                                (prob_start_intermediate * prob_intermediate_new,
                                 reward_start_intermediate + reward_intermediate_new)
                            )

            # sum up over all intermediate states
            print(f'Iteration {iteration}. Summing up.')
            self.P_lookup_prev.fill(None)
            for start_state in tqdm(range(self.nS)):
                for action in range(self.nA):
                    for new_state in range(self.nS):
                        outcomes = self.P_lookups_next[start_state, action, new_state] # {(prob, reward)}
                        prob = sum([outcome[0] for outcome in outcomes])
                        if prob > 0:
                            reward = sum(outcome[0] * outcome[1] for outcome in outcomes) / prob
                        else:
                            reward = 0
                        self.P_lookup_prev[start_state, action, new_state] = (prob, reward)

        print(f'Converting multi-step lookup table to outcomes list')
        for state in tqdm(range(self.nS)):
            for action in range(self.nA):
                for new_state in range(self.nS):
                    # TODO: way to speed this up? Just by flattening the array in some efficient way, rather than 3x for loop?
                    # Of course, not even sure this is a slow part of the code...
                    prob, reward = self.P_lookup_prev[state, action, new_state]
                    done = False
                    outcome = (prob, new_state, reward, done)
                    if not self.P[state, action]:
                        self.P[state, action] = []
                    self.P[state, action].append(outcome)
                    
        save_pickle(self.P, file_name)
        return self.P

    def state_obj_to_idx(self, state_obj):
        # Could be instead:
        # if not self._state_to_idx:
        #     self._state_to_idx = {self.states[idx]: idx for idx in range(len(self.states))}
        # return self._state_to_idx[state_obj]

        # Test for equivalence:
        # state_objs = [(0, 1000), (0, 2000), (5000, 0), (1, 1000), (8000, 1500), (9500, 1000)]
        # state_objs = [(0, 1000), (0, 2000), (8000, 1500), (9500, 1000)]
        # for state_obj in state_objs:
        #     print(state_obj, self.state_obj_to_idx(state_obj), self._state_to_idx[state_obj])
        #     assert (env.state_obj_to_idx(state_obj) == env._state_to_idx[state_obj])

        num_susceptible, num_infected = state_obj
        if self.track_immunity():
            num_infected_range = (self.observation_space.high[1] - self.observation_space.low[1] + 1)
            idx = num_infected_range * num_susceptible + num_infected
        else:
            idx = num_infected
            
        return idx
    
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
        # too long:
        # return f'R_0={self.R_0},distr_family={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},num_states={self.nS},num_actions={self.nA},dynamics={self.dynamics},action_frequency={action_frequency},vaccine_start_idx={self.vaccine_start_idx},vaccine_final_susceptible={self.vaccine_final_susceptible},custom={self.kwargs}'
        
        return f'num_population={self.num_population},R_0={self.R_0},distr_family={self.distr_family},imported_cases_per_step={self.imported_cases_per_step},dynamics={self.dynamics},custom={self.kwargs},vaccine_schedule={self.vaccine_schedule}'
        
        
    def _dynamics_file_name(self, iterations, lookup=False, **kwargs):
        param_str = self._param_string(iterations, **kwargs)
        file_name = f'../results/env=({param_str})/transition_dynamics{"_lookup" if lookup else ""}.pickle'
        return file_name
