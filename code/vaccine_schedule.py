import numpy as np
import logging


def vaccine_schedule(horizon, milestones):
    '''
    `horizon`: number of time steps
    `milestones`: [(time, reduction_factor), ...]

>>> horizon = 24
>>> R_0 = 4
>>> milestones = [
... (4, 3.5),
... (8, 3),
... (12, 2.5),
... (16, 2),
... (20, 1.5)
... ]
>>> vaccine_schedule.vaccine_schedule
<function vaccine_schedule at 0x7fd517402ca0>
>>> sched = vaccine_schedule.vaccine_schedule(horizon, milestones, R_0)
>>> sched
array([4. , 4. , 4. , 4. , 3.5, 3.5, 3.5, 3.5, 3. , 3. , 3. , 3. , 2.5,
       2.5, 2.5, 2.5, 2. , 2. , 2. , 2. , 1.5, 1.5, 1.5, 1.5])
>>> sched.shape
(24,)
>>> milestones = [
... (0, 4),
... (2, 3.5),
... (6, 3),
... (10, 2.5),
... (14, 2),
... (18, 1.5),
... (24, 1.0)
... ]
>>> sched = vaccine_schedule.vaccine_schedule(horizon, milestones, R_0)
>>> sched
array([4. , 4. , 3.5, 3.5, 3.5, 3.5, 3. , 3. , 3. , 3. , 2.5, 2.5, 2.5,
       2.5, 2. , 2. , 2. , 2. , 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    '''

    # TODO: add linear interpolation.
    
    schedule = np.ones(horizon)

    milestones = [(0, 1.0)] + milestones + [(horizon, milestones[-1][1])]
    milestones.sort()

    for i in range(len(milestones) - 1):
        factor_start = milestones[i][1]
        factor_end = milestones[i+1][1]
        t_start = round(milestones[i][0])
        t_end = round(milestones[i+1][0])
        for t in range(t_start, t_end):
            if t < len(schedule):
                ratio = (t - t_start) / (t_end - t_start)
                factor = factor_start * (1 - ratio) + factor_end * ratio
                schedule[t] = factor
            else:
                logging.warning(f'Vaccination milestones ([{t_start}, {t_end})) extend outside time horizon ({horizon}).')
                
    return schedule


def schedule_custom(horizon):
    milestones = [
        (horizon * 0.2, 1 - 0.1),
        (horizon * 0.4, 1 - 0.25),
        (horizon * 0.5, 1 - 0.5),
        (horizon * 0.6, 1 - 0.75),
        (horizon * 0.7, 1 - 0.85),
        (horizon * 0.8, 1 - 0.9),
        (horizon * 0.9, 1 - 0.95),
        (horizon * 1.0, 1 - 1.0)
    ]
    return vaccine_schedule(horizon, milestones)


def schedule_even(horizon, num_increments, final_susceptible):
    '''
    >>> from vaccine_schedule import schedule_even
    >>> schedule_even(24, 8, 0.0)
    array([1.   , 1.   , 1.   , 0.875, 0.875, 0.875, 0.75 , 0.75 , 0.75 ,
           0.625, 0.625, 0.625, 0.5  , 0.5  , 0.5  , 0.375, 0.375, 0.375,
           0.25 , 0.25 , 0.25 , 0.125, 0.125, 0.125])
    >>> len(schedule_even(24, 8, 0.0))
    24
    '''
    increment_duration = horizon / num_increments
    increment_reduction = (1 - final_susceptible) / (num_increments - 1)
    times = np.arange(0, horizon, increment_duration)
    factors = np.arange(1.0, final_susceptible - increment_reduction, -increment_reduction)
    milestones = list(zip(times, factors))
    return vaccine_schedule(horizon, milestones)


def schedule_none(horizon):
    '''
    >>> schedule_none(24)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1.])
    '''
    milestones = [(1, 1.0), (horizon, 1.0)]
    return vaccine_schedule(horizon, milestones)


def schedule_custom_delay(horizon, delay_horizon):
    none_horizon = delay_horizon
    even_horizon = horizon - delay_horizon

    schedule_1 = schedule_none(none_horizon)
    schedule_2 = schedule_custom(even_horizon)

    return np.hstack((schedule_1, schedule_2))


def schedule_even_delay(horizon, delay_horizon, num_increments, final_susceptible):
    '''
>>> from vaccine_schedule import schedule_even_delay
>>> schedule_even_delay(24, 16, 4, 0)
array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 6.66666667e-01, 6.66666667e-01,
       3.33333333e-01, 3.33333333e-01, 2.22044605e-16, 2.22044605e-16])
    '''
    none_horizon = delay_horizon
    even_horizon = horizon - delay_horizon

    schedule_1 = schedule_none(none_horizon)
    schedule_2 = schedule_even(even_horizon, num_increments, final_susceptible)

    return np.hstack((schedule_1, schedule_2))
