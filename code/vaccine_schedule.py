import numpy as np
import logging


def vaccine_schedule(horizon, milestones, R_0):
    '''
    `horizon`: number of time steps
    `milestones`: [(time, R_0), ...]


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
    
    schedule = np.ones(horizon) * R_0

    milestones = [(0, R_0)] + milestones + [(horizon, milestones[-1][1])]
    milestones.sort()

    last_R = R_0

    for i in range(len(milestones) - 1):
        R = milestones[i][1]
        t_start = milestones[i][0]
        t_end = milestones[i+1][0]
        for t in range(t_start, t_end):
            if t < len(schedule):
                schedule[t] = R
            else:
                logging.warning(f'Vaccination milestones ([{t_start}, {t_end})) extend outside time horizon ({horizon}).')
                
    return schedule


